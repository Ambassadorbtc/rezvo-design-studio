"""
REZVO DESIGN STUDIO — Server
FastAPI backend that proxies Anthropic API calls and serves the frontend.
Run: python server.py
Access: http://YOUR_IP:8500
"""

import os
import json
import base64
import uuid
import httpx
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Rezvo Design Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
PROJECTS_FILE = Path("projects.json")

def load_projects():
    if PROJECTS_FILE.exists():
        return json.loads(PROJECTS_FILE.read_text())
    return []

def save_projects(projects):
    PROJECTS_FILE.write_text(json.dumps(projects, indent=2))


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload a screenshot and return its ID + base64 data URL."""
    ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    file_id = f"{uuid.uuid4().hex[:12]}.{ext}"
    content = await file.read()
    
    # Save to disk
    path = UPLOAD_DIR / file_id
    path.write_bytes(content)
    
    # Return base64 for preview + API use
    b64 = base64.b64encode(content).decode()
    media_type = f"image/{ext}" if ext in ("png", "jpg", "jpeg", "gif", "webp") else "image/png"
    data_url = f"data:{media_type};base64,{b64}"
    
    return {"id": file_id, "data_url": data_url, "media_type": media_type, "base64": b64}


@app.post("/api/generate")
async def generate_design(
    api_key: str = Form(...),
    prompt: str = Form(""),
    image_base64: str = Form(""),
    image_media_type: str = Form("image/png"),
    device: str = Form("tablet"),
):
    """Send screenshot + prompt to Anthropic and return generated HTML."""
    
    if not api_key:
        raise HTTPException(400, "API key required")
    
    DEVICE_SIZES = {
        "mobile": (390, 844),
        "tablet": (1024, 768),
        "desktop": (1440, 900),
    }
    w, h = DEVICE_SIZES.get(device, (1024, 768))
    
    user_content = []
    
    if image_base64:
        user_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image_media_type,
                "data": image_base64,
            }
        })
    
    system_prompt = f"""You are an expert UI designer and frontend developer specializing in pixel-perfect replication.

CRITICAL RULES:
1. Output ONLY raw HTML. No markdown fences. No explanations. No backticks. Just the HTML.
2. Use Tailwind CSS via CDN: <script src="https://cdn.tailwindcss.com"></script>
3. EXTRACT exact hex colors from the screenshot. Do NOT use Tailwind default palette colors.
4. EXTRACT exact spacing, padding, border-radius from the screenshot. Use pixel values in style attributes when Tailwind classes don't match.
5. Match typography exactly - font family, weight, size.
6. Target viewport: {w}x{h}px ({device}).
7. Include realistic placeholder content matching the screenshot.
8. Make it responsive within the target viewport.
9. Use Google Fonts CDN if specific fonts are needed.
10. The output must be a COMPLETE standalone HTML file with <!DOCTYPE html> through </html>."""

    user_content.append({
        "type": "text",
        "text": f"""Replicate this UI design EXACTLY as a single HTML file.

Target: {device} ({w}x{h}px)

{prompt if prompt else "Replicate every element, color, spacing, and layout from the screenshot with pixel-perfect accuracy."}

Remember: Output ONLY the raw HTML code. No markdown, no backticks, no explanation."""
    })
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 16000,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_content}],
                },
            )
            
            if resp.status_code != 200:
                error_data = resp.json()
                raise HTTPException(resp.status_code, f"Anthropic API error: {error_data.get('error', {}).get('message', resp.text)}")
            
            data = resp.json()
            html = data["content"][0]["text"]
            
            # Strip markdown fences if model wrapped them
            html = html.strip()
            if html.startswith("```"):
                html = html.split("\n", 1)[1] if "\n" in html else html[3:]
            if html.endswith("```"):
                html = html[:-3]
            html = html.strip()
            
            return {"html": html, "model": data.get("model"), "usage": data.get("usage")}
            
        except httpx.TimeoutException:
            raise HTTPException(504, "Request timed out — try a simpler prompt")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            raise HTTPException(500, str(e))


@app.get("/api/projects")
async def get_projects():
    return load_projects()

@app.post("/api/projects")
async def save_projects_endpoint(projects: list = []):
    save_projects(projects)
    return {"ok": True}


# Serve frontend
@app.get("/")
async def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    print("\n  🎨 REZVO DESIGN STUDIO")
    print("  ═══════════════════════")
    print("  http://0.0.0.0:8500\n")
    uvicorn.run(app, host="0.0.0.0", port=8500)
