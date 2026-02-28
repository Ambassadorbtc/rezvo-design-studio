"""
REZVO DESIGN STUDIO — Server (Multi-Provider)
Supports: Anthropic (Claude), xAI (Grok), Google Gemini, OpenAI (GPT-4o)
"""

import os, json, base64, uuid, httpx
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Rezvo Design Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DEVICE_SIZES = {"mobile": (390, 844), "tablet": (1024, 768), "desktop": (1440, 900)}

SYSTEM_PROMPT = """You are an expert UI designer and frontend developer specializing in pixel-perfect replication.

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


def clean_html(html: str) -> str:
    html = html.strip()
    if html.startswith("```html"): html = html[7:]
    elif html.startswith("```"): html = html.split("\n", 1)[1] if "\n" in html else html[3:]
    if html.endswith("```"): html = html[:-3]
    return html.strip()


async def call_anthropic(api_key, system, user_text, img_b64, img_type):
    user_content = []
    if img_b64:
        user_content.append({"type": "image", "source": {"type": "base64", "media_type": img_type, "data": img_b64}})
    user_content.append({"type": "text", "text": user_text})
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post("https://api.anthropic.com/v1/messages", headers={
            "Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01",
        }, json={"model": "claude-sonnet-4-20250514", "max_tokens": 16000, "system": system,
                 "messages": [{"role": "user", "content": user_content}]})
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Anthropic: {r.json().get('error',{}).get('message', r.text)}")
        d = r.json()
        return {"html": clean_html(d["content"][0]["text"]), "model": d.get("model"), "usage": d.get("usage")}


async def call_xai(api_key, system, user_text, img_b64, img_type):
    msgs = [{"role": "system", "content": system}]
    uc = []
    if img_b64:
        uc.append({"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64}"}})
    uc.append({"type": "text", "text": user_text})
    msgs.append({"role": "user", "content": uc})
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post("https://api.x.ai/v1/chat/completions", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
        }, json={"model": "grok-2-vision-latest", "max_tokens": 16000, "messages": msgs})
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"xAI: {r.json().get('error',{}).get('message', r.text)}")
        d = r.json()
        return {"html": clean_html(d["choices"][0]["message"]["content"]), "model": d.get("model"), "usage": d.get("usage")}


async def call_gemini(api_key, system, user_text, img_b64, img_type):
    parts = []
    if img_b64:
        parts.append({"inline_data": {"mime_type": img_type, "data": img_b64}})
    parts.append({"text": user_text})
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={"system_instruction": {"parts": [{"text": system}]}, "contents": [{"parts": parts}],
                  "generationConfig": {"maxOutputTokens": 16000}})
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Gemini: {r.json().get('error',{}).get('message', r.text)}")
        d = r.json()
        return {"html": clean_html(d["candidates"][0]["content"]["parts"][0]["text"]), "model": "gemini-2.0-flash", "usage": d.get("usageMetadata")}


async def call_openai(api_key, system, user_text, img_b64, img_type):
    msgs = [{"role": "system", "content": system}]
    uc = []
    if img_b64:
        uc.append({"type": "image_url", "image_url": {"url": f"data:{img_type};base64,{img_b64}", "detail": "high"}})
    uc.append({"type": "text", "text": user_text})
    msgs.append({"role": "user", "content": uc})
    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post("https://api.openai.com/v1/chat/completions", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
        }, json={"model": "gpt-4o", "max_tokens": 16000, "messages": msgs})
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"OpenAI: {r.json().get('error',{}).get('message', r.text)}")
        d = r.json()
        return {"html": clean_html(d["choices"][0]["message"]["content"]), "model": d.get("model"), "usage": d.get("usage")}


PROVIDERS = {"anthropic": call_anthropic, "xai": call_xai, "gemini": call_gemini, "openai": call_openai}


@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1] if "." in file.filename else "png"
    file_id = f"{uuid.uuid4().hex[:12]}.{ext}"
    content = await file.read()
    (UPLOAD_DIR / file_id).write_bytes(content)
    b64 = base64.b64encode(content).decode()
    mt = f"image/{ext}" if ext in ("png","jpg","jpeg","gif","webp") else "image/png"
    return {"id": file_id, "data_url": f"data:{mt};base64,{b64}", "media_type": mt, "base64": b64}


@app.post("/api/generate")
async def generate_design(
    api_key: str = Form(...), provider: str = Form("anthropic"), prompt: str = Form(""),
    image_base64: str = Form(""), image_media_type: str = Form("image/png"), device: str = Form("tablet"),
):
    if not api_key: raise HTTPException(400, "API key required")
    if provider not in PROVIDERS: raise HTTPException(400, f"Unknown provider: {provider}")
    w, h = DEVICE_SIZES.get(device, (1024, 768))
    system = SYSTEM_PROMPT.format(w=w, h=h, device=device)
    user_text = f"Replicate this UI design EXACTLY as a single HTML file.\nTarget: {device} ({w}x{h}px)\n{prompt or 'Replicate every element, color, spacing, and layout from the screenshot with pixel-perfect accuracy.'}\nRemember: Output ONLY the raw HTML code. No markdown, no backticks, no explanation."
    try:
        result = await PROVIDERS[provider](api_key, system, user_text, image_base64, image_media_type)
        result["provider"] = provider
        return result
    except httpx.TimeoutException:
        raise HTTPException(504, "Request timed out")
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/")
async def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    print("\n  🎨 REZVO DESIGN STUDIO")
    print("  Providers: Anthropic · xAI · Gemini · OpenAI")
    print("  http://0.0.0.0:8500\n")
    uvicorn.run(app, host="0.0.0.0", port=8500)
