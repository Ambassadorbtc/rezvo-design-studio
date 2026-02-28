"""
REZVO DESIGN STUDIO — Server (Multi-Provider)
Supports: Anthropic (Claude), xAI (Grok), Google Gemini, OpenAI (GPT-4o)
"""

import os, json, base64, uuid, httpx
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Rezvo Design Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

DEVICE_SIZES = {"mobile": (390, 844), "tablet": (1024, 768), "desktop": (1440, 900)}

SYSTEM_PROMPT = """You are a desktop scanner that converts screenshots into pixel-perfect HTML replicas.

SCANNER MODE — CRITICAL RULES:

1. Output ONLY raw HTML. No markdown fences. No explanations. No backticks.
2. Use Tailwind CSS via CDN: <script src="https://cdn.tailwindcss.com"></script>
3. The ORIGINAL SCREENSHOT is available at this URL: {{IMAGE_URL}}
4. For EVERY photo, food image, product image, avatar, or visual element in the screenshot:
   - Use a <div> with background-image: url('{{IMAGE_URL}}') 
   - Use background-size and background-position (in percentages) to CROP the exact region from the original screenshot
   - This extracts the ACTUAL PIXELS from the original — not placeholders, not gradients, not stock images
   - Example: if a food photo is at roughly 15% from left, 55% from top of the screenshot:
     background-image: url('{{IMAGE_URL}}'); background-size: 400% 300%; background-position: 15% 55%;
   - Adjust background-size to control zoom level: larger % = more zoomed in on the region
5. EXTRACT exact hex colors from the screenshot. Do NOT use Tailwind default palette colors.
6. EXTRACT exact spacing, padding, margin, border-radius in pixels.
7. Match typography exactly — font family, weight, size, line-height.
8. Target viewport: {w}x{h}px ({device}).
9. Include ALL elements visible: every button, icon, text label, divider, badge, nav item.
10. Use inline SVG for icons that match the screenshot's icon style.
11. Complete standalone HTML: <!DOCTYPE html> through </html>.
12. Replicate EXACT layout proportions.
13. Match shadow depths, border widths, opacity levels.
14. Replicate any background behind the main card/window.
15. Use Google Fonts if specific fonts are needed."""


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
    return {"id": file_id, "data_url": f"data:{mt};base64,{b64}", "media_type": mt, "base64": b64, "public_url": f"/uploads/{file_id}"}


@app.post("/api/generate")
async def generate_design(
    api_key: str = Form(...), provider: str = Form("anthropic"), prompt: str = Form(""),
    image_base64: str = Form(""), image_media_type: str = Form("image/png"), device: str = Form("tablet"),
    image_url: str = Form(""),
):
    if not api_key: raise HTTPException(400, "API key required")
    if provider not in PROVIDERS: raise HTTPException(400, f"Unknown provider: {provider}")
    w, h = DEVICE_SIZES.get(device, (1024, 768))
    
    has_image = bool(image_base64)
    
    # Build system prompt — inject image URL for scanner mode
    if has_image and image_url:
        system = SYSTEM_PROMPT.format(w=w, h=h, device=device).replace("{{IMAGE_URL}}", image_url)
    else:
        # Fallback non-scanner system prompt
        system = SYSTEM_PROMPT.format(w=w, h=h, device=device).replace(
            "The ORIGINAL SCREENSHOT is available at this URL: {{IMAGE_URL}}\n4. For EVERY photo, food image, product image, avatar, or visual element in the screenshot:\n   - Use a <div> with background-image: url('{{IMAGE_URL}}') \n   - Use background-size and background-position (in percentages) to CROP the exact region from the original screenshot\n   - This extracts the ACTUAL PIXELS from the original — not placeholders, not gradients, not stock images\n   - Example: if a food photo is at roughly 15% from left, 55% from top of the screenshot:\n     background-image: url('{{IMAGE_URL}}'); background-size: 400% 300%; background-position: 15% 55%;\n   - Adjust background-size to control zoom level: larger % = more zoomed in on the region",
            "For images/photos, use gradient placeholder backgrounds matching the color tones visible in the reference."
        )
    
    if has_image:
        if not prompt or prompt.lower().strip() in ('copy this', 'replicate this', 'clone this', 'copy', 'replicate', 'clone', 'recreate this', 'rebuild this', 'make this', 'scan this', 'scan'):
            user_text = f"""SCAN this screenshot into pixel-perfect HTML.
Target: {device} ({w}x{h}px)

For every photo/image element visible in the screenshot, use background-image with the original screenshot URL and background-position to crop the EXACT pixels from the source image. This is critical — no placeholders, no gradients, no stock images. Extract the real pixels.

Reproduce EVERY element: headers, buttons, badges, icons, text labels, dividers, navigation, images. Missing any element is a failure.

Output ONLY the raw HTML code."""
        else:
            user_text = f"""SCAN this screenshot into HTML with these modifications:

{prompt}

Target: {device} ({w}x{h}px)

For every photo/image element, crop actual pixels from the original screenshot using background-image + background-position. Apply the modifications described above.

Output ONLY the raw HTML code."""
    else:
        user_text = f"""Create this UI as a single HTML file:

{prompt or 'Create a professional UI design.'}

Target: {device} ({w}x{h}px)

Output ONLY the raw HTML code."""
    try:
        result = await PROVIDERS[provider](api_key, system, user_text, image_base64, image_media_type)
        result["provider"] = provider
        
        # Post-process: replace server-relative image URL with base64 data URL for self-contained HTML
        if has_image and image_url and image_base64:
            data_url = f"data:{image_media_type};base64,{image_base64}"
            result["html"] = result["html"].replace(image_url, data_url)
        
        return result
    except httpx.TimeoutException:
        raise HTTPException(504, "Request timed out")
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/")
async def index():
    return FileResponse("static/index.html")


PREVIEWS_DIR = Path("previews")
PREVIEWS_DIR.mkdir(exist_ok=True)

@app.post("/api/preview")
async def save_preview(html: str = Form(...), name: str = Form("preview")):
    """Save HTML and return a unique preview URL."""
    preview_id = uuid.uuid4().hex[:10]
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()[:50] or "preview"
    (PREVIEWS_DIR / f"{preview_id}.html").write_text(html, encoding="utf-8")
    return {"id": preview_id, "url": f"/preview/{preview_id}", "name": safe_name}

@app.get("/preview/{preview_id}")
async def get_preview(preview_id: str):
    """Serve a saved preview."""
    path = PREVIEWS_DIR / f"{preview_id}.html"
    if not path.exists():
        raise HTTPException(404, "Preview not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("\n  🎨 REZVO DESIGN STUDIO")
    print("  Providers: Anthropic · xAI · Gemini · OpenAI")
    print("  http://0.0.0.0:8500\n")
    uvicorn.run(app, host="0.0.0.0", port=8500)
