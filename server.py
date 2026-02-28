"""
REZVO DESIGN STUDIO — Server v2 (Scanner Pipeline)
Two-pass generation: AI outputs crop markers → Pillow extracts real pixels.
"""

import os, json, base64, uuid, re, io, httpx
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Rezvo Design Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
PREVIEWS_DIR = Path("previews")
PREVIEWS_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

DEVICE_SIZES = {"mobile": (390, 844), "tablet": (1024, 768), "desktop": (1440, 900)}


# ═══════════════════════════════════════════════════
# IMAGE CROPPING ENGINE
# ═══════════════════════════════════════════════════

def crop_region_from_screenshot(img: Image.Image, x_pct: float, y_pct: float, w_pct: float, h_pct: float) -> str:
    """Crop a percentage-based region from the screenshot, return as base64 data URL."""
    iw, ih = img.size
    x = int(iw * x_pct / 100)
    y = int(ih * y_pct / 100)
    w = int(iw * w_pct / 100)
    h = int(ih * h_pct / 100)
    # Clamp to image bounds
    x = max(0, min(x, iw - 1))
    y = max(0, min(y, ih - 1))
    w = max(10, min(w, iw - x))
    h = max(10, min(h, ih - y))
    cropped = img.crop((x, y, x + w, y + h))
    buf = io.BytesIO()
    cropped.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def process_crop_markers(html: str, img: Image.Image) -> str:
    """Find data-crop attributes and replace with actual cropped image data URLs."""
    # Pattern: data-crop="x%,y%,w%,h%" on elements
    # We look for elements with data-crop and inject the cropped image
    
    # Pattern 1: <div ... data-crop="x,y,w,h" ...> → inject background-image
    def replace_div_crop(match):
        full_tag = match.group(0)
        coords = match.group(1)
        try:
            parts = [float(p.strip().replace('%', '')) for p in coords.split(',')]
            if len(parts) == 4:
                data_url = crop_region_from_screenshot(img, *parts)
                # Remove the data-crop attribute and inject background-image style
                new_tag = full_tag.replace(f'data-crop="{coords}"', '')
                if 'style="' in new_tag:
                    new_tag = new_tag.replace('style="', f'style="background-image:url(\'{data_url}\');background-size:cover;background-position:center;')
                else:
                    new_tag = new_tag.replace('>', f' style="background-image:url(\'{data_url}\');background-size:cover;background-position:center;">', 1)
                return new_tag
        except Exception:
            pass
        return full_tag
    
    html = re.sub(r'<div[^>]*data-crop="([^"]+)"[^>]*>', replace_div_crop, html)
    
    # Pattern 2: <img ... data-crop="x,y,w,h" ...> → replace src
    def replace_img_crop(match):
        full_tag = match.group(0)
        coords = match.group(1)
        try:
            parts = [float(p.strip().replace('%', '')) for p in coords.split(',')]
            if len(parts) == 4:
                data_url = crop_region_from_screenshot(img, *parts)
                # Replace src with cropped image
                new_tag = re.sub(r'src="[^"]*"', f'src="{data_url}"', full_tag)
                new_tag = new_tag.replace(f'data-crop="{coords}"', '')
                return new_tag
        except Exception:
            pass
        return full_tag
    
    html = re.sub(r'<img[^>]*data-crop="([^"]+)"[^>]*/?>', replace_img_crop, html)
    
    # Pattern 3: CROP(x,y,w,h) in CSS background-image or src values
    def replace_inline_crop(match):
        coords = match.group(1)
        try:
            parts = [float(p.strip().replace('%', '')) for p in coords.split(',')]
            if len(parts) == 4:
                return crop_region_from_screenshot(img, *parts)
        except Exception:
            pass
        return match.group(0)
    
    html = re.sub(r'CROP\(([^)]+)\)', replace_inline_crop, html)
    
    return html


# ═══════════════════════════════════════════════════
# SYSTEM PROMPT — SCANNER MODE
# ═══════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a desktop scanner that converts UI screenshots into pixel-perfect, functional HTML.

TARGET: {w}x{h}px ({device})

CRITICAL OUTPUT RULES:
1. Output ONLY raw HTML. No markdown, no backticks, no explanation.
2. Use Tailwind CSS: <script src="https://cdn.tailwindcss.com"></script>
3. Complete standalone HTML: <!DOCTYPE html> through </html>.

COLOR & LAYOUT RULES:
4. Extract EXACT hex colors from the screenshot. NEVER use Tailwind default colors like blue-500.
5. Match EXACT spacing, padding, margin, border-radius in pixels using style attributes.
6. Match EXACT typography — font family, weight, size, line-height.
7. Replicate EXACT layout proportions (e.g. 60/40 split).
8. Match shadow depths, border widths, opacity levels.
9. Use inline SVG icons matching the screenshot style.

IMAGE EXTRACTION — THIS IS CRITICAL:
10. For EVERY photograph, food image, product photo, or visual content in the screenshot, use this crop marker system:
    - Add data-crop="X,Y,W,H" attribute where X,Y,W,H are PERCENTAGES of the screenshot
    - X = percentage from left edge where the image starts
    - Y = percentage from top edge where the image starts  
    - W = percentage width of the image region
    - H = percentage height of the image region
    - Example: A food photo in the bottom-left quadrant might be: data-crop="5,55,20,15"
    - The server will use these coordinates to CROP actual pixels from the original screenshot
    - Use <div> elements with data-crop for background images
    - Use <img data-crop="..." src="placeholder"> for inline images

11. Be PRECISE with crop coordinates. Study where each image appears in the screenshot:
    - If the screenshot is 1000px wide and an image starts at 100px from left, X = 10
    - If it starts 500px from top of a 750px screenshot, Y = 66.7
    - Estimate the width and height of each image region as percentages

12. EVERY visible image MUST have a data-crop attribute. No grey boxes. No gradients. No placeholders.

COMPLETENESS:
13. Include ALL visible elements: buttons, badges, icons, nav items, text, dividers.
14. Make interactive elements look clickable (cursor, hover states via Tailwind).
15. Replicate any background/gradient behind the main UI card."""


def clean_html(html: str) -> str:
    html = html.strip()
    if html.startswith("```html"): html = html[7:]
    elif html.startswith("```"): html = html.split("\n", 1)[1] if "\n" in html else html[3:]
    if html.endswith("```"): html = html[:-3]
    return html.strip()


# ═══════════════════════════════════════════════════
# AI PROVIDERS
# ═══════════════════════════════════════════════════

async def call_anthropic(api_key, system, user_text, img_b64, img_type):
    user_content = []
    if img_b64:
        user_content.append({"type": "image", "source": {"type": "base64", "media_type": img_type, "data": img_b64}})
    user_content.append({"type": "text", "text": user_text})
    async with httpx.AsyncClient(timeout=180.0) as c:
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
    async with httpx.AsyncClient(timeout=180.0) as c:
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
    async with httpx.AsyncClient(timeout=180.0) as c:
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
    async with httpx.AsyncClient(timeout=180.0) as c:
        r = await c.post("https://api.openai.com/v1/chat/completions", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {api_key}",
        }, json={"model": "gpt-4o", "max_tokens": 16000, "messages": msgs})
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"OpenAI: {r.json().get('error',{}).get('message', r.text)}")
        d = r.json()
        return {"html": clean_html(d["choices"][0]["message"]["content"]), "model": d.get("model"), "usage": d.get("usage")}


PROVIDERS = {"anthropic": call_anthropic, "xai": call_xai, "gemini": call_gemini, "openai": call_openai}


# ═══════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════

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
    system = SYSTEM_PROMPT.format(w=w, h=h, device=device)
    
    # Build user prompt
    if has_image:
        is_simple = not prompt or prompt.lower().strip() in (
            'copy this', 'replicate this', 'clone this', 'copy', 'replicate', 
            'clone', 'recreate this', 'rebuild this', 'make this', 'scan this', 'scan'
        )
        if is_simple:
            user_text = f"""SCAN this screenshot into pixel-perfect HTML.
Target: {device} ({w}x{h}px)

IMPORTANT: For EVERY photograph/food image/product image visible in the screenshot, you MUST add a data-crop attribute with percentage coordinates. Study the screenshot carefully:
- Estimate where each image starts (X%, Y% from top-left)
- Estimate each image's width and height as percentages of the full screenshot
- The server will crop the actual pixels using these coordinates

Example: If a food photo occupies roughly the area from 10% left, 55% top, spanning 18% wide and 12% tall:
<div class="..." data-crop="10,55,18,12"></div>

Reproduce EVERY element. Every image MUST have data-crop. No grey boxes. No placeholder gradients.

Output ONLY raw HTML."""
        else:
            user_text = f"""SCAN this screenshot into HTML with these modifications:

{prompt}

Target: {device} ({w}x{h}px)

For EVERY photograph/image, add data-crop="X,Y,W,H" with percentage coordinates. The server crops real pixels.

Output ONLY raw HTML."""
    else:
        user_text = f"""Create this UI as a single HTML file:

{prompt or 'Create a professional UI design.'}

Target: {device} ({w}x{h}px)
Output ONLY raw HTML."""

    try:
        # ── PASS 1: AI generates HTML with crop markers ──
        result = await PROVIDERS[provider](api_key, system, user_text, image_base64, image_media_type)
        result["provider"] = provider
        
        # ── PASS 2: Server-side pixel cropping ──
        if has_image and image_base64:
            try:
                # Decode the original screenshot
                img_bytes = base64.b64decode(image_base64)
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Count crop markers before processing
                crop_count = len(re.findall(r'data-crop="[^"]+"', result["html"]))
                
                # Process all crop markers — extract real pixels
                if crop_count > 0:
                    result["html"] = process_crop_markers(result["html"], img)
                    result["crops_processed"] = crop_count
                else:
                    result["crops_processed"] = 0
                    
            except Exception as e:
                result["crop_error"] = str(e)
        
        return result
    except httpx.TimeoutException:
        raise HTTPException(504, "Request timed out — try again")
    except HTTPException: raise
    except Exception as e: raise HTTPException(500, str(e))


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/preview")
async def save_preview(html: str = Form(...), name: str = Form("preview")):
    preview_id = uuid.uuid4().hex[:10]
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()[:50] or "preview"
    (PREVIEWS_DIR / f"{preview_id}.html").write_text(html, encoding="utf-8")
    return {"id": preview_id, "url": f"/preview/{preview_id}", "name": safe_name}

@app.get("/preview/{preview_id}")
async def get_preview(preview_id: str):
    path = PREVIEWS_DIR / f"{preview_id}.html"
    if not path.exists():
        raise HTTPException(404, "Preview not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("\n  📸 REZVO DESIGN STUDIO — Scanner Mode")
    print("  ═══════════════════════════════════════")
    print("  Pipeline: Screenshot → AI + Crop Markers → Pillow Extraction")
    print("  Providers: Anthropic · xAI · Gemini · OpenAI")
    print("  http://0.0.0.0:8500\n")
    uvicorn.run(app, host="0.0.0.0", port=8500)
