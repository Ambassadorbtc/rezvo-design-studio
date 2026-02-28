"""
REZVO DESIGN STUDIO — Server v3 (Vision Detection Pipeline)
Step 1: Gemini detects image regions with bounding boxes
Step 2: Pillow crops actual pixels from screenshot
Step 3: Code generation LLM builds HTML referencing extracted images
"""

import os, json, base64, uuid, re, io, httpx, logging
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rezvo-studio")

app = FastAPI(title="Rezvo Design Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
PREVIEWS_DIR = Path("previews")
PREVIEWS_DIR.mkdir(exist_ok=True)
EXTRACTED_DIR = Path("extracted")
EXTRACTED_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/extracted", StaticFiles(directory="extracted"), name="extracted")

DEVICE_SIZES = {"mobile": (390, 844), "tablet": (1024, 768), "desktop": (1440, 900)}


# ═══════════════════════════════════════════════════════════
# STEP 1: GEMINI BOUNDING BOX DETECTION
# ═══════════════════════════════════════════════════════════

DETECTION_PROMPT = """Analyze this UI screenshot. Find ALL photographs, food images, product images, 
profile pictures, thumbnails, and any visual content that is NOT a solid color, icon, or text.

Return ONLY a JSON array. Each item must have:
- "label": short description (e.g. "banh mi sandwich photo", "chicken wrap photo")  
- "box": [y_min, x_min, y_max, x_max] as integers 0-1000 (Gemini's normalized coordinate system)

Example response:
[
  {"label": "banh mi sandwich photo", "box": [520, 50, 620, 250]},
  {"label": "chicken wrap photo", "box": [520, 260, 620, 460]}
]

Rules:
- Coordinates are 0-1000 scale (0=top-left, 1000=bottom-right)
- Only detect PHOTOGRAPHS and CONTENT IMAGES, not colored category tiles, icons, or UI elements
- Be precise — the bounding box should tightly contain just the image, not surrounding padding
- Return [] if no photographs/content images are found
- Return ONLY the JSON array, no other text"""


async def detect_image_regions(gemini_key: str, img_b64: str, img_type: str) -> list:
    """Use Gemini to detect photograph/image regions and return bounding boxes."""
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [
                        {"inline_data": {"mime_type": img_type, "data": img_b64}},
                        {"text": DETECTION_PROMPT}
                    ]
                }],
                "generationConfig": {"maxOutputTokens": 4096, "temperature": 0.1},
            }
        )
        if r.status_code != 200:
            log.error(f"Gemini detection failed: {r.text}")
            return []
        
        try:
            data = r.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            # Clean up response — extract JSON array
            text = text.strip()
            if text.startswith("```json"): text = text[7:]
            elif text.startswith("```"): text = text[3:]
            if text.endswith("```"): text = text[:-3]
            text = text.strip()
            
            regions = json.loads(text)
            if not isinstance(regions, list):
                return []
            
            # Validate each region
            valid = []
            for r in regions:
                if isinstance(r, dict) and "box" in r and len(r["box"]) == 4:
                    box = r["box"]
                    if all(isinstance(v, (int, float)) for v in box):
                        valid.append({
                            "label": r.get("label", f"image_{len(valid)}"),
                            "box": [int(v) for v in box]
                        })
            
            log.info(f"Gemini detected {len(valid)} image regions")
            return valid
        except Exception as e:
            log.error(f"Failed to parse Gemini detection response: {e}")
            return []


# ═══════════════════════════════════════════════════════════
# STEP 2: PILLOW CROPS ACTUAL PIXELS
# ═══════════════════════════════════════════════════════════

def crop_detected_regions(img: Image.Image, regions: list) -> list:
    """Crop each detected region from the screenshot, return list with base64 data."""
    iw, ih = img.size
    results = []
    
    for i, region in enumerate(regions):
        box = region["box"]  # [y_min, x_min, y_max, x_max] in 0-1000 scale
        
        # Convert from Gemini's 0-1000 normalized coords to actual pixels
        y_min = int(box[0] / 1000 * ih)
        x_min = int(box[1] / 1000 * iw)
        y_max = int(box[2] / 1000 * ih)
        x_max = int(box[3] / 1000 * iw)
        
        # Clamp to image bounds
        x_min = max(0, min(x_min, iw - 1))
        y_min = max(0, min(y_min, ih - 1))
        x_max = max(x_min + 10, min(x_max, iw))
        y_max = max(y_min + 10, min(y_max, ih))
        
        # Crop
        cropped = img.crop((x_min, y_min, x_max, y_max))
        
        # Save to file for serving
        crop_id = f"{uuid.uuid4().hex[:8]}.png"
        crop_path = EXTRACTED_DIR / crop_id
        cropped.save(crop_path, format="PNG", optimize=True)
        
        # Also create base64 for embedding
        buf = io.BytesIO()
        cropped.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        results.append({
            "index": i,
            "label": region["label"],
            "box_normalized": box,
            "box_pixels": [x_min, y_min, x_max, y_max],
            "size": [x_max - x_min, y_max - y_min],
            "path": f"/extracted/{crop_id}",
            "data_url": f"data:image/png;base64,{b64}",
        })
        
        log.info(f"  Cropped [{i}] '{region['label']}': {x_max-x_min}x{y_max-y_min}px")
    
    return results


# ═══════════════════════════════════════════════════════════
# STEP 3: CODE GENERATION WITH IMAGE REFERENCES
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a desktop scanner that converts UI screenshots into pixel-perfect HTML replicas.

TARGET: {w}x{h}px ({device})

OUTPUT RULES:
1. Output ONLY raw HTML. No markdown, no backticks, no explanation.
2. Use Tailwind CSS: <script src="https://cdn.tailwindcss.com"></script>
3. Complete standalone HTML: <!DOCTYPE html> through </html>.

COLOR & LAYOUT:
4. Extract EXACT hex colors from the screenshot. NEVER use Tailwind defaults like blue-500.
5. Match EXACT spacing, padding, margin, border-radius in pixels.
6. Match EXACT typography — font family, weight, size.
7. Replicate EXACT layout proportions.
8. Match shadow depths, border widths, opacity.
9. Use inline SVG for icons matching the screenshot style.

IMAGE HANDLING — CRITICAL:
10. The following images have been EXTRACTED from the screenshot and are available:
{image_manifest}
11. For EACH extracted image, use an <img> tag with the provided data URL as the src.
12. Position each image EXACTLY where it appears in the original screenshot.
13. Use object-fit: cover and appropriate border-radius to match the original.
14. If there are colored category tiles (like solid red, green, orange blocks), those are NOT images — 
    recreate those as colored divs with the exact hex colors.

COMPLETENESS:
15. Include ALL visible elements: buttons, badges, icons, nav items, text, dividers.
16. Make interactive elements look clickable with cursor-pointer.
17. Replicate any background/gradient behind the main UI card."""


def build_image_manifest(extracted_images: list) -> str:
    """Build a text manifest of extracted images for the code generation prompt."""
    if not extracted_images:
        return "No photographs were detected in this screenshot."
    
    lines = []
    for img in extracted_images:
        lines.append(f'  - Image {img["index"]}: "{img["label"]}" ({img["size"][0]}x{img["size"][1]}px)')
        lines.append(f'    Use: <img src="{img["data_url"]}" alt="{img["label"]}" style="object-fit:cover;">')
    return "\n".join(lines)


def clean_html(html: str) -> str:
    html = html.strip()
    if html.startswith("```html"): html = html[7:]
    elif html.startswith("```"): html = html.split("\n", 1)[1] if "\n" in html else html[3:]
    if html.endswith("```"): html = html[:-3]
    return html.strip()


# ═══════════════════════════════════════════════════════════
# AI PROVIDERS (Code Generation)
# ═══════════════════════════════════════════════════════════

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
        r = await c.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={"system_instruction": {"parts": [{"text": system}]}, "contents": [{"parts": parts}],
                  "generationConfig": {"maxOutputTokens": 16000}})
        if r.status_code != 200:
            raise HTTPException(r.status_code, f"Gemini: {r.json().get('error',{}).get('message', r.text)}")
        d = r.json()
        return {"html": clean_html(d["candidates"][0]["content"]["parts"][0]["text"]), "model": "gemini-2.5-flash", "usage": d.get("usageMetadata")}


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


# ═══════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════

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
    gemini_key: str = Form(""),
):
    if not api_key: raise HTTPException(400, "API key required")
    if provider not in PROVIDERS: raise HTTPException(400, f"Unknown provider: {provider}")
    w, h = DEVICE_SIZES.get(device, (1024, 768))
    
    has_image = bool(image_base64)
    extracted_images = []
    
    # ── STEP 1: Detect image regions using Gemini ──
    detection_key = gemini_key or (api_key if provider == "gemini" else "")
    
    if has_image and detection_key:
        log.info("Step 1: Detecting image regions with Gemini...")
        regions = await detect_image_regions(detection_key, image_base64, image_media_type)
        
        if regions:
            # ── STEP 2: Crop actual pixels ──
            log.info(f"Step 2: Cropping {len(regions)} detected regions with Pillow...")
            try:
                img_bytes = base64.b64decode(image_base64)
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                extracted_images = crop_detected_regions(img, regions)
                log.info(f"  Extracted {len(extracted_images)} images successfully")
            except Exception as e:
                log.error(f"  Crop failed: {e}")
    
    # ── STEP 3: Build prompt with image manifest ──
    image_manifest = build_image_manifest(extracted_images)
    system = SYSTEM_PROMPT.format(w=w, h=h, device=device, image_manifest=image_manifest)
    
    is_simple = not prompt or prompt.lower().strip() in (
        'copy this', 'replicate this', 'clone this', 'copy', 'replicate',
        'clone', 'recreate this', 'rebuild this', 'make this', 'scan this', 'scan'
    )
    
    if has_image:
        img_count = len(extracted_images)
        img_note = f"\n\n{img_count} photographs have been extracted from the screenshot and are listed in the system prompt. Use the provided <img> tags with data URLs for each one. Position them exactly where they appear in the original." if img_count > 0 else ""
        
        if is_simple:
            user_text = f"""SCAN this screenshot into pixel-perfect HTML.
Target: {device} ({w}x{h}px){img_note}

Reproduce EVERY element with exact colors, spacing, typography, and layout.
Output ONLY raw HTML."""
        else:
            user_text = f"""SCAN this screenshot into HTML with these modifications:

{prompt}

Target: {device} ({w}x{h}px){img_note}

Output ONLY raw HTML."""
    else:
        user_text = f"""Create this UI as a single HTML file:

{prompt or 'Create a professional UI design.'}

Target: {device} ({w}x{h}px)
Output ONLY raw HTML."""

    # ── Generate code ──
    log.info(f"Step 3: Generating HTML with {provider}...")
    try:
        result = await PROVIDERS[provider](api_key, system, user_text, image_base64, image_media_type)
        result["provider"] = provider
        result["images_detected"] = len(extracted_images)
        result["detection_used"] = bool(detection_key and has_image)
        
        if extracted_images:
            result["extracted_images"] = [{"label": img["label"], "path": img["path"], "size": img["size"]} for img in extracted_images]
        
        log.info(f"  Done! {len(extracted_images)} images embedded, provider={provider}")
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
    (PREVIEWS_DIR / f"{preview_id}.html").write_text(html, encoding="utf-8")
    return {"id": preview_id, "url": f"/preview/{preview_id}"}

@app.get("/preview/{preview_id}")
async def get_preview(preview_id: str):
    path = PREVIEWS_DIR / f"{preview_id}.html"
    if not path.exists(): raise HTTPException(404, "Preview not found")
    return HTMLResponse(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    print("\n  📸 REZVO DESIGN STUDIO v3 — Vision Detection Pipeline")
    print("  ═══════════════════════════════════════════════════════")
    print("  Step 1: Gemini detects image regions (bounding boxes)")
    print("  Step 2: Pillow crops actual pixels from screenshot")
    print("  Step 3: LLM generates HTML with embedded real images")
    print("  ─────────────────────────────────────────────────────")
    print("  Providers: Anthropic · xAI · Gemini · OpenAI")
    print("  http://0.0.0.0:8500\n")
    uvicorn.run(app, host="0.0.0.0", port=8500)
