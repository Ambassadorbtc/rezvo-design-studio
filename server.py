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
# STEP 1: IMAGE REGION DETECTION (Gemini + OpenCV fallback)
# ═══════════════════════════════════════════════════════════

import numpy as np
import cv2

DETECTION_PROMPT = """Look at this UI screenshot carefully. 

I need you to find every PHOTOGRAPH or REAL IMAGE in this screenshot. These are areas containing actual photos of food, products, people, or real-world objects — NOT solid colored rectangles, NOT icons, NOT text.

For each photograph found, return its bounding box as pixel coordinates relative to the image dimensions.

Return a JSON array like this:
[
  {"label": "description", "x1": 100, "y1": 400, "x2": 300, "y2": 550},
  {"label": "description", "x1": 310, "y1": 400, "x2": 500, "y2": 550}
]

Where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner in PIXELS based on the original image dimensions.

If there are NO photographs (only colored tiles, icons, and text), return an empty array: []

Return ONLY valid JSON, nothing else."""


def detect_images_opencv(img: Image.Image, min_area_pct: float = 0.5) -> list:
    """Use OpenCV texture analysis to find photograph regions in a UI screenshot.
    Photos have high color variance and texture. UI elements (buttons, tiles) are flat."""
    
    # Convert PIL to OpenCV
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    # Compute local texture using Laplacian variance in a grid
    # High variance = likely a photograph, low variance = flat UI element
    block_h, block_w = max(20, h // 30), max(20, w // 30)
    texture_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h - block_h, block_h // 2):
        for x in range(0, w - block_w, block_w // 2):
            block = gray[y:y+block_h, x:x+block_w]
            variance = cv2.Laplacian(block, cv2.CV_64F).var()
            texture_map[y:y+block_h, x:x+block_w] = np.maximum(
                texture_map[y:y+block_h, x:x+block_w], variance
            )
    
    # Also check color variance (photos have diverse colors, UI has flat colors)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)
    color_var_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h - block_h, block_h // 2):
        for x in range(0, w - block_w, block_w // 2):
            block_hsv = hsv[y:y+block_h, x:x+block_w]
            hue_std = np.std(block_hsv[:,:,0].astype(float))
            sat_std = np.std(block_hsv[:,:,1].astype(float))
            color_var_map[y:y+block_h, x:x+block_w] = np.maximum(
                color_var_map[y:y+block_h, x:x+block_w], hue_std + sat_std
            )
    
    # Combine texture + color variance
    # Normalize both to 0-1
    if texture_map.max() > 0:
        texture_norm = texture_map / texture_map.max()
    else:
        texture_norm = texture_map
    if color_var_map.max() > 0:
        color_norm = color_var_map / color_var_map.max()
    else:
        color_norm = color_var_map
    
    combined = (texture_norm * 0.6 + color_norm * 0.4)
    
    # Threshold to find high-texture regions (likely photos)
    threshold = 0.35
    binary = (combined > threshold).astype(np.uint8) * 255
    
    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (block_w, block_h))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    min_area = (w * h) * (min_area_pct / 100)  # Minimum area as % of image
    regions = []
    
    for contour in contours:
        x1, y1, cw, ch = cv2.boundingRect(contour)
        area = cw * ch
        if area < min_area:
            continue
        # Skip regions that span too much of the image (likely background)
        if cw > w * 0.8 and ch > h * 0.8:
            continue
        # Skip very thin strips
        aspect = max(cw, ch) / max(min(cw, ch), 1)
        if aspect > 8:
            continue
            
        regions.append({
            "label": f"detected_image_{len(regions)}",
            "box_pixels": [x1, y1, x1 + cw, y1 + ch],
            "area": area,
            "texture_score": float(np.mean(combined[y1:y1+ch, x1:x1+cw]))
        })
    
    # Sort by position (top to bottom, left to right)
    regions.sort(key=lambda r: (r["box_pixels"][1], r["box_pixels"][0]))
    
    log.info(f"OpenCV detected {len(regions)} photo regions")
    return regions


async def detect_image_regions(gemini_key: str, img_b64: str, img_type: str, img: Image.Image = None) -> list:
    """Try Gemini first for detection, fall back to OpenCV texture analysis."""
    
    regions = []
    
    # Try Gemini API detection
    if gemini_key:
        try:
            iw, ih = img.size if img else (1024, 768)
            async with httpx.AsyncClient(timeout=60.0) as c:
                r = await c.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{
                            "parts": [
                                {"inline_data": {"mime_type": img_type, "data": img_b64}},
                                {"text": DETECTION_PROMPT + f"\n\nThe image dimensions are {iw}x{ih} pixels."}
                            ]
                        }],
                        "generationConfig": {
                            "maxOutputTokens": 4096,
                            "temperature": 0.1,
                            "responseMimeType": "application/json",
                        },
                    }
                )
                if r.status_code == 200:
                    data = r.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    text = text.strip()
                    if text.startswith("```json"): text = text[7:]
                    elif text.startswith("```"): text = text[3:]
                    if text.endswith("```"): text = text[:-3]
                    text = text.strip()
                    
                    parsed = json.loads(text)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, dict) and all(k in item for k in ("x1", "y1", "x2", "y2")):
                                regions.append({
                                    "label": item.get("label", f"image_{len(regions)}"),
                                    "box_pixels": [int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])]
                                })
                        log.info(f"Gemini detected {len(regions)} regions")
                else:
                    log.warning(f"Gemini detection returned {r.status_code}: {r.text[:200]}")
        except Exception as e:
            log.warning(f"Gemini detection failed: {e}")
    
    # Fallback to OpenCV if Gemini found nothing
    if not regions and img:
        log.info("Gemini found no images, trying OpenCV texture analysis...")
        cv_regions = detect_images_opencv(img)
        for r in cv_regions:
            regions.append({
                "label": r["label"],
                "box_pixels": r["box_pixels"]
            })
    
    return regions


# ═══════════════════════════════════════════════════════════
# TRUE SCANNER MODE — Screenshot IS the visual
# ═══════════════════════════════════════════════════════════

SCANNER_ANALYSIS_PROMPT = """Analyze this UI screenshot and map its interactive structure.

You are NOT rebuilding this UI. The original screenshot will be used as the visual.
Your job is to identify every interactive element and return its position.

Return ONLY a JSON object with this structure:
{{
  "width": <detected UI width in pixels>,
  "height": <detected UI height in pixels>,
  "elements": [
    {{
      "type": "button|link|tab|input|nav-item|card",
      "label": "text on the element",
      "x": <x position as percentage 0-100>,
      "y": <y position as percentage 0-100>,
      "w": <width as percentage 0-100>,
      "h": <height as percentage 0-100>,
      "action": "description of what clicking would do"
    }}
  ]
}}

Rules:
- Coordinates are PERCENTAGES of the full image (0-100)
- Map EVERY clickable element: buttons, cards, tabs, nav items, menu items
- Food/product cards are clickable elements — include each one
- Include navigation bar items at the bottom
- Include the search icon, grid icon, scan button
- Include the Pay button
- Include Check/Actions/Guest tabs
- Return ONLY valid JSON"""


def build_scanner_html(img_data_url: str, elements: list, target_w: int, target_h: int) -> str:
    """Build HTML that uses the screenshot as visual + interactive overlays."""
    
    hotspots = []
    for el in elements:
        x = el.get("x", 0)
        y = el.get("y", 0)
        w = el.get("w", 5)
        h = el.get("h", 5)
        label = el.get("label", "")
        el_type = el.get("type", "button")
        
        # Style based on type
        hover_bg = "rgba(255,255,255,0.15)"
        border_radius = "8px"
        if el_type == "nav-item":
            hover_bg = "rgba(255,255,255,0.1)"
        elif el_type == "card":
            hover_bg = "rgba(255,255,255,0.12)"
            border_radius = "12px"
        elif el_type == "tab":
            hover_bg = "rgba(0,0,0,0.05)"
            border_radius = "4px"
        
        hotspots.append(f'''    <div class="hotspot" 
      style="left:{x}%;top:{y}%;width:{w}%;height:{h}%;border-radius:{border_radius};"
      data-label="{label}" data-type="{el_type}"
      onclick="handleClick('{label}','{el_type}')"
      title="{label}">
    </div>''')
    
    hotspots_html = "\n".join(hotspots)
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scanned UI</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ 
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; background: #1a1a2e;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  }}
  .scanner-frame {{
    position: relative;
    width: {target_w}px; height: {target_h}px;
    background-image: url('{img_data_url}');
    background-size: 100% 100%;
    background-repeat: no-repeat;
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 25px 60px rgba(0,0,0,0.5);
  }}
  .hotspot {{
    position: absolute;
    cursor: pointer;
    transition: all 0.15s ease;
    z-index: 2;
  }}
  .hotspot:hover {{
    background: rgba(255,255,255,0.15);
    box-shadow: 0 0 0 2px rgba(59,130,246,0.5);
  }}
  .hotspot:active {{
    background: rgba(59,130,246,0.2);
    transform: scale(0.98);
  }}
  .toast {{
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    background: #1e293b; color: #f1f5f9; padding: 12px 24px;
    border-radius: 10px; font-size: 14px; display: none; z-index: 100;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3); border: 1px solid #334155;
  }}
  .toast.show {{ display: block; animation: fadeIn 0.2s ease; }}
  @keyframes fadeIn {{ from {{ opacity: 0; transform: translateX(-50%) translateY(10px); }} to {{ opacity: 1; transform: translateX(-50%) translateY(0); }} }}
</style>
</head>
<body>
  <div class="scanner-frame">
{hotspots_html}
  </div>
  <div class="toast" id="toast"></div>
  <script>
    function handleClick(label, type) {{
      const toast = document.getElementById('toast');
      toast.textContent = `${{type}}: ${{label}}`;
      toast.className = 'toast show';
      setTimeout(() => toast.className = 'toast', 1500);
    }}
  </script>
</body>
</html>'''


async def scanner_analyze(gemini_key: str, img_b64: str, img_type: str) -> list:
    """Use Gemini to analyze UI structure for scanner mode."""
    if not gemini_key:
        return []
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{
                        "parts": [
                            {"inline_data": {"mime_type": img_type, "data": img_b64}},
                            {"text": SCANNER_ANALYSIS_PROMPT}
                        ]
                    }],
                    "generationConfig": {
                        "maxOutputTokens": 8000,
                        "temperature": 0.1,
                        "responseMimeType": "application/json",
                    },
                }
            )
            if r.status_code == 200:
                data = r.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text.startswith("```json"): text = text[7:]
                if text.startswith("```"): text = text[3:]
                if text.endswith("```"): text = text[:-3]
                parsed = json.loads(text.strip())
                elements = parsed.get("elements", []) if isinstance(parsed, dict) else []
                log.info(f"Scanner analysis found {len(elements)} interactive elements")
                return elements
            else:
                log.warning(f"Scanner analysis failed: {r.status_code}")
                return []
    except Exception as e:
        log.warning(f"Scanner analysis error: {e}")
        return []

def crop_detected_regions(img: Image.Image, regions: list, max_images: int = 20) -> list:
    """Crop each detected region from the screenshot, return list with URLs and base64."""
    iw, ih = img.size
    results = []
    
    for i, region in enumerate(regions[:max_images]):
        box = region["box_pixels"]  # [x1, y1, x2, y2] in actual pixels
        
        x_min = max(0, min(int(box[0]), iw - 1))
        y_min = max(0, min(int(box[1]), ih - 1))
        x_max = max(x_min + 10, min(int(box[2]), iw))
        y_max = max(y_min + 10, min(int(box[3]), ih))
        
        # Crop
        cropped = img.crop((x_min, y_min, x_max, y_max))
        
        # Save full quality for serving via URL
        crop_id = f"{uuid.uuid4().hex[:8]}.jpg"
        crop_path = EXTRACTED_DIR / crop_id
        cropped.save(crop_path, format="JPEG", quality=85, optimize=True)
        
        # Create base64 for embedding in final HTML (JPEG, compressed)
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=80, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        
        results.append({
            "index": i,
            "label": region["label"],
            "box_pixels": [x_min, y_min, x_max, y_max],
            "size": [x_max - x_min, y_max - y_min],
            "url": f"/extracted/{crop_id}",
            "data_url": f"data:image/jpeg;base64,{b64}",
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
    """Build a text manifest of extracted images for the code generation prompt.
    Uses server URLs (not base64) to keep prompt size manageable."""
    if not extracted_images:
        return "No photographs were detected in this screenshot."
    
    lines = []
    for img in extracted_images:
        lines.append(f'  - Image {img["index"]}: "{img["label"]}" ({img["size"][0]}x{img["size"][1]}px)')
        lines.append(f'    Use: <img src="{img["url"]}" alt="{img["label"]}" style="object-fit:cover;">')
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
    pil_img = None
    
    # Decode the screenshot once for reuse
    if has_image:
        try:
            img_bytes = base64.b64decode(image_base64)
            pil_img = Image.open(io.BytesIO(img_bytes))
            if pil_img.mode != 'RGB':
                pil_img = pil_img.convert('RGB')
            log.info(f"Screenshot decoded: {pil_img.size[0]}x{pil_img.size[1]}px")
        except Exception as e:
            log.error(f"Failed to decode screenshot: {e}")
    
    # ── STEP 1: Detect image regions ──
    detection_key = gemini_key or (api_key if provider == "gemini" else "")
    
    if has_image and pil_img:
        log.info("Step 1: Detecting image regions...")
        regions = await detect_image_regions(detection_key, image_base64, image_media_type, pil_img)
        
        if regions:
            # ── STEP 2: Crop actual pixels ──
            log.info(f"Step 2: Cropping {len(regions)} detected regions with Pillow...")
            try:
                extracted_images = crop_detected_regions(pil_img, regions)
                log.info(f"  Extracted {len(extracted_images)} images successfully")
            except Exception as e:
                log.error(f"  Crop failed: {e}")
    
    # ── STEP 3: Build prompt with image manifest ──
    image_manifest = build_image_manifest(extracted_images)
    system = SYSTEM_PROMPT.format(w=w, h=h, device=device, image_manifest=image_manifest)
    
    is_scan = False
    if not prompt:
        is_scan = True
    else:
        p = prompt.lower().strip().rstrip('.')
        # Exact matches
        if p in ('copy this', 'replicate this', 'clone this', 'copy', 'replicate',
                 'clone', 'recreate this', 'rebuild this', 'make this', 'scan this', 'scan'):
            is_scan = True
        # Fuzzy matches — if the prompt is basically saying "scan/copy this exactly"
        elif any(kw in p for kw in ('scan', 'copy this', 'pixel perfect', 'pixel-perfect', 'no placeholder', 'no place holder', 'no guessing', 'exactly', 'replicate', 'clone')):
            is_scan = True
    
    # ═══ TRUE SCANNER MODE ═══
    # For scan commands with a screenshot: use the screenshot AS the visual
    if has_image and pil_img and is_scan:
        log.info("TRUE SCANNER MODE: Using screenshot as visual + interactive overlay")
        
        # Create compressed JPEG data URL of the screenshot
        buf = io.BytesIO()
        # Resize if needed to keep reasonable size
        scan_img = pil_img.copy()
        max_dim = 1600
        if scan_img.width > max_dim or scan_img.height > max_dim:
            scan_img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        scan_img.save(buf, format="JPEG", quality=88, optimize=True)
        scan_b64 = base64.b64encode(buf.getvalue()).decode()
        scan_data_url = f"data:image/jpeg;base64,{scan_b64}"
        
        w, h = DEVICE_SIZES.get(device, (1024, 768))
        
        # Get interactive elements from Gemini
        detection_key = gemini_key or (api_key if provider == "gemini" else "")
        elements = await scanner_analyze(detection_key, image_base64, image_media_type)
        
        # Build the scanner HTML
        html = build_scanner_html(scan_data_url, elements, w, h)
        
        return {
            "html": html,
            "provider": "scanner",
            "model": "true-scanner + gemini-analysis",
            "images_detected": len(elements),
            "detection_used": True,
            "detection_method": "true-scanner",
            "usage": {"mode": "scanner", "elements": len(elements)},
        }
    
    if has_image:
        img_count = len(extracted_images)
        img_note = f"\n\n{img_count} photographs have been extracted from the screenshot and are listed in the system prompt. Use the provided <img> tags with data URLs for each one. Position them exactly where they appear in the original." if img_count > 0 else ""
        
        if not prompt:
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
        result["detection_used"] = bool(has_image)
        result["detection_method"] = "gemini+opencv" if has_image else "none"
        
        # Post-process: replace server URLs with base64 data URLs for self-contained HTML
        if extracted_images:
            html = result["html"]
            for img in extracted_images:
                html = html.replace(img["url"], img["data_url"])
            result["html"] = html
            result["extracted_images"] = [{"label": img["label"], "url": img["url"], "size": img["size"]} for img in extracted_images]
        
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
