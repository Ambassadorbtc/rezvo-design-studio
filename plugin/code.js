// Rezvo Scanner — Figma Plugin Code
// Creates proper Figma frames from scan data

figma.showUI(__html__, { width: 320, height: 500 });

// Color map for element types
const TYPE_COLORS = {
  'button': { r: 0.231, g: 0.510, b: 0.965 },   // blue
  'card': { r: 0.545, g: 0.361, b: 0.965 },       // purple  
  'nav-item': { r: 0.063, g: 0.725, b: 0.506 },   // green
  'tab': { r: 0.961, g: 0.620, b: 0.043 },         // amber
  'link': { r: 0.925, g: 0.282, b: 0.600 },        // pink
  'input': { r: 0.388, g: 0.400, b: 0.945 },       // indigo
};

const DEFAULT_COLOR = { r: 0.231, g: 0.510, b: 0.965 };

figma.ui.onmessage = async (msg) => {
  
  if (msg.type === 'import-scan') {
    try {
      const { name, width, height, imageBytes, elements } = msg;
      
      // 1. Create the main frame
      const frame = figma.createFrame();
      frame.name = name;
      frame.resize(width, height);
      frame.x = Math.round(figma.viewport.center.x - width / 2);
      frame.y = Math.round(figma.viewport.center.y - height / 2);
      frame.clipsContent = true;
      frame.cornerRadius = 16;
      
      // 2. Add screenshot as image fill
      if (imageBytes) {
        const uint8 = new Uint8Array(imageBytes);
        const image = figma.createImage(uint8);
        
        const imageRect = figma.createRectangle();
        imageRect.name = 'Screenshot';
        imageRect.resize(width, height);
        imageRect.x = 0;
        imageRect.y = 0;
        imageRect.fills = [{
          type: 'IMAGE',
          scaleMode: 'FILL',
          imageHash: image.hash,
        }];
        frame.appendChild(imageRect);
      }
      
      // 3. Create element overlay group
      if (elements && elements.length > 0) {
        const overlayGroup = figma.group([], frame);
        overlayGroup.name = 'Interactive Elements';
        
        for (const el of elements) {
          const x = (el.x / 100) * width;
          const y = (el.y / 100) * height;
          const w = (el.w / 100) * width;
          const h = (el.h / 100) * height;
          
          // Create rectangle for each element
          const rect = figma.createRectangle();
          rect.name = el.label || el.type || 'Element';
          rect.x = x;
          rect.y = y;
          rect.resize(Math.max(w, 4), Math.max(h, 4));
          rect.cornerRadius = el.type === 'card' ? 12 : el.type === 'tab' ? 4 : 8;
          
          // Semi-transparent fill
          const color = TYPE_COLORS[el.type] || DEFAULT_COLOR;
          rect.fills = [{
            type: 'SOLID',
            color: color,
            opacity: 0.06,
          }];
          
          // Subtle stroke
          rect.strokes = [{
            type: 'SOLID',
            color: color,
            opacity: 0.3,
          }];
          rect.strokeWeight = 1.5;
          
          // Add to overlay group
          frame.appendChild(rect);
          
          // Add text label
          const label = figma.createText();
          await figma.loadFontAsync({ family: "Inter", style: "Medium" });
          label.fontName = { family: "Inter", style: "Medium" };
          label.fontSize = Math.min(11, Math.max(8, h * 0.25));
          label.characters = el.label || el.type || '';
          label.fills = [{ type: 'SOLID', color: color, opacity: 0.6 }];
          label.x = x + 4;
          label.y = y + h - label.height - 2;
          label.name = `Label: ${el.label || el.type}`;
          frame.appendChild(label);
        }
      }
      
      // 4. Select and zoom to the new frame
      figma.currentPage.selection = [frame];
      figma.viewport.scrollAndZoomIntoView([frame]);
      
      figma.ui.postMessage({ type: 'status', text: `Imported: ${name} (${elements?.length || 0} elements)`, level: 'success' });
      
    } catch(e) {
      figma.ui.postMessage({ type: 'status', text: `Error: ${e.message}`, level: 'error' });
    }
  }
};
