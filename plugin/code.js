// Rezvo Scanner — Figma Plugin Code
figma.showUI(__html__, { width: 320, height: 500 });

var TYPE_COLORS = {
  'button': { r: 0.231, g: 0.510, b: 0.965 },
  'card': { r: 0.545, g: 0.361, b: 0.965 },
  'nav-item': { r: 0.063, g: 0.725, b: 0.506 },
  'tab': { r: 0.961, g: 0.620, b: 0.043 },
  'link': { r: 0.925, g: 0.282, b: 0.600 },
  'input': { r: 0.388, g: 0.400, b: 0.945 }
};

var DEFAULT_COLOR = { r: 0.231, g: 0.510, b: 0.965 };

figma.ui.onmessage = function(msg) {

  if (msg.type === 'import-scan') {
    var name = msg.name || 'Rezvo Scan';
    var width = msg.width || 1440;
    var height = msg.height || 900;
    var imageBytes = msg.imageBytes;
    var elements = msg.elements || [];

    try {
      // Create main frame
      var frame = figma.createFrame();
      frame.name = name;
      frame.resize(width, height);
      frame.x = Math.round(figma.viewport.center.x - width / 2);
      frame.y = Math.round(figma.viewport.center.y - height / 2);
      frame.clipsContent = true;

      // Add screenshot as image fill
      if (imageBytes) {
        var uint8 = new Uint8Array(imageBytes);
        var image = figma.createImage(uint8);

        var imageRect = figma.createRectangle();
        imageRect.name = 'Screenshot';
        imageRect.resize(width, height);
        imageRect.x = 0;
        imageRect.y = 0;
        imageRect.fills = [{
          type: 'IMAGE',
          scaleMode: 'FILL',
          imageHash: image.hash
        }];
        frame.appendChild(imageRect);
      }

      // Create element overlays
      if (elements.length > 0) {
        for (var i = 0; i < elements.length; i++) {
          var el = elements[i];
          var x = (el.x / 100) * width;
          var y = (el.y / 100) * height;
          var w = (el.w / 100) * width;
          var h = (el.h / 100) * height;

          var rect = figma.createRectangle();
          rect.name = el.label || el.type || 'Element';
          rect.x = x;
          rect.y = y;
          rect.resize(Math.max(w, 4), Math.max(h, 4));

          var color = TYPE_COLORS[el.type] || DEFAULT_COLOR;
          rect.fills = [{
            type: 'SOLID',
            color: color,
            opacity: 0.06
          }];
          rect.strokes = [{
            type: 'SOLID',
            color: color,
            opacity: 0.3
          }];
          rect.strokeWeight = 1.5;
          frame.appendChild(rect);
        }
      }

      // Select and zoom
      figma.currentPage.selection = [frame];
      figma.viewport.scrollAndZoomIntoView([frame]);

      var elCount = elements.length || 0;
      figma.ui.postMessage({ type: 'status', text: 'Imported: ' + name + ' (' + elCount + ' elements)', level: 'success' });

    } catch(e) {
      figma.ui.postMessage({ type: 'status', text: 'Error: ' + e.message, level: 'error' });
    }
  }
};
