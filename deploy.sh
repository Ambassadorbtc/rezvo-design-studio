#!/bin/bash
# ═══════════════════════════════════════════
# REZVO DESIGN STUDIO — Deploy Script
# Run this on your VPS to get the studio live
# ═══════════════════════════════════════════

set -e

echo ""
echo "  🎨 REZVO DESIGN STUDIO"
echo "  ═══════════════════════"
echo ""

# Install Python deps
echo "→ Installing dependencies..."
pip3 install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt -q

# Create uploads directory
mkdir -p uploads

echo "→ Starting server on port 8500..."
echo ""
echo "  ✅ Studio is live at:"
echo "  http://$(hostname -I | awk '{print $1}'):8500"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

python3 server.py
