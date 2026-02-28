#!/bin/bash
# ═══════════════════════════════════════════════════════
# REZVO DESIGN STUDIO — Full VPS Setup
# 
# Usage:
#   chmod +x setup.sh
#   sudo ./setup.sh
#
# This will:
#   1. Copy files to /opt/rezvo-studio
#   2. Install Python dependencies
#   3. Create systemd service (auto-restart, boot start)
#   4. Start the studio on port 8500
#
# Access: http://YOUR_IP:8500
# Or set up studio.rezvo.app via Nginx (instructions below)
# ═══════════════════════════════════════════════════════

set -e

echo ""
echo "  🎨 REZVO DESIGN STUDIO — Setup"
echo "  ════════════════════════════════"
echo ""

# Copy files
echo "→ Installing to /opt/rezvo-studio..."
mkdir -p /opt/rezvo-studio/static /opt/rezvo-studio/uploads
cp server.py /opt/rezvo-studio/
cp requirements.txt /opt/rezvo-studio/
cp static/index.html /opt/rezvo-studio/static/

# Install deps
echo "→ Installing Python dependencies..."
cd /opt/rezvo-studio
pip3 install -r requirements.txt -q 2>/dev/null || pip install -r requirements.txt -q

# Setup systemd
echo "→ Setting up systemd service..."
cp /opt/rezvo-studio/../rezvo-studio-app/rezvo-studio.service /etc/systemd/system/rezvo-studio.service 2>/dev/null || \
cat > /etc/systemd/system/rezvo-studio.service << 'EOF'
[Unit]
Description=Rezvo Design Studio
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/rezvo-studio
ExecStart=/usr/bin/python3 /opt/rezvo-studio/server.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable rezvo-studio
systemctl restart rezvo-studio

IP=$(hostname -I | awk '{print $1}')

echo ""
echo "  ✅ REZVO DESIGN STUDIO IS LIVE!"
echo "  ════════════════════════════════"
echo ""
echo "  🌍 http://${IP}:8500"
echo ""
echo "  Service commands:"
echo "    systemctl status rezvo-studio"
echo "    systemctl restart rezvo-studio"
echo "    journalctl -u rezvo-studio -f"
echo ""
echo "  ── Optional: Nginx for studio.rezvo.app ──"
echo ""
echo "  Add to your Nginx config:"
echo ""
echo "  server {"
echo "      listen 80;"
echo "      server_name studio.rezvo.app;"
echo "      location / {"
echo "          proxy_pass http://127.0.0.1:8500;"
echo "          proxy_set_header Host \$host;"
echo "          proxy_set_header X-Real-IP \$remote_addr;"
echo "          client_max_body_size 20M;"
echo "      }"
echo "  }"
echo ""
echo "  Then: sudo certbot --nginx -d studio.rezvo.app"
echo ""
