#!/usr/bin/env bash
# setup_pi.sh — provisions a fresh Raspberry Pi (Debian/Raspberry Pi OS,
# Bookworm/trixie or later) to run the AI Caller phone agent.
#
# What this does:
#   1. Installs the system packages the app needs (PipeWire/WirePlumber
#      audio stack, BlueZ Bluetooth stack, Python venv tooling).
#   2. Creates a Python virtualenv INSIDE this project directory and
#      installs requirements.txt into it (piwheels index included, since
#      Debian 13+ blocks system-wide pip installs by default).
#   3. Copies local_config.example.py -> local_config.py if it doesn't
#      exist yet — does NOT fill in any secrets, that's a manual step
#      (see the printed instructions at the end).
#   4. Does NOT touch systemd/autostart at all — that's a separate,
#      deliberate step (see ai-caller.service + INSTALL.md in this same
#      deploy/ folder), so you review and enable it yourself.
#
# Usage: run this FROM the project root (the directory containing
# ai_caller_final3.py), as the user that will actually run the bot
# (not root):
#   bash deploy/setup_pi.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

if [ "$(id -u)" -eq 0 ]; then
    echo "Don't run this as root — run it as the normal user (e.g. alfaleus)" \
         "that will own the venv and the systemd --user service. It will" \
         "ask for sudo itself for the one apt-get step." >&2
    exit 1
fi

if [ ! -f "ai_caller_final3.py" ]; then
    echo "ai_caller_final3.py not found in $PROJECT_DIR — run this script" \
         "from inside the project directory (or leave it at deploy/setup_pi.sh" \
         "relative to the project root, which it already expects)." >&2
    exit 1
fi

echo "== [1/4] Installing system packages =="
sudo apt-get update
sudo apt-get install -y \
    pipewire pipewire-pulse pipewire-bin wireplumber \
    libspa-0.2-bluetooth \
    bluez bluez-tools \
    python3 python3-venv python3-pip python3-dbus \
    git

echo "== [2/4] Creating Python virtualenv (.venv) =="
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# python3-dbus above is a SYSTEM package (dbus bindings need to match the
# system libdbus) — the venv needs access to it, so it must inherit
# system site-packages rather than being fully isolated.
if ! grep -q "^include-system-site-packages = true" .venv/pyvenv.cfg 2>/dev/null; then
    sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' .venv/pyvenv.cfg
fi

echo "== [3/4] Installing Python dependencies into .venv =="
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --extra-index-url https://www.piwheels.org/simple -r requirements.txt

echo "== [4/4] Setting up local_config.py =="
if [ ! -f "local_config.py" ]; then
    cp local_config.example.py local_config.py
    echo "Created local_config.py from the example template — IT IS NOT FILLED IN YET."
else
    echo "local_config.py already exists — leaving it untouched."
fi

cat <<'EOF'

===============================================================================
Setup script done. Three manual steps remain before this Pi can take a call:

1. Fill in local_config.py with:
     - GOOGLE_CLOUD_PROJECT   (your GCP project id/number)
     - GOOGLE_CLOUD_LOCATION  (e.g. asia-south1)
     - SARVAM_API_KEY         (from dashboard.sarvam.ai)

2. Authenticate this Pi with Google Cloud (one-time, opens a browser link
   you complete on any device):
     gcloud auth login
     gcloud auth application-default login
   And in that GCP project, enable these APIs if not already:
     Cloud Speech-to-Text API, Cloud Text-to-Speech API, Vertex AI API

3. Pair and TRUST the phone(s) that will call this bot (one-time per
   phone, needs you physically confirming on the phone):
     bluetoothctl
       power on
       agent on
       default-agent
       scan on
       pair   <PHONE_MAC>
       trust  <PHONE_MAC>
       scan off
       exit

Once those three are done, test it manually first:
     .venv/bin/python3 ai_caller_final3.py

Only after a manual test call works should you set up autostart — see
deploy/ai-caller.service and deploy/INSTALL.md in this same folder.
===============================================================================
EOF
