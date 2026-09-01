#!/bin/bash
# Rainmaker launcher. Run this from inside the Rainmaker folder:  ./start.sh
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Setting up (first run only)…"
  if ! python3 -m venv .venv; then
    echo ""
    echo "Couldn't create a Python environment. This usually means Xcode's"
    echo "command line tools aren't installed. Try:"
    echo "    xcode-select --install"
    echo "then run ./start.sh again."
    exit 1
  fi
fi

source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "Starting Rainmaker..."
echo "On your iPhone (same WiFi), open:  http://$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "<this-Mac's-IP>"):8000"
echo "Press Ctrl+C to stop."
echo ""
uvicorn app:app --host 0.0.0.0 --port 8000
