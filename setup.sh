#!/usr/bin/env bash
# setup.sh — Audio-to-Score setup for Linux/macOS
set -e

echo "======================================"
echo "  Audio-to-Score Setup"
echo "======================================"
echo ""

# Check Python version
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 9 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Error: Python 3.9+ is required. Please install Python from https://python.org"
    exit 1
fi

echo "Using Python: $PYTHON ($($PYTHON --version))"
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "======================================"
echo "  Optional dependencies"
echo "======================================"

# cairosvg for PDF rendering
echo ""
read -p "Install cairosvg for better PDF rendering? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install cairosvg
fi

# demucs for source separation
echo ""
read -p "Install demucs for source separation? (y/N) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pip install demucs
fi

echo ""
echo "======================================"
echo "  Setup complete!"
echo "======================================"
echo ""
echo "Activate the environment with:"
echo "  source venv/bin/activate"
echo ""
echo "Usage:"
echo "  python transcribe.py song.mp3"
echo "  python transcribe.py song.wav --instrument piano"
echo "  python transcribe.py *.mp3 --output ./scores"
echo ""

# Check for PDF renderers
if $PYTHON -c "import verovio" 2>/dev/null; then
    echo "PDF renderer: verovio (installed)"
elif command -v mscore &>/dev/null || command -v musescore &>/dev/null; then
    echo "PDF renderer: MuseScore CLI (installed)"
else
    echo "Note: For PDF output, verovio is included in requirements."
    echo "  If PDF export fails, install MuseScore: https://musescore.org/download"
fi

echo ""
