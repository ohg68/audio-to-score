# Audio-to-Score

Transcribe audio files (WAV/MP3) into professional sheet music exported as **PDF**, **MusicXML**, and **MIDI**.

## Features

- **Piano** — Grand staff (treble + bass clef), automatic hand separation
- **Guitar** — Standard notation + TAB with intelligent string/fret assignment
- **Voice** — Single staff with pYIN pitch detection optimized for vocals
- **Auto** — Detects mono/polyphonic content automatically
- **Source separation** — Optional Demucs integration for mixed recordings
- **Professional output** — A4 PDF with dynamics, key/time signatures, tempo markings

## Quick Start

```bash
# Setup
chmod +x setup.sh && ./setup.sh   # Linux/macOS
setup.bat                           # Windows

# Activate environment
source venv/bin/activate            # Linux/macOS
venv\Scripts\activate.bat           # Windows

# Transcribe
python transcribe.py song.mp3
python transcribe.py song.wav --instrument piano
python transcribe.py *.mp3 --instrument guitar --output ./scores
python transcribe.py song.mp3 --formats pdf musicxml
python transcribe.py mix.mp3 --separate --instrument piano
```

## Output

For each input file, generates:

| File | Description |
|------|-------------|
| `song.pdf` | A4 sheet music ready to print |
| `song.musicxml` | Editable in Finale, Sibelius, MuseScore |
| `song.mid` | Editable in any DAW |

## CLI Options

| Option | Description |
|--------|-------------|
| `--instrument`, `-i` | `piano`, `guitar`, `voice`, `auto` (default) |
| `--output`, `-o` | Output directory (default: same as input) |
| `--formats`, `-f` | Output formats: `pdf`, `musicxml`, `midi` |
| `--separate`, `-s` | Enable Demucs source separation |
| `--title`, `-t` | Custom score title |
| `--quiet`, `-q` | Suppress progress output |

## Requirements

- Python 3.9+
- Dependencies installed via `requirements.txt`

### Core

- `basic-pitch` — Spotify's polyphonic AMT
- `librosa` — Audio analysis + pYIN pitch detection
- `music21` — Score construction and music theory
- `verovio` — MusicXML to PDF rendering

### Optional

- `demucs` — Source separation (`--separate` flag)
- `cairosvg` — Better SVG-to-PDF conversion
- MuseScore 4 — Fallback PDF renderer

## How It Works

1. Load audio, normalize to 22050 Hz mono
2. Detect BPM and time signature via beat tracking
3. (Optional) Separate sources with Demucs
4. Transcribe with basic-pitch (polyphonic) or pYIN (voice)
5. Quantize notes to rhythmic grid
6. Build music21 Score with appropriate instrument layout
7. Analyze key signature
8. Add dynamics (pp-ff from RMS amplitude)
9. Export MusicXML + MIDI via music21, PDF via verovio

## Project Structure

```
audio-to-score/
├── transcribe.py          # CLI entry point
├── src/
│   ├── audio_processor.py # Load, normalize, source separation
│   ├── pitch_detector.py  # basic-pitch + pYIN
│   ├── rhythm_analyzer.py # BPM, time signature, quantization
│   ├── score_builder.py   # music21 Score (piano/guitar/voice)
│   ├── guitar_tab.py      # Tablature string/fret assignment
│   ├── pdf_exporter.py    # verovio/MuseScore PDF rendering
│   └── exporters.py       # MusicXML and MIDI export
├── requirements.txt
├── setup.sh               # Linux/macOS setup
└── setup.bat              # Windows setup
```
