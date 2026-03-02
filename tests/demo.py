#!/usr/bin/env python3
"""
Demo: Audio-to-Score step-by-step walkthrough.
Shows each stage of the transcription pipeline.
"""

import sys
import os
import time
import json
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────────────
# ANSI colors
# ─────────────────────────────────────────────────
CYAN    = '\033[96m'
GREEN   = '\033[92m'
YELLOW  = '\033[93m'
MAGENTA = '\033[95m'
BOLD    = '\033[1m'
DIM     = '\033[2m'
RESET   = '\033[0m'

def header(text):
    print(f"\n{'='*60}")
    print(f"  {BOLD}{CYAN}{text}{RESET}")
    print(f"{'='*60}\n")

def step(num, text):
    print(f"  {BOLD}{MAGENTA}[PASO {num}]{RESET} {text}")

def info(label, value):
    print(f"    {DIM}{label}:{RESET} {GREEN}{value}{RESET}")

def note_display(midi_num):
    """Convert MIDI number to note name."""
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    octave = (midi_num // 12) - 1
    name = names[midi_num % 12]
    return f"{name}{octave}"

def bar_chart(label, value, max_val, width=30):
    """Simple horizontal bar."""
    filled = int((value / max_val) * width) if max_val > 0 else 0
    bar = '█' * filled + '░' * (width - filled)
    print(f"    {label:>6s} |{bar}| {value}")

# ─────────────────────────────────────────────────
# STEP 0: Generate test audio
# ─────────────────────────────────────────────────
header("AUDIO-TO-SCORE: Demo Completa")

print(f"  {DIM}Sistema de transcripcion musical{RESET}")
print(f"  {DIM}Audio WAV/MP3 -> Partitura PDF + MusicXML + MIDI{RESET}\n")

step(0, "Generando audio de prueba (piano, 8 compases, C mayor)")

SR = 22050
BPM = 120
QUARTER = 60.0 / BPM

def sine_note(freq, duration, velocity=0.7):
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    wave = (
        np.sin(2 * np.pi * freq * t) * 1.0 +
        np.sin(2 * np.pi * freq * 2 * t) * 0.4 +
        np.sin(2 * np.pi * freq * 3 * t) * 0.15
    )
    envelope = np.exp(-t * 2.5 / duration)
    attack = min(int(0.008 * SR), len(t))
    envelope[:attack] *= np.linspace(0, 1, attack)
    return wave * envelope * velocity

def midi_to_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12.0))

# Melody: C major arpeggiated pattern
melody = [
    (60,1),(64,1),(67,1),(72,1),  # C E G C
    (71,1),(67,1),(64,1),(60,1),  # B G E C
    (62,1),(65,1),(69,1),(74,1),  # D F A D
    (72,1),(69,1),(65,1),(62,1),  # C A F D
    (64,1),(67,1),(71,1),(76,1),  # E G B E
    (74,1),(71,1),(67,1),(64,1),  # D B G E
    (60,1),(64,1),(67,1),(72,1),  # C E G C
    (72,4),                        # C (whole)
]

bass = [
    (48,4),(48,4),  # C3
    (50,4),(50,4),  # D3
    (52,4),(52,4),  # E3
    (48,4),(48,4),  # C3
]

def render_voice(notes, vel=0.6):
    audio = np.zeros(0)
    for midi, dur in notes:
        tone = sine_note(midi_to_freq(midi), dur * QUARTER, velocity=vel)
        audio = np.concatenate([audio, tone])
    return audio

mel_audio = render_voice(melody, 0.7)
bas_audio = render_voice(bass, 0.45)
maxlen = max(len(mel_audio), len(bas_audio))
mel_audio = np.pad(mel_audio, (0, maxlen - len(mel_audio)))
bas_audio = np.pad(bas_audio, (0, maxlen - len(bas_audio)))
mixed = (mel_audio + bas_audio)
mixed = (mixed / np.max(np.abs(mixed)) * 0.9).astype(np.float32)

wav_path = os.path.join(os.path.dirname(__file__), 'sample_audio', 'demo_piano.wav')
os.makedirs(os.path.dirname(wav_path), exist_ok=True)
sf.write(wav_path, mixed, SR)

info("Archivo", wav_path.split('/')[-1])
info("Duracion", f"{len(mixed)/SR:.1f}s")
info("Sample rate", f"{SR} Hz")
info("BPM original", f"{BPM}")
info("Tonalidad", "C mayor")
info("Compas", "4/4")
info("Notas melodia", f"{len(melody)} (mano derecha)")
info("Notas bajo", f"{len(bass)} (mano izquierda)")

# Waveform ASCII visualization
print(f"\n    {DIM}Forma de onda:{RESET}")
chunk_size = len(mixed) // 60
for i in range(60):
    chunk = mixed[i*chunk_size:(i+1)*chunk_size]
    amplitude = int(np.max(np.abs(chunk)) * 15)
    bar = '▓' * amplitude + '░' * (15 - amplitude)
    rbar = '░' * (15 - amplitude) + '▓' * amplitude
    print(f"    {rbar}|{bar}", end='')
    if i % 60 == 59:
        print()
print()

# ─────────────────────────────────────────────────
# STEP 1: Load Audio
# ─────────────────────────────────────────────────
step(1, "Cargando y normalizando audio")
t0 = time.time()

from src.audio_processor import AudioProcessor
processor = AudioProcessor()
audio_data = processor.load(wav_path)

info("Mono", "Si")
info("Normalizado", "[-1.0, 1.0]")
info("Muestras", f"{len(audio_data['waveform']):,}")
info("Tiempo", f"{time.time()-t0:.2f}s")

# ─────────────────────────────────────────────────
# STEP 2: Rhythm Analysis
# ─────────────────────────────────────────────────
step(2, "Analizando ritmo (BPM + compas)")
t0 = time.time()

from src.rhythm_analyzer import RhythmAnalyzer
rhythm = RhythmAnalyzer()
rhythm_info = rhythm.analyze(audio_data['waveform'], audio_data['sr'])

info("BPM detectado", f"{rhythm_info['bpm']}")
info("Compas", rhythm_info['time_signature'])
info("Dur. negra", f"{rhythm_info['quarter_duration']:.3f}s")
info("Beats detectados", f"{len(rhythm_info['beat_times'])}")
info("Tiempo", f"{time.time()-t0:.2f}s")

# Beat visualization
print(f"\n    {DIM}Beats detectados (|):{RESET}")
total_dur = len(mixed) / SR
beat_line = [' '] * 60
for bt in rhythm_info['beat_times']:
    pos = int(bt / total_dur * 59)
    if 0 <= pos < 60:
        beat_line[pos] = '|'
print(f"    {''.join(beat_line)}")
print(f"    {DIM}0s{'':>27s}{total_dur:.0f}s{RESET}")

# ─────────────────────────────────────────────────
# STEP 3: Pitch Detection
# ─────────────────────────────────────────────────
step(3, "Detectando notas (basic-pitch AMT)")
t0 = time.time()

from src.pitch_detector import PitchDetector
detector = PitchDetector()
notes = detector.detect(
    audio_data['waveform'],
    audio_data['sr'],
    instrument='piano',
    filepath=audio_data['filepath'],
)

info("Notas detectadas", f"{len(notes)}")
info("Tiempo", f"{time.time()-t0:.2f}s")

if notes:
    pitches = [n.pitch_midi for n in notes]
    info("Rango", f"{note_display(min(pitches))} - {note_display(max(pitches))}")
    info("Velocidad media", f"{int(np.mean([n.velocity for n in notes]))}")

    # Pitch histogram
    print(f"\n    {DIM}Distribucion de notas:{RESET}")
    from collections import Counter
    pitch_counts = Counter(n.pitch_midi for n in notes)
    top_notes = pitch_counts.most_common(10)
    max_count = top_notes[0][1] if top_notes else 1
    for midi, count in sorted(top_notes, key=lambda x: x[0]):
        bar_chart(note_display(midi), count, max_count)

    # Piano roll ASCII
    print(f"\n    {DIM}Piano roll (tiempo -> notas):{RESET}")
    min_p = min(pitches)
    max_p = max(pitches)
    total_time = max(n.end_time for n in notes)
    rows = min(max_p - min_p + 1, 20)
    pitch_step = max(1, (max_p - min_p) // rows)

    grid_w = 60
    for row in range(rows, -1, -1):
        p = min_p + row * pitch_step
        line = [' '] * grid_w
        for n in notes:
            if abs(n.pitch_midi - p) < pitch_step:
                start = int(n.start_time / total_time * (grid_w - 1))
                end = min(int(n.end_time / total_time * (grid_w - 1)), grid_w - 1)
                for x in range(start, end + 1):
                    line[x] = '█'
        label = note_display(p)
        print(f"    {label:>4s} |{''.join(line)}|")
    print(f"    {'':>4s} +{'─'*grid_w}+")

# ─────────────────────────────────────────────────
# STEP 4: Quantization
# ─────────────────────────────────────────────────
step(4, "Cuantizando al grid ritmico")

notes = rhythm.quantize(notes, rhythm_info)

dur_counts = Counter()
for n in notes:
    if n.quantized_duration:
        dur_counts[n.quantized_duration] += 1

dur_names = {
    4.0: 'Redonda', 3.0: 'Blanca.', 2.0: 'Blanca',
    1.5: 'Negra.', 1.0: 'Negra', 0.75: 'Corchea.',
    0.5: 'Corchea', 0.25: 'Semicorchea', 1/3: 'Tresillo'
}

print(f"\n    {DIM}Duraciones detectadas:{RESET}")
for dur, count in sorted(dur_counts.items(), reverse=True):
    name = dur_names.get(dur, f'{dur:.2f}')
    bar_chart(name[:8], count, max(dur_counts.values()))

# ─────────────────────────────────────────────────
# STEP 5: Score Building
# ─────────────────────────────────────────────────
step(5, "Construyendo partitura music21 (modo Piano)")
t0 = time.time()

from src.score_builder import ScoreBuilder
builder = ScoreBuilder()
score = builder.build(
    notes=notes,
    instrument_mode='piano',
    title='Demo Piano',
    bpm=rhythm_info['bpm'],
    time_signature=rhythm_info['time_signature'],
    audio_data=audio_data,
)

treble_notes = [n for n in notes if n.pitch_midi >= 60]
bass_notes = [n for n in notes if n.pitch_midi < 60]

info("Tonalidad", builder.detected_key or "C major")
info("Partes", f"{len(score.parts)}")
info("Notas clave Sol", f"{len(treble_notes)} (mano derecha)")
info("Notas clave Fa", f"{len(bass_notes)} (mano izquierda)")
info("Compases", f"{len(list(score.parts[0].getElementsByClass('Measure')))}")
info("Tiempo", f"{time.time()-t0:.2f}s")

# ─────────────────────────────────────────────────
# STEP 6: Export
# ─────────────────────────────────────────────────
step(6, "Exportando archivos")

out_dir = os.path.join(os.path.dirname(__file__), 'demo_output')
os.makedirs(out_dir, exist_ok=True)

# MusicXML
t0 = time.time()
from src.exporters import MusicXMLExporter, MIDIExporter
xml_path = os.path.join(out_dir, 'demo_piano.musicxml')
MusicXMLExporter.export(score, xml_path)
xml_size = os.path.getsize(xml_path)
info("MusicXML", f"demo_piano.musicxml ({xml_size/1024:.0f} KB) [{time.time()-t0:.2f}s]")

# MIDI
t0 = time.time()
mid_path = os.path.join(out_dir, 'demo_piano.mid')
MIDIExporter.export(score, mid_path)
mid_size = os.path.getsize(mid_path)
info("MIDI", f"demo_piano.mid ({mid_size/1024:.1f} KB) [{time.time()-t0:.2f}s]")

# PDF
t0 = time.time()
from src.pdf_exporter import PDFExporter
pdf_path = os.path.join(out_dir, 'demo_piano.pdf')
try:
    PDFExporter.export(xml_path, pdf_path, score=score)
    pdf_size = os.path.getsize(pdf_path)
    info("PDF", f"demo_piano.pdf ({pdf_size/1024:.0f} KB) [{time.time()-t0:.2f}s]")
except Exception as e:
    info("PDF", f"No disponible: {e}")

# ─────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────
header("RESUMEN FINAL")

print(f"  {BOLD}Entrada:{RESET}")
info("Audio", "demo_piano.wav (16s, piano sintetico)")
print()
print(f"  {BOLD}Analisis:{RESET}")
info("Tonalidad", builder.detected_key or "C major")
info("Tempo", f"{rhythm_info['bpm']} BPM")
info("Compas", rhythm_info['time_signature'])
info("Notas", f"{len(notes)} detectadas")
print()
print(f"  {BOLD}Salida:{RESET}")
for f in sorted(os.listdir(out_dir)):
    fpath = os.path.join(out_dir, f)
    size = os.path.getsize(fpath)
    icon = {'pdf': '📄', 'musicxml': '🎼', 'mid': '🎹'}.get(f.split('.')[-1], '📁')
    print(f"    {icon}  {f:30s} {size/1024:>6.1f} KB")

print(f"\n  {GREEN}{BOLD}Pipeline completado exitosamente.{RESET}\n")
