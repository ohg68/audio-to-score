"""
Generate a test WAV file: 8 bars of a simple piano-like melody in C major, 4/4 at 120 BPM.

Melody (right hand): C4 E4 G4 C5 | B4 G4 E4 C4 | D4 F4 A4 D5 | C5 A4 F4 D4
                      E4 G4 B4 E5 | D5 B4 G4 E4 | C4 E4 G4 C5 | C5 - - -
Bass (left hand):     C3 (whole) | C3 | D3 | D3 | E3 | E3 | C3 | C3
"""

import numpy as np
import soundfile as sf
import os

SR = 22050
BPM = 120
QUARTER = 60.0 / BPM  # 0.5s per quarter note

def sine_note(freq, duration, sr=SR, velocity=0.7):
    """Generate a piano-like tone with exponential decay."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Fundamental + harmonics for richer sound
    wave = (
        np.sin(2 * np.pi * freq * t) * 1.0 +
        np.sin(2 * np.pi * freq * 2 * t) * 0.5 +
        np.sin(2 * np.pi * freq * 3 * t) * 0.25 +
        np.sin(2 * np.pi * freq * 4 * t) * 0.125
    )
    # Exponential decay envelope
    envelope = np.exp(-t * 3.0 / duration)
    # Attack (short fade-in)
    attack_samples = min(int(0.01 * sr), len(t))
    envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)
    return wave * envelope * velocity


def midi_to_freq(midi):
    return 440.0 * (2 ** ((midi - 69) / 12.0))


# Define melody as (midi_note, duration_in_quarters)
# C4=60, D4=62, E4=64, F4=65, G4=67, A4=69, B4=71, C5=72
# C3=48, D3=50, E3=52

melody = [
    # Bar 1
    (60, 1), (64, 1), (67, 1), (72, 1),
    # Bar 2
    (71, 1), (67, 1), (64, 1), (60, 1),
    # Bar 3
    (62, 1), (65, 1), (69, 1), (74, 1),
    # Bar 4
    (72, 1), (69, 1), (65, 1), (62, 1),
    # Bar 5
    (64, 1), (67, 1), (71, 1), (76, 1),
    # Bar 6
    (74, 1), (71, 1), (67, 1), (64, 1),
    # Bar 7
    (60, 1), (64, 1), (67, 1), (72, 1),
    # Bar 8 - whole note
    (72, 4),
]

bass = [
    # Bar 1-2
    (48, 4), (48, 4),
    # Bar 3-4
    (50, 4), (50, 4),
    # Bar 5-6
    (52, 4), (52, 4),
    # Bar 7-8
    (48, 4), (48, 4),
]

def render_voice(notes, velocity=0.6):
    """Render a list of (midi, dur_quarters) to audio."""
    audio = np.zeros(0)
    for midi, dur_q in notes:
        duration = dur_q * QUARTER
        freq = midi_to_freq(midi)
        tone = sine_note(freq, duration, velocity=velocity)
        audio = np.concatenate([audio, tone])
    return audio

# Render
melody_audio = render_voice(melody, velocity=0.7)
bass_audio = render_voice(bass, velocity=0.5)

# Ensure same length
max_len = max(len(melody_audio), len(bass_audio))
melody_audio = np.pad(melody_audio, (0, max_len - len(melody_audio)))
bass_audio = np.pad(bass_audio, (0, max_len - len(bass_audio)))

# Mix
mixed = melody_audio + bass_audio
# Normalize
mixed = mixed / np.max(np.abs(mixed)) * 0.9

# Save
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, 'sample_audio', 'test_piano_8bars.wav')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

sf.write(output_path, mixed.astype(np.float32), SR)
print(f"Generated: {output_path}")
print(f"Duration:  {len(mixed)/SR:.1f}s")
print(f"Tempo:     {BPM} BPM")
print(f"Bars:      8 (4/4)")
print(f"Key:       C major")
