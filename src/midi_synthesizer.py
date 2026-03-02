"""
MIDI-to-audio synthesizer for score verification playback.

Provides three synthesis backends (automatic fallback):
  1. FluidSynth (via pretty_midi) — high-quality SoundFont rendering
  2. pretty_midi.synthesize() — built-in synth with custom piano waveform
  3. Numpy fallback — custom additive piano synthesis with harmonics + ADSR

The best available backend is selected automatically.
"""

import sys
import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SR = 44100

# Piano harmonic profile: (harmonic_number, relative_amplitude)
# Based on acoustic piano spectral analysis
PIANO_HARMONICS = [
    (1, 1.00),     # fundamental
    (2, 0.55),     # octave
    (3, 0.30),     # octave + fifth
    (4, 0.15),     # 2 octaves
    (5, 0.08),     # 2 octaves + major third
    (6, 0.04),     # 2 octaves + fifth
    (7, 0.02),     # higher partial (adds shimmer)
]

# Sum of amplitudes for normalization
_HARMONICS_SUM = sum(amp for _, amp in PIANO_HARMONICS)

# ADSR envelope parameters (seconds)
ADSR_ATTACK = 0.008         # 8ms — fast piano hammer attack
ADSR_DECAY = 0.12           # 120ms decay to sustain level
ADSR_SUSTAIN_LEVEL = 0.55   # sustain amplitude ratio
ADSR_RELEASE = 0.25         # 250ms release after note-off


# ---------------------------------------------------------------------------
# MidiSynthesizer class
# ---------------------------------------------------------------------------

class MidiSynthesizer:
    """
    Synthesize piano audio from MIDI files or note data.

    Usage
    -----
    >>> synth = MidiSynthesizer()
    >>> audio = synth.synthesize_from_midi("score.mid")
    >>> # or
    >>> audio = synth.synthesize_from_notes(notes_data)
    """

    @staticmethod
    def _check_fluidsynth():
        """Check if pyfluidsynth is available for high-quality synthesis."""
        try:
            import fluidsynth  # noqa: F401
            return True
        except ImportError:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesize_from_midi(self, midi_path, sr=DEFAULT_SR):
        """
        Synthesize audio from a MIDI file.

        Tries FluidSynth first, then pretty_midi.synthesize with custom
        piano waveform, then full numpy synthesis.

        Parameters
        ----------
        midi_path : str
            Path to .mid file.
        sr : int
            Output sample rate (default 44100).

        Returns
        -------
        audio : np.ndarray
            Mono float32 waveform, normalized to [-1, 1].
        """
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(midi_path)

        # Strategy 1: FluidSynth (best quality)
        if self._check_fluidsynth():
            try:
                audio = pm.fluidsynth(fs=sr)
                if audio is not None and len(audio) > 0:
                    peak = np.max(np.abs(audio))
                    if peak > 0:
                        audio = audio / peak * 0.92
                    return audio.astype(np.float32)
            except Exception:
                pass  # Fall through

        # Strategy 2: pretty_midi.synthesize() with custom piano wave
        try:
            audio = pm.synthesize(fs=sr, wave=self._piano_wave)
            if audio is not None and len(audio) > 0:
                return self._apply_master_envelope(audio, sr)
        except Exception:
            pass  # Fall through

        # Strategy 3: Full custom numpy synthesis
        notes_data = []
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note_obj in inst.notes:
                notes_data.append({
                    'pitch_midi': note_obj.pitch,
                    'start_time': note_obj.start,
                    'end_time': note_obj.end,
                    'velocity': note_obj.velocity,
                })

        return self.synthesize_from_notes(notes_data, sr=sr)

    def synthesize_from_notes(self, notes_data, sr=DEFAULT_SR):
        """
        Synthesize audio from a list of note dictionaries.

        Uses additive synthesis with piano harmonics and ADSR envelopes.

        Parameters
        ----------
        notes_data : list[dict]
            Each dict has: pitch_midi (int), start_time (float),
            end_time (float), velocity (int), and optionally confidence.
        sr : int
            Sample rate (default 44100).

        Returns
        -------
        audio : np.ndarray
            Mono float32 waveform, normalized to [-1, 1].
        """
        if not notes_data:
            return np.zeros(0, dtype=np.float32)

        # Calculate total duration (add release tail + small padding)
        max_end = max(n['end_time'] for n in notes_data)
        total_samples = int((max_end + ADSR_RELEASE + 0.1) * sr)
        output = np.zeros(total_samples, dtype=np.float64)

        for note in notes_data:
            note_audio = self._render_note(
                pitch_midi=note['pitch_midi'],
                start_time=note['start_time'],
                end_time=note['end_time'],
                velocity=note.get('velocity', 80),
                sr=sr,
            )

            if len(note_audio) == 0:
                continue

            start_sample = int(note['start_time'] * sr)
            end_sample = start_sample + len(note_audio)

            # Ensure we don't exceed buffer
            if end_sample > total_samples:
                note_audio = note_audio[:total_samples - start_sample]
                end_sample = total_samples

            if start_sample < total_samples:
                # Polyphonic mixing: additive
                output[start_sample:end_sample] += note_audio

        # Anti-clipping normalization
        peak = np.max(np.abs(output))
        if peak > 0:
            output = output / peak * 0.92  # 8% headroom

        return output.astype(np.float32)

    @staticmethod
    def save_wav(audio, sr, path):
        """
        Save audio waveform to a WAV file.

        Parameters
        ----------
        audio : np.ndarray
            Audio waveform (float32 or float64).
        sr : int
            Sample rate.
        path : str
            Output file path.
        """
        sf.write(path, audio, sr, subtype='PCM_16')

    def get_engine_name(self):
        """Return the name of the synthesis engine that will be used."""
        if self._check_fluidsynth():
            return "FluidSynth (SoundFont)"

        try:
            import pretty_midi  # noqa: F401
            return "Pretty MIDI (Piano Wave)"
        except ImportError:
            pass

        return "Numpy Piano (Additive Synthesis)"

    # ------------------------------------------------------------------
    # Internal: per-note rendering
    # ------------------------------------------------------------------

    def _render_note(self, pitch_midi, start_time, end_time, velocity, sr):
        """
        Render a single piano note with harmonics and ADSR envelope.

        Parameters
        ----------
        pitch_midi : int
            MIDI pitch number (0-127).
        start_time : float
            Note start in seconds (used only for duration calc).
        end_time : float
            Note end in seconds.
        velocity : int
            MIDI velocity (1-127).
        sr : int
            Sample rate.

        Returns
        -------
        audio : np.ndarray (float64)
        """
        duration = end_time - start_time
        if duration <= 0:
            return np.zeros(0, dtype=np.float64)

        # Total rendered length includes release tail
        total_duration = duration + ADSR_RELEASE
        n_samples = int(total_duration * sr)
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float64)

        t = np.arange(n_samples, dtype=np.float64) / sr

        # Fundamental frequency from MIDI number
        freq = 440.0 * (2.0 ** ((pitch_midi - 69) / 12.0))

        # Build harmonic content
        waveform = np.zeros(n_samples, dtype=np.float64)
        for harmonic_num, amplitude in PIANO_HARMONICS:
            harmonic_freq = freq * harmonic_num

            # Skip harmonics above Nyquist to prevent aliasing
            if harmonic_freq >= sr / 2:
                break

            # Higher harmonics decay faster (piano string physics)
            decay_rate = 1.0 + (harmonic_num - 1) * 0.8
            harmonic_envelope = np.exp(-t * decay_rate / max(duration, 0.05))

            waveform += amplitude * np.sin(
                2.0 * np.pi * harmonic_freq * t
            ) * harmonic_envelope

        # Normalize harmonic sum
        waveform /= _HARMONICS_SUM

        # Apply ADSR envelope
        envelope = self._adsr_envelope(n_samples, duration, sr)

        # Velocity mapping: slightly compressed curve for natural dynamics
        velocity_amp = (max(1, min(127, velocity)) / 127.0) ** 0.7

        return waveform * envelope * velocity_amp

    def _adsr_envelope(self, n_samples, note_duration, sr):
        """
        Generate an ADSR (Attack-Decay-Sustain-Release) envelope.

        Parameters
        ----------
        n_samples : int
            Total number of samples (note + release).
        note_duration : float
            Duration of the note-on period in seconds.
        sr : int
            Sample rate.

        Returns
        -------
        envelope : np.ndarray (float64), values in [0, 1].
        """
        envelope = np.zeros(n_samples, dtype=np.float64)

        attack_samples = int(ADSR_ATTACK * sr)
        decay_samples = int(ADSR_DECAY * sr)
        note_on_samples = int(note_duration * sr)

        # Clamp for very short notes
        attack_samples = min(attack_samples, max(note_on_samples, 1))
        decay_samples = min(decay_samples, max(note_on_samples - attack_samples, 0))

        idx = 0

        # Attack: linear ramp 0 → 1
        if attack_samples > 0:
            end = min(idx + attack_samples, n_samples)
            length = end - idx
            if length > 0:
                envelope[idx:end] = np.linspace(0, 1, length)
                idx = end

        # Decay: exponential curve 1 → sustain_level
        if decay_samples > 0 and idx < n_samples:
            end = min(idx + decay_samples, n_samples)
            length = end - idx
            if length > 0:
                decay_curve = np.exp(-np.linspace(0, 5, length))
                envelope[idx:end] = (
                    ADSR_SUSTAIN_LEVEL + (1.0 - ADSR_SUSTAIN_LEVEL) * decay_curve
                )
                idx = end

        # Sustain: slight natural decay during sustain (piano strings lose energy)
        sustain_end = min(note_on_samples, n_samples)
        if idx < sustain_end:
            length = sustain_end - idx
            sustain_decay = np.exp(-np.linspace(0, 0.5, length))
            envelope[idx:sustain_end] = ADSR_SUSTAIN_LEVEL * sustain_decay
            idx = sustain_end

        # Release: exponential decay → 0
        if idx < n_samples:
            release_start_level = envelope[idx - 1] if idx > 0 else ADSR_SUSTAIN_LEVEL
            length = n_samples - idx
            release_curve = np.exp(-np.linspace(0, 6, length))
            envelope[idx:] = release_start_level * release_curve

        return envelope

    # ------------------------------------------------------------------
    # Internal: pretty_midi custom wave and post-processing
    # ------------------------------------------------------------------

    @staticmethod
    def _piano_wave(phase):
        """
        Custom piano-like waveform for pretty_midi.synthesize().

        Parameters
        ----------
        phase : np.ndarray
            Phase values (as passed to np.sin by pretty_midi).

        Returns
        -------
        waveform : np.ndarray
        """
        result = np.sin(phase)                     # fundamental
        result += 0.55 * np.sin(2 * phase)         # 2nd harmonic
        result += 0.30 * np.sin(3 * phase)         # 3rd harmonic
        result += 0.15 * np.sin(4 * phase)         # 4th harmonic
        result += 0.08 * np.sin(5 * phase)         # 5th harmonic
        result += 0.04 * np.sin(6 * phase)         # 6th harmonic
        result /= _HARMONICS_SUM
        return result

    @staticmethod
    def _apply_master_envelope(audio, sr):
        """
        Apply fade-in/fade-out to pretty_midi.synthesize() output.

        pretty_midi's built-in synthesizer can produce abrupt note endings.
        """
        audio = audio.copy()

        # Fade in (10ms)
        fade_in = int(0.01 * sr)
        if fade_in > 0 and len(audio) > fade_in:
            audio[:fade_in] *= np.linspace(0, 1, fade_in)

        # Fade out (50ms)
        fade_out = int(0.05 * sr)
        if fade_out > 0 and len(audio) > fade_out:
            audio[-fade_out:] *= np.linspace(1, 0, fade_out)

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.92

        return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def synthesize_score(notes_data=None, midi_path=None, sr=DEFAULT_SR):
    """
    Convenience function for synthesizing a score to audio.

    Provide either notes_data or midi_path.

    Parameters
    ----------
    notes_data : list[dict], optional
        Note dictionaries from transcription results.
    midi_path : str, optional
        Path to a MIDI file.
    sr : int
        Sample rate.

    Returns
    -------
    audio : np.ndarray
        Synthesized audio waveform.
    sr : int
        Sample rate used.
    """
    synth = MidiSynthesizer()

    if midi_path:
        audio = synth.synthesize_from_midi(midi_path, sr=sr)
    elif notes_data:
        audio = synth.synthesize_from_notes(notes_data, sr=sr)
    else:
        raise ValueError("Provide either notes_data or midi_path.")

    return audio, sr
