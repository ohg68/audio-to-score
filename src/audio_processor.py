"""
Audio loading, normalization, noise gate, bandpass filtering,
and optional source separation.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt


TARGET_SR = 44100  # Increased from 22050 for better frequency resolution


# Instrument-specific frequency ranges for bandpass filtering (Hz)
INSTRUMENT_FREQ_RANGES = {
    'piano':  (27,   4200),   # A0 – C8
    'guitar': (80,   1200),   # E2 – D6
    'voice':  (80,   1100),   # E2 – C6
    'song':   (27,   4200),   # Full range (dual pipeline)
    'auto':   (27,   4200),   # Full range
}


class AudioProcessor:
    """Load, normalize, filter, and optionally separate audio sources."""

    def load(self, filepath, separate=False, instrument='auto'):
        """
        Load an audio file and return processed waveform data.

        Parameters
        ----------
        filepath : str
            Path to audio file (WAV, MP3, FLAC, OGG, M4A, AAC).
        separate : bool
            If True, use Demucs for source separation.
        instrument : str
            Target instrument ('piano', 'guitar', 'voice', 'auto').

        Returns
        -------
        dict with keys:
            'waveform': np.ndarray (mono, float32, normalized)
            'sr': int (sample rate)
            'duration': float (seconds)
            'filepath': str
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        # Load and convert to mono at target sample rate
        try:
            waveform, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load '{path.name}'. File may be corrupted: {e}"
            )

        if len(waveform) == 0:
            raise RuntimeError(f"Audio file '{path.name}' is empty.")

        # Normalize peak amplitude to [-1, 1]
        waveform = self._normalize(waveform)

        # Apply noise gate to suppress low-level noise
        waveform = self._noise_gate(waveform, sr)

        # Apply bandpass filter for the target instrument
        waveform = self._bandpass_filter(waveform, sr, instrument)

        # Re-normalize after filtering
        waveform = self._normalize(waveform)

        duration = len(waveform) / sr

        result = {
            'waveform': waveform,
            'sr': sr,
            'duration': duration,
            'filepath': str(path),
        }

        # Source separation with Demucs
        if separate:
            result = self._separate_sources(result, instrument)

        return result

    def _normalize(self, waveform):
        """Normalize waveform peak to [-1, 1]."""
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
        return waveform.astype(np.float32)

    def _noise_gate(self, waveform, sr, frame_length=2048, hop_length=512):
        """
        Apply a noise gate to suppress low-level noise.

        Uses adaptive threshold based on median RMS energy.
        Applies a smooth fade (5ms) to avoid clicks at gate boundaries.
        """
        rms = librosa.feature.rms(
            y=waveform, frame_length=frame_length, hop_length=hop_length
        )[0]

        if len(rms) == 0 or np.max(rms) == 0:
            return waveform

        # Dynamic threshold: 10% of median RMS
        threshold = np.median(rms) * 0.1

        # Build per-sample gain envelope
        n_frames = len(rms)
        gain = np.ones(len(waveform), dtype=np.float32)

        fade_samples = int(0.005 * sr)  # 5ms fade to avoid clicks

        for i in range(n_frames):
            if rms[i] < threshold:
                start = i * hop_length
                end = min(start + frame_length, len(waveform))
                # Smoothly ramp down to zero
                frame_gain = np.zeros(end - start, dtype=np.float32)
                if fade_samples > 0 and end - start > fade_samples * 2:
                    # Fade in/out at edges
                    frame_gain[:fade_samples] = np.linspace(1, 0, fade_samples)
                    frame_gain[-fade_samples:] = np.linspace(0, 1, fade_samples)
                gain[start:end] = np.minimum(gain[start:end], frame_gain)

        return waveform * gain

    def _bandpass_filter(self, waveform, sr, instrument):
        """
        Apply a bandpass filter based on instrument frequency range.

        Removes frequencies outside the instrument's useful range,
        reducing false pitch detections from noise or harmonics.
        """
        freq_range = INSTRUMENT_FREQ_RANGES.get(instrument, INSTRUMENT_FREQ_RANGES['auto'])
        low_hz, high_hz = freq_range

        # Ensure Nyquist compliance
        nyquist = sr / 2.0
        low = max(low_hz / nyquist, 0.001)
        high = min(high_hz / nyquist, 0.999)

        if low >= high:
            return waveform

        try:
            sos = butter(N=4, Wn=[low, high], btype='band', output='sos')
            filtered = sosfiltfilt(sos, waveform).astype(np.float32)
            return filtered
        except Exception:
            # If filtering fails, return original
            return waveform

    def _separate_sources(self, audio_data, instrument):
        """
        Use Demucs to separate sources, then select the relevant stem.

        Maps instrument -> Demucs stem:
            piano  -> 'other' (Demucs outputs: drums, bass, vocals, other)
            guitar -> 'other'
            voice  -> 'vocals'
            auto   -> 'other' (melodic/harmonic content)
        """
        if not self._check_demucs():
            print(
                "  Warning: Demucs not installed. Install with: pip install demucs\n"
                "  Proceeding without source separation.",
                file=sys.stderr,
            )
            return audio_data

        stem_map = {
            'piano': 'other',
            'guitar': 'other',
            'voice': 'vocals',
            'auto': 'other',
        }
        target_stem = stem_map.get(instrument, 'other')

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write waveform to temp WAV for Demucs
            tmp_wav = os.path.join(tmpdir, 'input.wav')
            sf.write(tmp_wav, audio_data['waveform'], audio_data['sr'])

            # Run Demucs
            try:
                subprocess.run(
                    [
                        sys.executable, '-m', 'demucs',
                        '--two-stems', 'vocals' if target_stem == 'vocals' else 'vocals',
                        '-o', tmpdir,
                        '-n', 'htdemucs',
                        tmp_wav,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                print(
                    f"  Warning: Demucs failed: {e.stderr.strip()}\n"
                    f"  Proceeding without source separation.",
                    file=sys.stderr,
                )
                return audio_data

            # Find the separated stem
            # Demucs outputs to: <out_dir>/htdemucs/input/<stem>.wav
            stem_dir = Path(tmpdir) / 'htdemucs' / 'input'
            stem_file = stem_dir / f'{target_stem}.wav'

            if target_stem == 'other' and not stem_file.exists():
                # If 'other' not found, try 'no_vocals'
                stem_file = stem_dir / 'no_vocals.wav'

            if not stem_file.exists():
                # Fallback: list what's available
                available = list(stem_dir.glob('*.wav')) if stem_dir.exists() else []
                names = [f.stem for f in available]
                print(
                    f"  Warning: Stem '{target_stem}' not found. "
                    f"Available: {names}. Using original audio.",
                    file=sys.stderr,
                )
                return audio_data

            # Load the separated stem
            separated, sr = librosa.load(str(stem_file), sr=TARGET_SR, mono=True)
            separated = self._normalize(separated)

            audio_data['waveform'] = separated
            audio_data['duration'] = len(separated) / sr

        return audio_data

    def separate_sources_dual(self, audio_data):
        """
        Separate audio into vocals and accompaniment using Demucs CLI.

        Returns both stems for independent processing in 'song' mode.
        Uses CLI (subprocess) to avoid importing torch in the main process,
        which prevents circular import issues in Streamlit Cloud.

        Parameters
        ----------
        audio_data : dict
            Audio data dict from load() with 'waveform' and 'sr' keys.

        Returns
        -------
        dict with keys:
            'vocals': np.ndarray (mono, float32, normalized)
            'accompaniment': np.ndarray (mono, float32, normalized)
            'sr': int
        """
        if not self._check_demucs():
            raise RuntimeError(
                "Demucs not installed. Required for song mode.\n"
                "Install with: pip install demucs"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write waveform to temp WAV for Demucs
            tmp_wav = os.path.join(tmpdir, 'input.wav')
            sf.write(tmp_wav, audio_data['waveform'], audio_data['sr'])

            # Run Demucs CLI with --two-stems vocals
            # This produces both vocals.wav and no_vocals.wav
            try:
                result = subprocess.run(
                    [
                        sys.executable, '-m', 'demucs',
                        '--two-stems', 'vocals',
                        '-o', tmpdir,
                        '-n', 'htdemucs',
                        tmp_wav,
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Demucs separation failed: {e.stderr.strip()}"
                )

            # Find separated stems
            # Demucs outputs to: <out_dir>/htdemucs/input/vocals.wav + no_vocals.wav
            stem_dir = Path(tmpdir) / 'htdemucs' / 'input'

            vocals_file = stem_dir / 'vocals.wav'
            no_vocals_file = stem_dir / 'no_vocals.wav'

            if not vocals_file.exists() or not no_vocals_file.exists():
                available = list(stem_dir.glob('*.wav')) if stem_dir.exists() else []
                names = [f.stem for f in available]
                raise RuntimeError(
                    f"Demucs did not produce expected stems. "
                    f"Available: {names}"
                )

            # Load separated stems
            vocals_np, _ = librosa.load(str(vocals_file), sr=TARGET_SR, mono=True)
            accomp_np, _ = librosa.load(str(no_vocals_file), sr=TARGET_SR, mono=True)

            # Apply bandpass filters per stem
            vocals_np = self._bandpass_filter(vocals_np, TARGET_SR, 'voice')
            accomp_np = self._bandpass_filter(accomp_np, TARGET_SR, 'piano')

            # Normalize each stem
            vocals_np = self._normalize(vocals_np)
            accomp_np = self._normalize(accomp_np)

            return {
                'vocals': vocals_np,
                'accompaniment': accomp_np,
                'sr': TARGET_SR,
            }

    def _check_demucs(self):
        """Check if Demucs is installed."""
        try:
            import demucs  # noqa: F401
            return True
        except ImportError:
            return False
