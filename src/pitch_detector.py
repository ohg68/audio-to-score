"""
Pitch detection using basic-pitch (Spotify) for polyphonic audio
and pYIN (via librosa) for monophonic voice.

Improvements over original:
  - Uses real confidence scores from basic-pitch (not hardcoded 1.0)
  - Configurable confidence threshold to filter ghost notes
  - Instrument-specific pYIN frequency ranges
  - Octave error correction for pYIN
  - Higher frame resolution (hop_length=256, frame_length=4096)
  - Onset refinement using librosa.onset
  - Note merging to de-duplicate rapid repeated detections
"""

import sys
import numpy as np
import librosa


# Instrument-specific frequency ranges for pYIN (Hz)
PYIN_FREQ_RANGES = {
    'voice':  (82,   1047),   # E2 – C6
    'piano':  (27,   4186),   # A0 – C8
    'guitar': (82,   1175),   # E2 – D6
    'song':   (82,   1047),   # Same as voice (used for vocal stem)
    'auto':   (27,   4186),   # Full range
}


class NoteEvent:
    """Represents a single detected note."""

    __slots__ = (
        'pitch_midi', 'start_time', 'end_time', 'velocity', 'confidence',
        'duration', 'quantized_duration', 'quantized_start',
    )

    def __init__(self, pitch_midi, start_time, end_time, velocity=80, confidence=1.0):
        self.pitch_midi = int(round(pitch_midi))
        self.start_time = float(start_time)
        self.end_time = float(end_time)
        self.duration = self.end_time - self.start_time
        self.velocity = int(min(127, max(1, velocity)))
        self.confidence = float(confidence)
        # Filled later by quantization
        self.quantized_duration = None
        self.quantized_start = None

    def __repr__(self):
        from music21 import pitch as m21pitch
        try:
            name = m21pitch.Pitch(midi=self.pitch_midi).nameWithOctave
        except Exception:
            name = f'MIDI {self.pitch_midi}'
        return (
            f'NoteEvent({name}, {self.start_time:.3f}-{self.end_time:.3f}s, '
            f'vel={self.velocity}, conf={self.confidence:.2f})'
        )


class PitchDetector:
    """Detect notes from audio using basic-pitch or pYIN."""

    # Minimum note duration in seconds (filter very short artifacts)
    MIN_NOTE_DURATION = 0.04
    # Minimum velocity to keep
    MIN_VELOCITY = 10
    # Default confidence threshold — notes below this are filtered out
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    # Merge threshold — consecutive same-pitch notes closer than this (seconds) are merged
    MERGE_THRESHOLD = 0.03

    def detect(self, waveform, sr, instrument='auto', filepath=None,
               confidence_threshold=None):
        """
        Detect notes in audio.

        Parameters
        ----------
        waveform : np.ndarray
            Mono audio waveform.
        sr : int
            Sample rate.
        instrument : str
            'piano', 'guitar', 'voice', or 'auto'.
        filepath : str or None
            Path to audio file (needed for basic-pitch).
        confidence_threshold : float or None
            Minimum confidence to keep a note (0.0-1.0).
            Defaults to DEFAULT_CONFIDENCE_THRESHOLD.

        Returns
        -------
        list[NoteEvent]
            Detected notes sorted by start time.
        """
        if confidence_threshold is None:
            confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD

        if instrument == 'voice':
            notes = self._detect_pyin(waveform, sr, instrument)
        else:
            notes = self._detect_basic_pitch(filepath or waveform, sr)
            # Fallback to pYIN if basic-pitch found nothing
            if not notes:
                notes = self._detect_pyin(waveform, sr, instrument)

        # Filter by confidence threshold
        notes = [n for n in notes if n.confidence >= confidence_threshold]

        # Filter short/quiet notes
        notes = [
            n for n in notes
            if n.duration >= self.MIN_NOTE_DURATION and n.velocity >= self.MIN_VELOCITY
        ]

        # Merge repeated same-pitch detections (de-duplicate)
        notes = self._merge_repeated_notes(notes)

        # Refine onset times using librosa onset detection
        notes = self._refine_onsets(notes, waveform, sr)

        # Fix octave errors (mainly for pYIN)
        if instrument == 'voice':
            notes = self._fix_octave_errors(notes)

        # Sort by start time, then pitch
        notes.sort(key=lambda n: (n.start_time, n.pitch_midi))

        return notes

    def _detect_basic_pitch(self, audio_path_or_waveform, sr):
        """
        Use Spotify's basic-pitch for polyphonic AMT.

        Now uses real confidence scores instead of hardcoding 1.0.

        Parameters
        ----------
        audio_path_or_waveform : str or np.ndarray
            File path to audio (preferred) or waveform array.
        sr : int
            Sample rate (used for pYIN fallback).

        Returns list of NoteEvent.
        """
        try:
            from basic_pitch.inference import predict
        except ImportError:
            print(
                "  Warning: basic-pitch not installed. Install with: "
                "pip install basic-pitch\n"
                "  Falling back to pYIN (monophonic only).",
                file=sys.stderr,
            )
            waveform = audio_path_or_waveform if isinstance(audio_path_or_waveform, np.ndarray) else None
            return self._detect_pyin(waveform, sr, 'auto') if waveform is not None else []

        # basic-pitch.predict() expects a file path, not a numpy array
        audio_path = audio_path_or_waveform
        if isinstance(audio_path, np.ndarray):
            # No filepath available — write to temp file
            import tempfile, soundfile as sf
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sf.write(tmp.name, audio_path, sr)
            audio_path = tmp.name

        try:
            model_output, midi_data, note_events = predict(audio_path)
        except Exception as e:
            print(
                f"  Warning: basic-pitch failed: {e}\n"
                f"  Falling back to pYIN.",
                file=sys.stderr,
            )
            waveform = audio_path_or_waveform if isinstance(audio_path_or_waveform, np.ndarray) else None
            return self._detect_pyin(waveform, sr, 'auto') if waveform is not None else []

        # Extract note confidence from model_output (pitch posterior probabilities)
        # model_output is a dict with 'note' key containing frame-level note posteriors
        note_posteriors = None
        if isinstance(model_output, dict) and 'note' in model_output:
            note_posteriors = model_output['note']

        notes = []
        for onset, offset, pitch_midi, velocity, pitch_bend in note_events:
            vel_int = int(velocity * 127) if velocity <= 1.0 else int(velocity)

            # Compute confidence from note posteriors if available
            confidence = self._compute_basic_pitch_confidence(
                note_posteriors, onset, offset, pitch_midi, sr
            )

            note = NoteEvent(
                pitch_midi=pitch_midi,
                start_time=onset,
                end_time=offset,
                velocity=vel_int,
                confidence=confidence,
            )
            notes.append(note)

        return notes

    def _compute_basic_pitch_confidence(self, note_posteriors, onset, offset,
                                         pitch_midi, sr):
        """
        Compute confidence for a basic-pitch note from frame-level posteriors.

        If posteriors are not available, estimate confidence from velocity.
        """
        if note_posteriors is not None:
            try:
                # basic-pitch uses ~256 hop_length at 22050 sr internally
                # note_posteriors shape: (n_frames, 264) covering MIDI 21-108
                bp_hop = 256
                bp_sr = 22050
                onset_frame = int(onset * bp_sr / bp_hop)
                offset_frame = int(offset * bp_sr / bp_hop)
                midi_idx = int(round(pitch_midi)) - 21  # basic-pitch starts at MIDI 21

                if (0 <= midi_idx < note_posteriors.shape[-1] and
                        onset_frame < note_posteriors.shape[0]):
                    offset_frame = min(offset_frame, note_posteriors.shape[0])
                    if offset_frame > onset_frame:
                        frame_probs = note_posteriors[onset_frame:offset_frame, midi_idx]
                        confidence = float(np.mean(frame_probs))
                        return np.clip(confidence, 0.0, 1.0)
            except (IndexError, TypeError, AttributeError):
                pass

        # Fallback: estimate confidence from velocity (normalized)
        return 0.7  # Moderate default when posteriors unavailable

    def _detect_pyin(self, waveform, sr, instrument='auto'):
        """
        Use pYIN for monophonic pitch detection (best for voice).

        Uses instrument-specific frequency ranges and higher resolution frames.

        Returns list of NoteEvent.
        """
        # Get instrument-specific frequency range
        freq_range = PYIN_FREQ_RANGES.get(instrument, PYIN_FREQ_RANGES['auto'])
        fmin, fmax = freq_range

        # Higher resolution: frame_length=4096 for better frequency resolution at 44100 Hz
        # hop_length=256 for ~5.8ms time resolution at 44100 Hz
        frame_length = 4096
        hop_length = 256

        # pYIN returns fundamental frequency estimates
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        if f0 is None or len(f0) == 0:
            return []

        times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)

        # Compute per-frame amplitude for velocity estimation
        rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
        if len(rms) < len(f0):
            rms = np.pad(rms, (0, len(f0) - len(rms)))
        rms = rms[:len(f0)]
        rms_max = np.max(rms) if np.max(rms) > 0 else 1.0

        # Group consecutive voiced frames into notes
        notes = []
        in_note = False
        note_start_idx = 0
        prev_midi = -1

        for i in range(len(f0)):
            freq = f0[i]
            voiced = voiced_flag[i] if voiced_flag is not None else (not np.isnan(freq))

            if voiced and not np.isnan(freq) and freq > 0:
                midi_val = int(round(librosa.hz_to_midi(freq)))

                if not in_note:
                    # Start new note
                    in_note = True
                    note_start_idx = i
                    prev_midi = midi_val
                elif abs(midi_val - prev_midi) >= 1:
                    # Pitch changed — end previous note, start new one
                    note = self._create_note_from_frames(
                        note_start_idx, i, prev_midi, times, rms, rms_max, voiced_probs,
                    )
                    if note:
                        notes.append(note)
                    note_start_idx = i
                    prev_midi = midi_val
            else:
                if in_note:
                    # End note
                    note = self._create_note_from_frames(
                        note_start_idx, i, prev_midi, times, rms, rms_max, voiced_probs,
                    )
                    if note:
                        notes.append(note)
                    in_note = False

        # Handle final note
        if in_note:
            note = self._create_note_from_frames(
                note_start_idx, len(f0), prev_midi, times, rms, rms_max, voiced_probs,
            )
            if note:
                notes.append(note)

        return notes

    def _create_note_from_frames(self, start_idx, end_idx, midi_val, times, rms, rms_max, voiced_probs):
        """Create a NoteEvent from a range of frames."""
        if end_idx <= start_idx:
            return None

        start_time = times[start_idx]
        end_time = times[min(end_idx, len(times) - 1)]

        if end_time - start_time < self.MIN_NOTE_DURATION:
            return None

        # Velocity from RMS amplitude
        seg_rms = rms[start_idx:end_idx]
        mean_rms = np.mean(seg_rms) if len(seg_rms) > 0 else 0
        velocity = int(np.clip(mean_rms / rms_max * 120 + 7, 1, 127))

        # Confidence from voiced probability
        if voiced_probs is not None:
            seg_conf = voiced_probs[start_idx:end_idx]
            confidence = float(np.mean(seg_conf)) if len(seg_conf) > 0 else 0.5
        else:
            confidence = 0.8

        return NoteEvent(
            pitch_midi=midi_val,
            start_time=start_time,
            end_time=end_time,
            velocity=velocity,
            confidence=confidence,
        )

    def detect_pyin(self, waveform, sr, instrument='voice', confidence_threshold=None):
        """
        Public pYIN detection for monophonic audio (e.g., isolated vocal stem).

        Parameters
        ----------
        waveform : np.ndarray
        sr : int
        instrument : str
        confidence_threshold : float or None

        Returns
        -------
        list[NoteEvent]
        """
        if confidence_threshold is None:
            confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD

        notes = self._detect_pyin(waveform, sr, instrument)
        return self.post_process(notes, waveform, sr,
                                 confidence_threshold=confidence_threshold,
                                 fix_octaves=True)

    def detect_basic_pitch(self, filepath_or_waveform, sr,
                            confidence_threshold=None):
        """
        Public basic-pitch detection for polyphonic audio (e.g., accompaniment).

        Parameters
        ----------
        filepath_or_waveform : str or np.ndarray
        sr : int
        confidence_threshold : float or None

        Returns
        -------
        list[NoteEvent]
        """
        if confidence_threshold is None:
            confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD

        notes = self._detect_basic_pitch(filepath_or_waveform, sr)
        return self.post_process(notes, None, sr,
                                 confidence_threshold=confidence_threshold,
                                 fix_octaves=False)

    def post_process(self, notes, waveform, sr,
                     confidence_threshold=None, fix_octaves=False):
        """
        Apply standard post-processing to detected notes.

        Filters by confidence and duration, merges duplicates,
        refines onsets, and optionally fixes octave errors.

        Parameters
        ----------
        notes : list[NoteEvent]
        waveform : np.ndarray or None
            If None, onset refinement is skipped.
        sr : int
        confidence_threshold : float or None
        fix_octaves : bool

        Returns
        -------
        list[NoteEvent]
        """
        if confidence_threshold is None:
            confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD

        # Filter by confidence threshold
        notes = [n for n in notes if n.confidence >= confidence_threshold]

        # Filter short/quiet notes
        notes = [
            n for n in notes
            if n.duration >= self.MIN_NOTE_DURATION and n.velocity >= self.MIN_VELOCITY
        ]

        # Merge repeated same-pitch detections
        notes = self._merge_repeated_notes(notes)

        # Refine onset times
        if waveform is not None and len(waveform) > 0:
            notes = self._refine_onsets(notes, waveform, sr)

        # Fix octave errors (mainly for pYIN)
        if fix_octaves:
            notes = self._fix_octave_errors(notes)

        # Sort by start time, then pitch
        notes.sort(key=lambda n: (n.start_time, n.pitch_midi))

        return notes

    def _fix_octave_errors(self, notes):
        """
        Correct common octave errors from pYIN.

        Detects notes that jump ±12 semitones from the local context
        and corrects them back if confidence is low.
        """
        if len(notes) < 3:
            return notes

        pitches = np.array([n.pitch_midi for n in notes])
        median_pitch = np.median(pitches)

        for i, note in enumerate(notes):
            # Check if this note is exactly ±12 semitones from median
            diff_from_median = note.pitch_midi - median_pitch
            if abs(abs(diff_from_median) - 12) < 1.0 and note.confidence < 0.7:
                # Check neighboring notes
                neighbors = []
                if i > 0:
                    neighbors.append(notes[i - 1].pitch_midi)
                if i < len(notes) - 1:
                    neighbors.append(notes[i + 1].pitch_midi)

                if neighbors:
                    avg_neighbor = np.mean(neighbors)
                    # If the jump to neighbors is ~12 semitones, likely octave error
                    diff_to_neighbors = note.pitch_midi - avg_neighbor
                    if abs(diff_to_neighbors) >= 10 and abs(diff_to_neighbors) <= 14:
                        # Correct octave: move toward neighbors
                        if diff_to_neighbors > 0:
                            note.pitch_midi -= 12
                        else:
                            note.pitch_midi += 12

        return notes

    def _merge_repeated_notes(self, notes):
        """
        Merge consecutive notes of the same pitch that are very close together.

        This eliminates "stuttering" caused by duplicate detections where
        basic-pitch or pYIN detects the same note as two rapid events.
        """
        if len(notes) < 2:
            return notes

        # Sort by start time, then pitch
        notes.sort(key=lambda n: (n.start_time, n.pitch_midi))

        merged = [notes[0]]
        for note in notes[1:]:
            prev = merged[-1]
            # Same pitch and gap < merge threshold
            gap = note.start_time - prev.end_time
            if (note.pitch_midi == prev.pitch_midi and
                    gap < self.MERGE_THRESHOLD and gap >= -0.01):
                # Merge: extend previous note to cover both
                prev.end_time = note.end_time
                prev.duration = prev.end_time - prev.start_time
                prev.velocity = max(prev.velocity, note.velocity)
                prev.confidence = max(prev.confidence, note.confidence)
            else:
                merged.append(note)

        return merged

    def _refine_onsets(self, notes, waveform, sr):
        """
        Refine note start times using librosa onset detection.

        For each detected note, looks for the nearest onset within ±50ms
        and adjusts the start time. This corrects timing misalignment
        from basic-pitch and pYIN frame-based detection.
        """
        if not notes or len(waveform) == 0:
            return notes

        try:
            onsets = librosa.onset.onset_detect(
                y=waveform, sr=sr, units='time', backtrack=True,
                hop_length=256,
            )
        except Exception:
            return notes

        if len(onsets) == 0:
            return notes

        onset_arr = np.array(onsets)
        max_shift = 0.05  # ±50ms maximum correction

        for note in notes:
            # Find nearest onset
            diffs = np.abs(onset_arr - note.start_time)
            nearest_idx = np.argmin(diffs)
            nearest_onset = onset_arr[nearest_idx]
            shift = nearest_onset - note.start_time

            if abs(shift) <= max_shift:
                # Adjust start time while keeping duration constant
                note.start_time = nearest_onset
                note.end_time = note.start_time + note.duration

        return notes
