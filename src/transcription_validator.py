"""
Transcription validation and auto-correction.

Compares quantized note transcription against original audio using
chromagram analysis, then applies conservative corrections before
delivering the final score.

Source priority system:
  - Voice:  HPSS harmonic component + CQT chroma (melody-focused)
  - Guitar: Full spectrum + STFT chroma (chord-focused)
  - Piano:  Full spectrum + CQT chroma (wide range)

Correction types:
  - Pitch class mismatch: shift ±1-2 semitones
  - Octave error: shift ±12 semitones
  - Ghost notes: remove spurious detections
  - Missing notes: flag in report (no insertion)
"""

import numpy as np
import librosa


# Chromagram extraction parameters
CHROMA_HOP_LENGTH = 512
CHROMA_N_CHROMA = 12

# Validation thresholds
CHROMA_MATCH_THRESHOLD = 0.4
ONSET_TOLERANCE_SEC = 0.05
GHOST_NOTE_ENERGY_RATIO = 0.15
GHOST_NOTE_SILENCE = 0.05
CORRECTION_CONFIDENCE = 0.6
PITCH_CLASS_THRESHOLD = 0.3
MIN_IMPROVEMENT = 0.005

# Instrument-specific configuration
INSTRUMENT_CONFIGS = {
    'voice': {
        'use_hpss': True,
        'chroma_type': 'cqt',
        'max_polyphony': 1,
        'onset_weight': 0.35,
        'chroma_weight': 0.50,
        'confidence_weight': 0.15,
        'correction_aggression': 0.3,
    },
    'guitar': {
        'use_hpss': False,
        'chroma_type': 'stft',
        'max_polyphony': 6,
        'onset_weight': 0.25,
        'chroma_weight': 0.55,
        'confidence_weight': 0.20,
        'correction_aggression': 0.4,
    },
    'piano': {
        'use_hpss': False,
        'chroma_type': 'cqt',
        'max_polyphony': 10,
        'onset_weight': 0.20,
        'chroma_weight': 0.55,
        'confidence_weight': 0.25,
        'correction_aggression': 0.4,
    },
    'song': {
        'use_hpss': True,
        'chroma_type': 'cqt',
        'max_polyphony': 1,
        'onset_weight': 0.35,
        'chroma_weight': 0.50,
        'confidence_weight': 0.15,
        'correction_aggression': 0.3,
    },
    'auto': {
        'use_hpss': False,
        'chroma_type': 'cqt',
        'max_polyphony': 10,
        'onset_weight': 0.25,
        'chroma_weight': 0.50,
        'confidence_weight': 0.25,
        'correction_aggression': 0.35,
    },
}


class ValidationReport:
    """Container for validation/correction results."""

    def __init__(self):
        self.overall_similarity_before = 0.0
        self.overall_similarity_after = 0.0
        self.confirmed_notes = 0
        self.corrected_notes = 0
        self.removed_ghost_notes = 0
        self.flagged_missing_notes = 0
        self.iterations_run = 0
        self.per_note_details = []

    def to_dict(self):
        return {
            'overall_similarity_before': round(self.overall_similarity_before, 1),
            'overall_similarity_after': round(self.overall_similarity_after, 1),
            'improvement': round(
                self.overall_similarity_after - self.overall_similarity_before, 1
            ),
            'confirmed_notes': self.confirmed_notes,
            'corrected_notes': self.corrected_notes,
            'removed_ghost_notes': self.removed_ghost_notes,
            'flagged_missing_notes': self.flagged_missing_notes,
            'iterations_run': self.iterations_run,
        }


class TranscriptionValidator:
    """Validate and correct note transcriptions against original audio."""

    def __init__(self, max_iterations=2):
        self.max_iterations = max_iterations
        self._synth = None

    @property
    def synth(self):
        if self._synth is None:
            from src.midi_synthesizer import MidiSynthesizer
            self._synth = MidiSynthesizer()
        return self._synth

    def validate_and_correct(self, notes, audio_data, rhythm_info,
                              instrument_mode='auto'):
        """
        Validate transcribed notes against original audio and correct errors.

        Returns
        -------
        corrected_notes : list[NoteEvent]
        report : ValidationReport
        """
        report = ValidationReport()

        if not notes:
            return notes, report

        config = INSTRUMENT_CONFIGS.get(instrument_mode, INSTRUMENT_CONFIGS['auto'])
        sr = audio_data['sr']

        # Step 1: Prepare original audio
        orig_waveform = self._prepare_original_audio(audio_data, config)

        # Step 2: Extract chromagram from original
        chroma_orig, chroma_times = self._extract_chromagram(
            orig_waveform, sr, config['chroma_type']
        )

        # Step 3: Detect onsets in original
        onset_times = self._detect_original_onsets(orig_waveform, sr)

        # Step 4: Synthesize current notes and compute baseline similarity
        synth_waveform = self._synthesize_notes(notes, sr)
        chroma_synth, _ = self._extract_chromagram(
            synth_waveform, sr, config['chroma_type']
        )
        sim_before, _ = self._compute_overall_similarity(chroma_orig, chroma_synth)
        report.overall_similarity_before = sim_before * 100

        # Step 5: Iterative correction loop
        current_notes = list(notes)
        best_similarity = sim_before

        for iteration in range(self.max_iterations):
            report.iterations_run = iteration + 1
            corrections_made = 0
            notes_to_remove = []

            for i, note in enumerate(current_notes):
                score, details = self._validate_note(
                    note, chroma_orig, chroma_times, onset_times, config
                )

                if score < CHROMA_MATCH_THRESHOLD:
                    # Check ghost note first
                    if self._check_ghost_note(
                        note, chroma_orig, chroma_times, onset_times
                    ):
                        notes_to_remove.append(i)
                        report.removed_ghost_notes += 1
                        corrections_made += 1
                    else:
                        # Try pitch correction
                        new_midi = self._try_pitch_correction(
                            note, chroma_orig, chroma_times, config
                        )
                        if new_midi is not None:
                            note.pitch_midi = new_midi
                            report.corrected_notes += 1
                            corrections_made += 1

                elif score >= CORRECTION_CONFIDENCE:
                    report.confirmed_notes += 1

            # Remove ghost notes (reverse order to preserve indices)
            for idx in sorted(notes_to_remove, reverse=True):
                current_notes.pop(idx)

            if corrections_made == 0:
                break

            # Re-evaluate similarity
            synth_waveform = self._synthesize_notes(current_notes, sr)
            chroma_synth, _ = self._extract_chromagram(
                synth_waveform, sr, config['chroma_type']
            )
            sim_after, _ = self._compute_overall_similarity(chroma_orig, chroma_synth)

            if sim_after - best_similarity < MIN_IMPROVEMENT:
                break
            best_similarity = sim_after

        # Step 6: Detect missing notes (report only)
        missing = self._detect_missing_notes(
            current_notes, chroma_orig, chroma_times, onset_times
        )
        report.flagged_missing_notes = len(missing)

        # Final similarity
        report.overall_similarity_after = best_similarity * 100

        current_notes.sort(key=lambda n: (n.start_time, n.pitch_midi))
        return current_notes, report

    # ------------------------------------------------------------------
    # Audio preparation
    # ------------------------------------------------------------------

    def _prepare_original_audio(self, audio_data, config):
        """Prepare original audio. HPSS for voice to isolate harmonic content."""
        waveform = audio_data['waveform']

        if config['use_hpss']:
            try:
                stft = librosa.stft(
                    waveform, n_fft=4096, hop_length=CHROMA_HOP_LENGTH
                )
                harmonic_stft, _ = librosa.decompose.hpss(stft, margin=2.0)
                waveform = librosa.istft(
                    harmonic_stft, hop_length=CHROMA_HOP_LENGTH,
                    length=len(audio_data['waveform'])
                )
            except Exception:
                pass  # Fall back to original waveform

        return waveform

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_chromagram(self, waveform, sr, chroma_type='cqt'):
        """Extract normalized chromagram from audio."""
        if len(waveform) < CHROMA_HOP_LENGTH * 2:
            # Too short for chroma extraction
            return np.zeros((CHROMA_N_CHROMA, 1)), np.array([0.0])

        if chroma_type == 'cqt':
            chroma = librosa.feature.chroma_cqt(
                y=waveform, sr=sr,
                hop_length=CHROMA_HOP_LENGTH,
                n_chroma=CHROMA_N_CHROMA,
                bins_per_octave=36,
            )
        else:
            chroma = librosa.feature.chroma_stft(
                y=waveform, sr=sr,
                n_fft=4096,
                hop_length=CHROMA_HOP_LENGTH,
                n_chroma=CHROMA_N_CHROMA,
            )

        # Normalize per frame
        chroma_max = np.max(chroma, axis=0, keepdims=True)
        chroma_max[chroma_max == 0] = 1.0
        chroma = chroma / chroma_max

        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]), sr=sr, hop_length=CHROMA_HOP_LENGTH
        )
        return chroma, times

    def _detect_original_onsets(self, waveform, sr):
        """Detect onsets in the original audio."""
        try:
            return librosa.onset.onset_detect(
                y=waveform, sr=sr, units='time',
                hop_length=CHROMA_HOP_LENGTH, backtrack=True,
            )
        except Exception:
            return np.array([])

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    def _compute_overall_similarity(self, chroma_orig, chroma_synth):
        """Cosine similarity per frame between two chromagrams."""
        min_frames = min(chroma_orig.shape[1], chroma_synth.shape[1])
        if min_frames == 0:
            return 0.0, np.array([])

        c_orig = chroma_orig[:, :min_frames]
        c_synth = chroma_synth[:, :min_frames]

        dot = np.sum(c_orig * c_synth, axis=0)
        norm_orig = np.linalg.norm(c_orig, axis=0) + 1e-10
        norm_synth = np.linalg.norm(c_synth, axis=0) + 1e-10
        similarities = dot / (norm_orig * norm_synth)

        # Only consider frames where original has energy
        energy_mask = np.max(c_orig, axis=0) > 0.05
        if np.any(energy_mask):
            mean_sim = float(np.mean(similarities[energy_mask]))
        else:
            mean_sim = float(np.mean(similarities))

        return mean_sim, similarities

    # ------------------------------------------------------------------
    # Per-note validation
    # ------------------------------------------------------------------

    def _validate_note(self, note, chroma_orig, chroma_times, onset_times, config):
        """Compute composite validation score for a single note."""
        pitch_class = note.pitch_midi % 12

        # Time window in chroma frames
        start_frame = int(np.searchsorted(chroma_times, note.start_time))
        end_frame = int(np.searchsorted(chroma_times, note.end_time))
        end_frame = max(end_frame, start_frame + 1)
        start_frame = min(start_frame, chroma_orig.shape[1] - 1)
        end_frame = min(end_frame, chroma_orig.shape[1])

        # 1. Chroma match
        note_chroma = chroma_orig[:, start_frame:end_frame]
        if note_chroma.shape[1] > 0:
            avg_chroma = np.mean(note_chroma, axis=1)
            max_energy = np.max(avg_chroma) + 1e-10
            chroma_score = float(avg_chroma[pitch_class] / max_energy)
        else:
            chroma_score = 0.0

        # 2. Onset alignment
        if len(onset_times) > 0:
            nearest_dist = float(np.min(np.abs(onset_times - note.start_time)))
            onset_score = max(0.0, 1.0 - nearest_dist / ONSET_TOLERANCE_SEC)
        else:
            onset_score = 0.5

        # 3. Detector confidence
        confidence_score = note.confidence

        # Weighted composite
        composite = (
            chroma_score * config['chroma_weight'] +
            onset_score * config['onset_weight'] +
            confidence_score * config['confidence_weight']
        )

        details = {
            'chroma_score': round(chroma_score, 4),
            'onset_score': round(onset_score, 4),
            'confidence_score': round(confidence_score, 4),
            'composite': round(composite, 4),
        }

        return composite, details

    # ------------------------------------------------------------------
    # Correction logic
    # ------------------------------------------------------------------

    def _try_pitch_correction(self, note, chroma_orig, chroma_times, config):
        """Try shifting pitch ±1, ±2, ±12 semitones to find better chroma match."""
        pitch_class = note.pitch_midi % 12

        start_frame = int(np.searchsorted(chroma_times, note.start_time))
        end_frame = int(np.searchsorted(chroma_times, note.end_time))
        end_frame = max(end_frame, start_frame + 1)
        start_frame = min(start_frame, chroma_orig.shape[1] - 1)
        end_frame = min(end_frame, chroma_orig.shape[1])

        note_chroma = chroma_orig[:, start_frame:end_frame]
        if note_chroma.shape[1] == 0:
            return None

        avg_chroma = np.mean(note_chroma, axis=1)
        current_energy = avg_chroma[pitch_class]

        # Try small pitch shifts first, then octave
        best_shift = None
        best_energy = current_energy

        for shift in [1, -1, 2, -2]:
            new_pc = (pitch_class + shift) % 12
            new_energy = avg_chroma[new_pc]
            if new_energy > best_energy:
                best_shift = shift
                best_energy = new_energy

        # Octave corrections (pitch class stays the same, check range validity)
        for shift in [12, -12]:
            new_midi = note.pitch_midi + shift
            if 0 <= new_midi <= 127:
                # For octave shifts, prefer if original confidence was low
                if note.confidence < 0.5 and best_shift is None:
                    best_shift = shift
                    best_energy = current_energy + PITCH_CLASS_THRESHOLD

        if best_shift is None:
            return None

        improvement = best_energy - current_energy
        aggression = config['correction_aggression']

        if improvement > PITCH_CLASS_THRESHOLD * aggression:
            new_midi = note.pitch_midi + best_shift
            if 0 <= new_midi <= 127:
                return new_midi

        return None

    def _check_ghost_note(self, note, chroma_orig, chroma_times, onset_times):
        """Check if a note is spurious (no corresponding energy in original)."""
        pitch_class = note.pitch_midi % 12

        start_frame = int(np.searchsorted(chroma_times, note.start_time))
        end_frame = int(np.searchsorted(chroma_times, note.end_time))
        end_frame = max(end_frame, start_frame + 1)
        start_frame = min(start_frame, chroma_orig.shape[1] - 1)
        end_frame = min(end_frame, chroma_orig.shape[1])

        note_chroma = chroma_orig[:, start_frame:end_frame]
        if note_chroma.shape[1] == 0:
            return True

        avg_chroma = np.mean(note_chroma, axis=1)
        total_energy = float(np.sum(avg_chroma))
        pc_energy = float(avg_chroma[pitch_class])

        # Silence region
        if total_energy < GHOST_NOTE_SILENCE:
            return True

        # Pitch class has negligible energy relative to total
        if pc_energy / (total_energy + 1e-10) < GHOST_NOTE_ENERGY_RATIO:
            # Also check: no onset nearby
            if len(onset_times) > 0:
                nearest = float(np.min(np.abs(onset_times - note.start_time)))
                if nearest > ONSET_TOLERANCE_SEC * 2:
                    return True
            else:
                return True

        return False

    def _detect_missing_notes(self, notes, chroma_orig, chroma_times, onset_times):
        """Detect onsets in original with no transcribed note (report only)."""
        missing = []
        if len(onset_times) == 0:
            return missing

        note_starts = np.array([n.start_time for n in notes]) if notes else np.array([])

        for onset_t in onset_times:
            if len(note_starts) > 0:
                nearest_dist = float(np.min(np.abs(note_starts - onset_t)))
            else:
                nearest_dist = float('inf')

            if nearest_dist > ONSET_TOLERANCE_SEC * 1.5:
                frame_idx = int(np.searchsorted(chroma_times, onset_t))
                frame_idx = min(frame_idx, chroma_orig.shape[1] - 1)
                window_end = min(frame_idx + 3, chroma_orig.shape[1])
                window_chroma = chroma_orig[:, frame_idx:window_end]

                if window_chroma.shape[1] > 0:
                    avg = np.mean(window_chroma, axis=1)
                    max_energy = float(np.max(avg))
                    if max_energy > 0.3:
                        missing.append({
                            'time': float(onset_t),
                            'suspected_pitch_class': int(np.argmax(avg)),
                            'energy': max_energy,
                        })

        return missing

    # ------------------------------------------------------------------
    # Synthesis helper
    # ------------------------------------------------------------------

    def _synthesize_notes(self, notes, sr):
        """Convert NoteEvent list to audio via MidiSynthesizer."""
        notes_data = [
            {
                'pitch_midi': n.pitch_midi,
                'start_time': n.start_time,
                'end_time': n.end_time,
                'velocity': n.velocity,
            }
            for n in notes
        ]
        if not notes_data:
            return np.zeros(int(sr * 0.1), dtype=np.float32)

        try:
            return self.synth.synthesize_from_notes(notes_data, sr=sr)
        except Exception:
            return np.zeros(int(sr * 0.1), dtype=np.float32)
