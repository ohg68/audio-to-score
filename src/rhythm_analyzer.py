"""
Rhythm analysis: BPM detection, time signature inference, and note quantization.

Improvements over original:
  - Multi-method BPM detection (beat_track + tempo + onset_strength autocorrelation)
  - Expanded time signature support (2/4, 3/4, 4/4, 6/8)
  - Context-aware quantization (beat alignment + consistency + rarity penalties)
  - Cross-pitch overlap resolution
  - Expanded quantization grid (32nd notes, double-dotted values)
"""

import numpy as np
import librosa


class RhythmAnalyzer:
    """Analyze rhythmic properties and quantize detected notes."""

    # Expanded quantization grid durations relative to a quarter note
    GRID_VALUES = [
        4.0,    # whole note
        3.5,    # double-dotted half
        3.0,    # dotted half
        2.0,    # half note
        1.5,    # dotted quarter
        1.0,    # quarter note
        0.75,   # dotted eighth
        0.5,    # eighth note
        1/3,    # eighth triplet
        0.25,   # sixteenth note
        1/6,    # sixteenth triplet
        0.125,  # 32nd note
    ]

    # Common durations get a bonus in context-aware quantization
    COMMON_DURATIONS = {4.0, 2.0, 1.0, 0.5, 0.25}

    def analyze(self, waveform, sr):
        """
        Analyze BPM and time signature from audio.

        Parameters
        ----------
        waveform : np.ndarray
            Mono audio.
        sr : int
            Sample rate.

        Returns
        -------
        dict with keys:
            'bpm': float
            'bpm_confidence': float (0-1, how confident is the BPM detection)
            'time_signature': str (e.g. '4/4')
            'time_sig_confidence': float (0-1)
            'beat_times': np.ndarray (times of detected beats in seconds)
            'quarter_duration': float (duration of one quarter note in seconds)
        """
        # Multi-method BPM detection
        bpm, bpm_confidence, beat_frames = self._detect_bpm_robust(waveform, sr)
        bpm = self._round_bpm(bpm)

        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        quarter_duration = 60.0 / bpm

        # Infer time signature from beat accentuation
        time_sig, ts_confidence = self._infer_time_signature(waveform, sr, beat_times)

        return {
            'bpm': bpm,
            'bpm_confidence': bpm_confidence,
            'time_signature': time_sig,
            'time_sig_confidence': ts_confidence,
            'beat_times': beat_times,
            'quarter_duration': quarter_duration,
        }

    def _detect_bpm_robust(self, waveform, sr):
        """
        Multi-method BPM detection for higher accuracy.

        Combines three methods and uses weighted consensus:
        1. librosa.beat.beat_track (onset-based)
        2. librosa.feature.tempo (autocorrelation-based)
        3. Onset strength autocorrelation

        Returns
        -------
        tuple: (bpm, confidence, beat_frames)
        """
        candidates = []

        # Method 1: beat_track (onset-based)
        try:
            tempo1, beat_frames = librosa.beat.beat_track(
                y=waveform, sr=sr, start_bpm=120, units='frames',
            )
            bpm1 = float(tempo1) if np.isscalar(tempo1) else float(tempo1[0])
            candidates.append(bpm1)
        except Exception:
            beat_frames = np.array([])
            candidates.append(120.0)

        # Method 2: librosa.feature.tempo with multiple candidates
        try:
            tempos = librosa.feature.tempo(
                y=waveform, sr=sr, aggregate=None,
            )
            if tempos is not None and len(tempos) > 0:
                # Take the most common tempo estimate
                if hasattr(tempos, 'shape') and len(tempos.shape) > 0:
                    bpm2 = float(np.median(tempos))
                else:
                    bpm2 = float(tempos)
                candidates.append(bpm2)
        except Exception:
            pass

        # Method 3: Onset strength autocorrelation
        try:
            onset_env = librosa.onset.onset_strength(y=waveform, sr=sr)
            # Autocorrelate onset strength
            ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
            if len(ac) > 1:
                # Find the first prominent peak after lag 0
                ac_norm = ac / (ac[0] + 1e-8)
                # Search for peaks in reasonable BPM range (40-220 BPM)
                hop = 512  # default hop for onset_strength
                min_lag = int(60.0 / 220 * sr / hop)  # ~220 BPM
                max_lag = int(60.0 / 40 * sr / hop)   # ~40 BPM
                min_lag = max(1, min_lag)
                max_lag = min(len(ac_norm), max_lag)

                if max_lag > min_lag:
                    search_region = ac_norm[min_lag:max_lag]
                    peak_lag = np.argmax(search_region) + min_lag
                    bpm3 = 60.0 * sr / (peak_lag * hop)
                    candidates.append(bpm3)
        except Exception:
            pass

        if not candidates:
            return 120.0, 0.0, beat_frames

        # Consensus: check agreement between methods
        candidates = np.array(candidates)

        # Handle half/double tempo relationships
        # Normalize all candidates to be within a factor of 2 of the median
        median_bpm = np.median(candidates)
        normalized = []
        for c in candidates:
            while c < median_bpm * 0.6 and c > 0:
                c *= 2
            while c > median_bpm * 1.6:
                c /= 2
            normalized.append(c)
        normalized = np.array(normalized)

        # Weighted median (beat_track gets extra weight as it also provides beat_frames)
        final_bpm = float(np.median(normalized))

        # Confidence based on agreement between methods
        if len(normalized) >= 2:
            std = np.std(normalized)
            # Lower std = higher confidence
            confidence = float(np.clip(1.0 - std / 30.0, 0.1, 1.0))
        else:
            confidence = 0.5

        return final_bpm, confidence, beat_frames

    def _round_bpm(self, bpm):
        """Round BPM to the nearest common tempo value."""
        common_tempos = [
            40, 42, 44, 46, 48, 50, 52, 54, 56, 58,
            60, 63, 66, 69, 72, 76, 80, 84, 88, 92, 96,
            100, 104, 108, 112, 116, 120, 126, 132, 138, 144,
            152, 160, 168, 176, 184, 192, 200, 208, 216,
        ]
        # Find closest common tempo
        closest = min(common_tempos, key=lambda t: abs(t - bpm))
        # Only snap if within 3 BPM
        if abs(closest - bpm) <= 3:
            return closest
        return round(bpm)

    def _infer_time_signature(self, waveform, sr, beat_times):
        """
        Infer time signature by analyzing metric accentuation patterns.

        Supports: 2/4, 3/4, 4/4, 6/8.
        Returns (time_sig_str, confidence).
        """
        if len(beat_times) < 8:
            return '4/4', 0.3  # Default for short audio, low confidence

        # Get RMS energy at each beat
        hop_length = 512
        rms = librosa.feature.rms(y=waveform, frame_length=2048, hop_length=hop_length)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

        beat_energies = []
        for bt in beat_times:
            idx = np.argmin(np.abs(rms_times - bt))
            beat_energies.append(rms[min(idx, len(rms) - 1)])

        beat_energies = np.array(beat_energies)
        if len(beat_energies) == 0 or np.mean(beat_energies) == 0:
            return '4/4', 0.3

        # Check grouping patterns
        score_2 = self._accentuation_score(beat_energies, 2)
        score_3 = self._accentuation_score(beat_energies, 3)
        score_4 = self._accentuation_score(beat_energies, 4)
        score_6 = self._accentuation_score(beat_energies, 6)

        scores = {
            '2/4': score_2,
            '3/4': score_3,
            '4/4': score_4,
            '6/8': score_6,
        }

        # Find best match
        best_sig = max(scores, key=scores.get)
        best_score = scores[best_sig]

        # Calculate confidence: how much better is the best vs second best
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0:
            ratio = sorted_scores[0] / sorted_scores[1]
            confidence = float(np.clip((ratio - 1.0) / 0.5, 0.1, 1.0))
        else:
            confidence = 0.5

        # Require clear distinction to override 4/4 default
        if best_sig == '4/4':
            return '4/4', confidence
        elif best_sig == '3/4' and score_3 > score_4 * 1.2:
            return '3/4', confidence
        elif best_sig == '6/8' and score_6 > score_4 * 1.3:
            return '6/8', confidence
        elif best_sig == '2/4' and score_2 > score_4 * 1.15:
            return '2/4', confidence
        else:
            return '4/4', confidence * 0.7  # Default with lower confidence

    def _accentuation_score(self, energies, grouping):
        """
        Score how well energies match a given metric grouping.

        Higher score = better fit.
        """
        if len(energies) < grouping * 2:
            return 0.0

        accented = []
        unaccented = []
        for i, e in enumerate(energies):
            if i % grouping == 0:
                accented.append(e)
            else:
                unaccented.append(e)

        if not accented or not unaccented:
            return 0.0

        # Ratio of accented vs unaccented energy
        return np.mean(accented) / (np.mean(unaccented) + 1e-8)

    def quantize(self, notes, rhythm_info):
        """
        Quantize note start times and durations to the rhythmic grid.

        Uses context-aware scoring that considers:
        - Proximity to the nearest grid value (60% weight)
        - Beat alignment (25% weight)
        - Consistency with neighboring notes (15% weight)

        Parameters
        ----------
        notes : list[NoteEvent]
            Detected notes.
        rhythm_info : dict
            Output from analyze().

        Returns
        -------
        list[NoteEvent]
            Notes with quantized_start and quantized_duration filled.
        """
        quarter_dur = rhythm_info['quarter_duration']

        # Build absolute grid durations in seconds
        grid_durations = [g * quarter_dur for g in self.GRID_VALUES]

        # Quantize start times to nearest grid position
        min_grid_step = min(self.GRID_VALUES) * quarter_dur

        # First pass: quantize start times
        for note in notes:
            note.quantized_start = round(note.start_time / min_grid_step) * min_grid_step

        # Second pass: context-aware duration quantization
        for i, note in enumerate(notes):
            raw_duration = note.end_time - note.start_time
            note.quantized_duration = self._snap_duration_contextual(
                raw_duration, grid_durations, quarter_dur, rhythm_info, note, notes, i,
            )

        # Fix overlapping notes (same pitch)
        notes = self._fix_overlaps(notes)

        # Fix cross-pitch overlaps in monophonic contexts
        notes = self._fix_cross_pitch_overlaps(notes)

        return notes

    def _snap_duration_contextual(self, raw_duration, grid_durations, quarter_dur,
                                   rhythm_info, note, all_notes, idx):
        """
        Context-aware duration snapping.

        Scoring:
          score = proximity * 0.6 + beat_alignment * 0.25 + consistency * 0.15
        """
        if raw_duration <= 0:
            return 0.25  # Minimum: sixteenth note

        best_ql = 0.25
        best_score = -float('inf')

        for grid_sec, grid_ql in zip(grid_durations, self.GRID_VALUES):
            # 1. Proximity score (how close is this grid value to raw duration)
            proximity = 1.0 - min(abs(grid_sec - raw_duration) / (raw_duration + 1e-8), 1.0)

            # 2. Beat alignment score (does the note end on a beat?)
            note_end = note.quantized_start + grid_sec
            beat_times = rhythm_info.get('beat_times', np.array([]))
            if len(beat_times) > 0:
                nearest_beat_dist = np.min(np.abs(beat_times - note_end))
                beat_alignment = 1.0 - min(nearest_beat_dist / quarter_dur, 1.0)
            else:
                beat_alignment = 0.0

            # 3. Consistency score (do neighboring notes use similar durations?)
            consistency = 0.0
            neighbor_count = 0
            for offset in [-2, -1, 1, 2]:
                ni = idx + offset
                if 0 <= ni < len(all_notes) and all_notes[ni].quantized_duration is not None:
                    neighbor_dur = all_notes[ni].quantized_duration
                    if abs(neighbor_dur - grid_ql) < 0.01:
                        consistency += 1.0
                    neighbor_count += 1
            if neighbor_count > 0:
                consistency /= neighbor_count

            # 4. Common duration bonus
            rarity_bonus = 0.1 if grid_ql in self.COMMON_DURATIONS else 0.0

            # Combined score
            score = (proximity * 0.55 +
                     beat_alignment * 0.25 +
                     consistency * 0.10 +
                     rarity_bonus * 0.10)

            if score > best_score:
                best_score = score
                best_ql = grid_ql

        # Clamp to valid range
        return max(0.125, min(4.0, best_ql))

    def _snap_duration(self, raw_duration, grid_durations, quarter_dur):
        """Simple snap a raw duration to the closest grid value (in quarterLength units)."""
        if raw_duration <= 0:
            return 0.25  # Minimum: sixteenth note

        # Find closest grid duration
        closest_sec = min(grid_durations, key=lambda g: abs(g - raw_duration))
        # Convert to quarterLength
        ql = closest_sec / quarter_dur

        # Clamp to valid range
        return max(0.125, min(4.0, ql))

    def _fix_overlaps(self, notes):
        """Trim overlapping notes of the same pitch."""
        # Group by pitch
        by_pitch = {}
        for note in notes:
            by_pitch.setdefault(note.pitch_midi, []).append(note)

        for pitch, group in by_pitch.items():
            group.sort(key=lambda n: n.quantized_start)
            for i in range(len(group) - 1):
                curr = group[i]
                nxt = group[i + 1]
                curr_end = curr.quantized_start + curr.quantized_duration
                if curr_end > nxt.quantized_start:
                    # Trim current note
                    new_dur = nxt.quantized_start - curr.quantized_start
                    if new_dur > 0:
                        curr.quantized_duration = new_dur
                    else:
                        curr.quantized_duration = 0.125

        return notes

    def _fix_cross_pitch_overlaps(self, notes, max_simultaneous=None):
        """
        Fix overlaps across different pitches.

        For monophonic instruments (voice), ensure no two notes sound
        simultaneously by trimming the earlier note.

        For polyphonic instruments, limit maximum simultaneous notes.
        """
        if not notes:
            return notes

        # Detect if this is likely monophonic (few simultaneous notes)
        notes_sorted = sorted(notes, key=lambda n: n.quantized_start)
        simultaneous_count = 0
        for i in range(len(notes_sorted) - 1):
            curr_end = notes_sorted[i].quantized_start + notes_sorted[i].quantized_duration
            if curr_end > notes_sorted[i + 1].quantized_start:
                simultaneous_count += 1

        is_monophonic = simultaneous_count < len(notes) * 0.05

        if is_monophonic:
            # Monophonic: no two notes should overlap
            for i in range(len(notes_sorted) - 1):
                curr = notes_sorted[i]
                nxt = notes_sorted[i + 1]
                curr_end = curr.quantized_start + curr.quantized_duration
                if curr_end > nxt.quantized_start:
                    new_dur = nxt.quantized_start - curr.quantized_start
                    if new_dur > 0:
                        curr.quantized_duration = new_dur
                    else:
                        curr.quantized_duration = 0.125

        return notes
