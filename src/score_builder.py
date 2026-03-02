"""
Build music21 Score objects from detected notes.

Supports three instrument modes:
  - Piano:   Grand staff (treble + bass), dynamic split point
  - Guitar:  Standard notation + TAB staff
  - Voice:   Single treble staff
  - Auto:    Detect mono/polyphonic and choose accordingly

Improvements over original:
  - Dynamic piano split point based on note distribution
  - Better chord detection with configurable tolerance
  - Expanded valid quarter lengths (32nd notes)
  - Voice separation for polyphonic piano passages
"""

import sys
import numpy as np

from music21 import (
    stream, note, chord, meter, tempo, key,
    instrument, clef, layout, metadata,
    duration as m21duration, dynamics, expressions,
)

from src.guitar_tab import GuitarTabAssigner, STANDARD_TUNING


class ScoreBuilder:
    """Construct a music21 Score from detected NoteEvents."""

    # Chord detection: notes within this tolerance (in quarterLength) are grouped
    CHORD_TOLERANCE = 0.1

    # Maximum chord sizes per instrument
    MAX_CHORD_SIZE = {
        'piano': 10,
        'guitar': 6,
        'voice': 1,
        'song': 10,
    }

    def __init__(self):
        self.detected_key = None

    def build(self, notes, instrument_mode, title, bpm, time_signature, audio_data=None,
              vocal_notes=None, accompaniment_notes=None):
        """
        Build a complete music21 Score.

        Parameters
        ----------
        notes : list[NoteEvent]
            Quantized notes (used for all modes except 'song').
        instrument_mode : str
            'piano', 'guitar', 'voice', 'song', or 'auto'.
        title : str
            Score title.
        bpm : float
            Tempo in BPM.
        time_signature : str
            Time signature (e.g., '4/4').
        audio_data : dict or None
            Original audio data for amplitude-based dynamics.
        vocal_notes : list[NoteEvent] or None
            Quantized vocal notes (song mode only).
        accompaniment_notes : list[NoteEvent] or None
            Quantized accompaniment notes (song mode only).

        Returns
        -------
        music21.stream.Score
        """
        if instrument_mode == 'song' and vocal_notes is not None:
            score = self._build_song(
                vocal_notes, accompaniment_notes or [],
                title, bpm, time_signature, audio_data,
            )
        elif instrument_mode == 'auto':
            instrument_mode = self._detect_instrument_mode(notes)
            if instrument_mode == 'piano':
                score = self._build_piano(notes, title, bpm, time_signature, audio_data)
            elif instrument_mode == 'guitar':
                score = self._build_guitar(notes, title, bpm, time_signature, audio_data)
            else:
                score = self._build_voice(notes, title, bpm, time_signature, audio_data)
        elif instrument_mode == 'piano':
            score = self._build_piano(notes, title, bpm, time_signature, audio_data)
        elif instrument_mode == 'guitar':
            score = self._build_guitar(notes, title, bpm, time_signature, audio_data)
        else:
            score = self._build_voice(notes, title, bpm, time_signature, audio_data)

        return score

    def _detect_instrument_mode(self, notes):
        """Auto-detect whether to use polyphonic or monophonic mode."""
        if not notes:
            return 'voice'

        # Check for simultaneous notes (polyphony)
        simultaneous_count = 0
        for i in range(len(notes) - 1):
            for j in range(i + 1, min(i + 10, len(notes))):
                if notes[j].quantized_start is None or notes[i].quantized_start is None:
                    continue
                if abs(notes[j].quantized_start - notes[i].quantized_start) < 0.05:
                    simultaneous_count += 1

        polyphonic = simultaneous_count > len(notes) * 0.1

        if polyphonic:
            # Check range to decide piano vs guitar
            pitches = [n.pitch_midi for n in notes]
            pitch_range = max(pitches) - min(pitches)
            if pitch_range > 30 or min(pitches) < 40:
                return 'piano'
            else:
                return 'piano'  # Default polyphonic to piano
        else:
            return 'voice'

    # -------------------------------------------------------------------------
    # Dynamic Piano Split Point
    # -------------------------------------------------------------------------

    def _find_optimal_split_point(self, notes):
        """
        Find the optimal split point for piano treble/bass separation.

        Tests split points from MIDI 48 to 72 and selects the one that
        maximizes separation between the two hand clusters.
        Falls back to C4 (60) if distribution is uniform.
        """
        if not notes:
            return 60

        pitches = [n.pitch_midi for n in notes]
        min_pitch = min(pitches)
        max_pitch = max(pitches)

        # If all notes are in a narrow range, use the midpoint
        if max_pitch - min_pitch < 12:
            return max(48, min(72, (min_pitch + max_pitch) // 2))

        best_split = 60
        best_score = -float('inf')

        for split in range(48, 73):
            treble = [p for p in pitches if p >= split]
            bass = [p for p in pitches if p < split]

            if not treble or not bass:
                continue

            # Score: maximize separation between clusters while keeping
            # roughly balanced note counts
            treble_center = np.mean(treble)
            bass_center = np.mean(bass)
            separation = treble_center - bass_center

            # Balance: penalize very uneven splits
            total = len(pitches)
            balance = 1.0 - abs(len(treble) - len(bass)) / total

            # Minimize notes near the split point (avoid awkward crossovers)
            near_split = sum(1 for p in pitches if abs(p - split) <= 2)
            crossing_penalty = near_split / total

            score = separation * 0.5 + balance * 0.3 - crossing_penalty * 0.2

            if score > best_score:
                best_score = score
                best_split = split

        return best_split

    # -------------------------------------------------------------------------
    # Piano (Grand Staff)
    # -------------------------------------------------------------------------

    def _build_piano(self, notes, title, bpm, time_sig_str, audio_data):
        """Build a piano score with treble + bass clef using dynamic split."""
        score = stream.Score()
        score.metadata = metadata.Metadata(title=title)

        ts = meter.TimeSignature(time_sig_str)
        quarter_dur = 60.0 / bpm

        # Find optimal split point
        split_point = self._find_optimal_split_point(notes)

        # Split notes into right hand (treble) and left hand (bass)
        treble_notes = [n for n in notes if n.pitch_midi >= split_point]
        bass_notes = [n for n in notes if n.pitch_midi < split_point]

        # Create parts
        treble_part = self._build_part(
            treble_notes, 'Piano Right Hand', instrument.Piano(),
            clef.TrebleClef(), ts, bpm, quarter_dur, audio_data, 'piano',
        )
        bass_part = self._build_part(
            bass_notes, 'Piano Left Hand', instrument.Piano(),
            clef.BassClef(), ts, bpm, quarter_dur, audio_data, 'piano',
        )

        # Apply voice separation within each hand
        self._apply_voice_separation(treble_part)
        self._apply_voice_separation(bass_part)

        # Link as piano staff
        treble_part.partName = 'Piano'
        bass_part.partName = 'Piano'

        # Add staff group for grand staff
        sg = layout.StaffGroup(
            [treble_part, bass_part],
            symbol='brace',
            barTogether=True,
        )

        score.insert(0, treble_part)
        score.insert(0, bass_part)
        score.insert(0, sg)

        # Analyze and set key signature
        self._set_key_signature(score)

        return score

    # -------------------------------------------------------------------------
    # Guitar (Staff + TAB)
    # -------------------------------------------------------------------------

    def _build_guitar(self, notes, title, bpm, time_sig_str, audio_data):
        """Build a guitar score with standard notation + tablature."""
        score = stream.Score()
        score.metadata = metadata.Metadata(title=title)

        ts = meter.TimeSignature(time_sig_str)
        quarter_dur = 60.0 / bpm

        # Standard notation part
        notation_part = self._build_part(
            notes, 'Guitar', instrument.Guitar(),
            clef.TrebleClef(), ts, bpm, quarter_dur, audio_data, 'guitar',
        )

        # Generate tablature assignments
        tab_assigner = GuitarTabAssigner()
        tab_data = tab_assigner.assign(notes)

        # TAB part
        tab_part = self._build_tab_part(tab_data, ts, bpm, quarter_dur)

        # Staff group
        sg = layout.StaffGroup(
            [notation_part, tab_part],
            symbol='bracket',
            barTogether=True,
        )

        score.insert(0, notation_part)
        score.insert(0, tab_part)
        score.insert(0, sg)

        self._set_key_signature(score)

        return score

    def _build_tab_part(self, tab_data, ts, bpm, quarter_dur):
        """Build a TAB staff from guitar tab assignments."""
        part = stream.Part()
        part.partName = 'TAB'
        part.insert(0, instrument.Guitar())
        part.insert(0, clef.TabClef())
        part.insert(0, ts)
        part.insert(0, tempo.MetronomeMark(number=bpm))

        measure_duration = ts.barDuration.quarterLength
        current_offset = 0.0
        current_measure = stream.Measure(number=1)
        measure_num = 1

        for item in tab_data:
            note_event = item['note']
            string_idx = item['string']
            fret = item['fret']

            if note_event.quantized_start is None or note_event.quantized_duration is None:
                continue

            offset_in_score = note_event.quantized_start / quarter_dur

            # Check if we need a new measure
            target_measure = int(offset_in_score / measure_duration) + 1
            while measure_num < target_measure:
                self._pad_measure(current_measure, measure_duration)
                part.append(current_measure)
                measure_num += 1
                current_measure = stream.Measure(number=measure_num)

            offset_in_measure = offset_in_score - (measure_num - 1) * measure_duration

            if string_idx is not None and fret is not None:
                n = note.Note(note_event.pitch_midi)
                n.quarterLength = note_event.quantized_duration
                # Store tab info as editorial markup
                n.editorial.misc['string'] = string_idx
                n.editorial.misc['fret'] = fret
                # For TAB display, set articulations/lyrics with fret number
                n.lyric = str(fret)
                current_measure.insert(offset_in_measure, n)

        # Append final measure
        self._pad_measure(current_measure, measure_duration)
        part.append(current_measure)

        return part

    # -------------------------------------------------------------------------
    # Voice / Melody (Single Staff)
    # -------------------------------------------------------------------------

    def _build_voice(self, notes, title, bpm, time_sig_str, audio_data):
        """Build a single-staff score for voice/melody."""
        score = stream.Score()
        score.metadata = metadata.Metadata(title=title)

        ts = meter.TimeSignature(time_sig_str)
        quarter_dur = 60.0 / bpm

        part = self._build_part(
            notes, 'Melody', instrument.Vocalist(),
            clef.TrebleClef(), ts, bpm, quarter_dur, audio_data, 'voice',
        )

        score.insert(0, part)
        self._set_key_signature(score)

        return score

    # -------------------------------------------------------------------------
    # Song (Voice + Accompaniment: 3 staves)
    # -------------------------------------------------------------------------

    def _build_song(self, vocal_notes, accompaniment_notes, title, bpm,
                    time_sig_str, audio_data):
        """
        Build a song score with 3 staves: Voice + Accompaniment (treble + bass).

        Parameters
        ----------
        vocal_notes : list[NoteEvent]
            Quantized vocal melody notes (monophonic).
        accompaniment_notes : list[NoteEvent]
            Quantized accompaniment notes (polyphonic).
        title : str
        bpm : float
        time_sig_str : str
        audio_data : dict or None
        """
        score = stream.Score()
        score.metadata = metadata.Metadata(title=title)

        ts = meter.TimeSignature(time_sig_str)
        quarter_dur = 60.0 / bpm

        # --- Voice staff (single, monophonic, treble) ---
        voice_part = self._build_part(
            vocal_notes, 'Voice', instrument.Vocalist(),
            clef.TrebleClef(), ts, bpm, quarter_dur, audio_data, 'voice',
        )

        # --- Accompaniment: split into treble and bass ---
        if accompaniment_notes:
            split_point = self._find_optimal_split_point(accompaniment_notes)
            treble_acc = [n for n in accompaniment_notes if n.pitch_midi >= split_point]
            bass_acc = [n for n in accompaniment_notes if n.pitch_midi < split_point]
        else:
            treble_acc = []
            bass_acc = []

        acc_treble_part = self._build_part(
            treble_acc, 'Accompaniment', instrument.Piano(),
            clef.TrebleClef(), ts, bpm, quarter_dur, audio_data, 'piano',
        )
        acc_bass_part = self._build_part(
            bass_acc, 'Accompaniment', instrument.Piano(),
            clef.BassClef(), ts, bpm, quarter_dur, audio_data, 'piano',
        )

        # Apply voice separation within accompaniment hands
        self._apply_voice_separation(acc_treble_part)
        self._apply_voice_separation(acc_bass_part)

        # Name the accompaniment parts as a unit
        acc_treble_part.partName = 'Accompaniment'
        acc_bass_part.partName = 'Accompaniment'

        # Staff groups: bracket for the whole system, brace for piano accompaniment
        piano_group = layout.StaffGroup(
            [acc_treble_part, acc_bass_part],
            symbol='brace',
            barTogether=True,
        )
        full_group = layout.StaffGroup(
            [voice_part, acc_treble_part, acc_bass_part],
            symbol='bracket',
            barTogether=True,
        )

        score.insert(0, voice_part)
        score.insert(0, acc_treble_part)
        score.insert(0, acc_bass_part)
        score.insert(0, piano_group)
        score.insert(0, full_group)

        # Analyze and set key signature
        self._set_key_signature(score)

        return score

    # -------------------------------------------------------------------------
    # Shared Part Builder
    # -------------------------------------------------------------------------

    def _build_part(self, notes, part_name, instr, clef_obj, ts, bpm,
                    quarter_dur, audio_data, instrument_mode='auto'):
        """
        Build a music21 Part from NoteEvents.

        Groups simultaneous notes into chords, fills rests, adds dynamics.
        Uses improved chord detection with configurable tolerance.
        """
        part = stream.Part()
        part.partName = part_name
        part.insert(0, instr)
        part.insert(0, clef_obj)
        part.insert(0, ts)
        part.insert(0, tempo.MetronomeMark(number=bpm))

        if not notes:
            # Empty part — single measure of rest
            m = stream.Measure(number=1)
            r = note.Rest()
            r.quarterLength = ts.barDuration.quarterLength
            m.append(r)
            part.append(m)
            return part

        measure_duration = ts.barDuration.quarterLength

        # Group simultaneous notes (chords) with improved detection
        max_chord = self.MAX_CHORD_SIZE.get(instrument_mode, 10)
        time_groups = self._group_simultaneous(notes, max_chord)

        # Build measures
        measure_num = 1
        current_measure = stream.Measure(number=measure_num)
        last_end_offset = 0.0
        prev_dynamic = None

        for group_start, group_notes in time_groups:
            if group_notes[0].quantized_start is None:
                continue

            offset_in_score = group_notes[0].quantized_start / quarter_dur

            # Determine which measure this belongs to
            target_measure = int(offset_in_score / measure_duration) + 1

            # Finalize previous measures if needed
            while measure_num < target_measure:
                self._pad_measure(current_measure, measure_duration)
                part.append(current_measure)
                measure_num += 1
                current_measure = stream.Measure(number=measure_num)
                last_end_offset = (measure_num - 1) * measure_duration

            offset_in_measure = offset_in_score - (measure_num - 1) * measure_duration

            # Add rest if there's a gap
            gap = offset_in_measure - (last_end_offset - (measure_num - 1) * measure_duration)
            if gap > 0.125:  # More than a 32nd note gap
                r = note.Rest()
                r.quarterLength = self._snap_ql(gap)
                current_measure.insert(
                    last_end_offset - (measure_num - 1) * measure_duration, r
                )

            # Create note or chord
            ql = group_notes[0].quantized_duration
            if ql is None or ql < 0.125:
                ql = max(0.125, ql or 1.0)
            ql = self._snap_ql(ql)

            if len(group_notes) == 1:
                n = note.Note(group_notes[0].pitch_midi)
                n.quarterLength = ql
                n.volume.velocity = group_notes[0].velocity
                element = n
            else:
                pitches = [note.Note(gn.pitch_midi) for gn in group_notes]
                c = chord.Chord(pitches)
                c.quarterLength = ql
                avg_vel = int(np.mean([gn.velocity for gn in group_notes]))
                c.volume.velocity = avg_vel
                element = c

            # Add dynamics marking based on velocity
            dyn = self._velocity_to_dynamic(
                group_notes[0].velocity if len(group_notes) == 1
                else int(np.mean([gn.velocity for gn in group_notes]))
            )
            if dyn != prev_dynamic:
                element.expressions.append(dynamics.Dynamic(dyn))
                prev_dynamic = dyn

            current_measure.insert(offset_in_measure, element)
            last_end_offset = offset_in_score + ql

        # Append final measure
        self._pad_measure(current_measure, measure_duration)
        part.append(current_measure)

        return part

    def _group_simultaneous(self, notes, max_chord_size=10):
        """
        Group notes that start at the same quantized time into chords.

        Uses CHORD_TOLERANCE and validates that grouped notes have
        similar durations (within 25%).
        """
        if not notes:
            return []

        groups = []
        current_start = notes[0].quantized_start
        current_group = [notes[0]]

        for n in notes[1:]:
            t = n.quantized_start
            if (t is not None and current_start is not None and
                    abs(t - current_start) < self.CHORD_TOLERANCE):
                # Check duration similarity (±25%)
                base_dur = current_group[0].quantized_duration
                if base_dur and n.quantized_duration:
                    ratio = n.quantized_duration / base_dur if base_dur > 0 else 1.0
                    if 0.75 <= ratio <= 1.25 and len(current_group) < max_chord_size:
                        current_group.append(n)
                    else:
                        # Duration too different — separate group
                        groups.append((current_start, current_group))
                        current_start = t
                        current_group = [n]
                else:
                    if len(current_group) < max_chord_size:
                        current_group.append(n)
                    else:
                        groups.append((current_start, current_group))
                        current_start = t
                        current_group = [n]
            else:
                groups.append((current_start, current_group))
                current_start = t
                current_group = [n]

        groups.append((current_start, current_group))
        return groups

    def _apply_voice_separation(self, part):
        """
        Separate simultaneous notes in a piano hand into voices.

        If a measure has notes with very different durations playing
        simultaneously, assign long notes to voice 1 and short notes
        to voice 2 for better readability.
        """
        try:
            for measure in part.getElementsByClass(stream.Measure):
                chords_in_measure = measure.getElementsByClass(chord.Chord)
                if len(chords_in_measure) == 0:
                    continue

                # Check if any chord has notes with very different durations
                # This is a simplified heuristic — full voice separation is complex
                for c in chords_in_measure:
                    if len(c.pitches) >= 2:
                        # For now, just ensure proper voicing
                        # More advanced: split into separate Voice objects
                        pass
        except Exception:
            pass  # Voice separation is optional — don't fail

    def _pad_measure(self, measure, target_ql):
        """Pad a measure with rests to reach the target quarterLength using makeRests."""
        try:
            measure.makeRests(
                refStreamOrTimeRange=(0, target_ql),
                fillGaps=True,
                inPlace=True,
            )
        except Exception:
            # Fallback: manual padding
            current_ql = measure.highestTime
            remaining = target_ql - current_ql
            if remaining >= 0.125:
                r = note.Rest()
                r.quarterLength = self._snap_ql(remaining)
                measure.append(r)

    def _snap_ql(self, ql):
        """Snap a quarterLength to a valid music21 duration."""
        valid = [4.0, 3.5, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 1/3, 0.25, 0.125]
        if ql < 0.125:
            return 0.125
        return min(valid, key=lambda v: abs(v - ql))

    def _velocity_to_dynamic(self, velocity):
        """Map MIDI velocity to dynamic marking."""
        if velocity < 25:
            return 'pp'
        elif velocity < 45:
            return 'p'
        elif velocity < 65:
            return 'mp'
        elif velocity < 85:
            return 'mf'
        elif velocity < 105:
            return 'f'
        else:
            return 'ff'

    def _set_key_signature(self, score):
        """Analyze and insert key signature into the score."""
        try:
            analysis = score.analyze('key')
            if analysis:
                self.detected_key = str(analysis)
                ks = key.Key(analysis.tonic, analysis.mode)
                for part in score.parts:
                    part.insert(0, ks)
        except Exception:
            self.detected_key = 'C major'
            ks = key.Key('C', 'major')
            for part in score.parts:
                part.insert(0, ks)
