"""
Guitar tablature generation: assign MIDI notes to guitar strings/frets.
"""


# Standard guitar tuning (low to high): E2 A2 D3 G3 B3 E4
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # MIDI note numbers
STRING_NAMES = ['E2', 'A2', 'D3', 'G3', 'B3', 'E4']
MAX_FRET = 24
# Maximum comfortable fret span in a single hand position
MAX_SPAN = 5


class GuitarTabAssigner:
    """
    Assign MIDI notes to guitar strings and frets.

    Uses an optimization approach that minimizes hand position changes
    and favors lower positions when possible.
    """

    def __init__(self, tuning=None):
        self.tuning = tuning or STANDARD_TUNING

    def assign(self, notes):
        """
        Assign string and fret to each note.

        Parameters
        ----------
        notes : list[NoteEvent]
            Notes to assign. Each NoteEvent must have pitch_midi.

        Returns
        -------
        list[dict]
            Each dict has: 'note' (NoteEvent), 'string' (0-5), 'fret' (0-24).
            String 0 = lowest (E2), string 5 = highest (E4).
            Returns None for notes outside guitar range.
        """
        assignments = []
        prev_position = 0  # Track hand position (fret center)

        # Process notes in time order, group simultaneous notes (chords)
        time_groups = self._group_by_time(notes)

        for group in time_groups:
            if len(group) == 1:
                # Single note — find best string/fret
                result = self._assign_single(group[0], prev_position)
                if result:
                    assignments.append(result)
                    prev_position = result['fret']
            else:
                # Chord — assign all notes trying to minimize span
                chord_results = self._assign_chord(group, prev_position)
                assignments.extend(chord_results)
                frets = [r['fret'] for r in chord_results if r['fret'] is not None]
                if frets:
                    prev_position = int(np.mean(frets)) if frets else prev_position

        return assignments

    def _group_by_time(self, notes):
        """Group notes by quantized start time (simultaneous = chord)."""
        if not notes:
            return []

        groups = []
        current_group = [notes[0]]
        current_time = notes[0].quantized_start

        for note in notes[1:]:
            t = note.quantized_start
            if t is not None and current_time is not None and abs(t - current_time) < 0.01:
                current_group.append(note)
            else:
                groups.append(current_group)
                current_group = [note]
                current_time = t

        groups.append(current_group)
        return groups

    def _assign_single(self, note, prev_position):
        """Find the best string/fret for a single note."""
        midi = note.pitch_midi
        candidates = self._get_candidates(midi)

        if not candidates:
            return {'note': note, 'string': None, 'fret': None}

        # Score candidates: prefer lower frets and closer to previous position
        best = min(candidates, key=lambda c: self._position_cost(c, prev_position))

        return {'note': note, 'string': best[0], 'fret': best[1]}

    def _assign_chord(self, notes, prev_position):
        """
        Assign strings/frets for a chord (multiple simultaneous notes).

        Each note must be on a different string. Minimize total fret span.
        """
        # Sort notes by pitch (low to high)
        sorted_notes = sorted(notes, key=lambda n: n.pitch_midi)

        # Get candidates for each note
        all_candidates = []
        for note in sorted_notes:
            candidates = self._get_candidates(note.pitch_midi)
            all_candidates.append((note, candidates))

        # Greedy assignment: assign from lowest pitch up, avoid string conflicts
        used_strings = set()
        results = []

        for note, candidates in all_candidates:
            # Filter out already-used strings
            available = [(s, f) for s, f in candidates if s not in used_strings]

            if not available:
                results.append({'note': note, 'string': None, 'fret': None})
                continue

            # Pick best: prefer close to prev_position and lower frets
            best = min(available, key=lambda c: self._position_cost(c, prev_position))
            used_strings.add(best[0])
            results.append({'note': note, 'string': best[0], 'fret': best[1]})

        return results

    def _get_candidates(self, midi_note):
        """
        Get all valid (string, fret) pairs for a MIDI note.

        Returns list of (string_index, fret) tuples.
        """
        candidates = []
        for string_idx, open_note in enumerate(self.tuning):
            fret = midi_note - open_note
            if 0 <= fret <= MAX_FRET:
                candidates.append((string_idx, fret))
        return candidates

    def _position_cost(self, candidate, prev_position):
        """
        Cost function for string/fret assignment.

        Lower cost = better assignment.
        """
        string_idx, fret = candidate
        # Prefer lower frets (open strings are best)
        fret_cost = fret * 0.3
        # Prefer staying close to previous position
        move_cost = abs(fret - prev_position) * 0.5
        # Slight preference for middle strings (avoid extremes)
        string_cost = abs(string_idx - 2.5) * 0.1
        return fret_cost + move_cost + string_cost

    def get_tab_notation(self, string_idx, fret):
        """Get human-readable tab notation for a string/fret pair."""
        if string_idx is None or fret is None:
            return 'X'
        return f'{STRING_NAMES[string_idx]}:{fret}'


# Avoid import at module level to keep lightweight
import numpy as np  # noqa: E402
