"""
MusicXML and MIDI exporters using music21.
"""

import os
import sys


class MusicXMLExporter:
    """Export a music21 Score to MusicXML format."""

    @staticmethod
    def export(score, output_path):
        """
        Export score to MusicXML file.

        Parameters
        ----------
        score : music21.stream.Score
            The score to export.
        output_path : str
            Path for the output .musicxml file.
        """
        output_path = str(output_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        try:
            score.write('musicxml', fp=output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to export MusicXML: {e}")

        if not os.path.exists(output_path):
            raise RuntimeError(f"MusicXML export produced no file at {output_path}")


class MIDIExporter:
    """Export a music21 Score to MIDI format."""

    @staticmethod
    def export(score, output_path):
        """
        Export score to MIDI file.

        Parameters
        ----------
        score : music21.stream.Score
            The score to export.
        output_path : str
            Path for the output .mid file.
        """
        output_path = str(output_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        try:
            score.write('midi', fp=output_path)
        except Exception as e:
            raise RuntimeError(f"Failed to export MIDI: {e}")

        if not os.path.exists(output_path):
            raise RuntimeError(f"MIDI export produced no file at {output_path}")
