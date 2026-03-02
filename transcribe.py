#!/usr/bin/env python3
"""
Audio-to-Score: Transcribe audio files to professional sheet music.

Outputs PDF (printable), MusicXML (editable in Finale/Sibelius/MuseScore),
and MIDI (DAW-compatible) from WAV/MP3 input.
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from src.audio_processor import AudioProcessor
from src.pitch_detector import PitchDetector
from src.rhythm_analyzer import RhythmAnalyzer
from src.score_builder import ScoreBuilder
from src.pdf_exporter import PDFExporter
from src.exporters import MusicXMLExporter, MIDIExporter


SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
ALL_OUTPUT_FORMATS = {'pdf', 'musicxml', 'midi'}
INSTRUMENTS = {'piano', 'guitar', 'voice', 'song', 'auto'}


def parse_args():
    parser = argparse.ArgumentParser(
        prog='transcribe',
        description='Transcribe audio files to sheet music (PDF, MusicXML, MIDI).',
        epilog='Examples:\n'
               '  python transcribe.py song.mp3\n'
               '  python transcribe.py song.wav --instrument piano\n'
               '  python transcribe.py *.mp3 --instrument guitar --output ./scores\n'
               '  python transcribe.py song.mp3 --formats pdf musicxml\n'
               '  python transcribe.py mix.mp3 --separate --instrument piano\n',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'input',
        nargs='+',
        help='Audio file(s) to transcribe (WAV, MP3, FLAC, OGG, M4A, AAC). '
             'Supports glob patterns.',
    )
    parser.add_argument(
        '--instrument', '-i',
        choices=sorted(INSTRUMENTS),
        default='auto',
        help='Instrument mode: piano (grand staff), guitar (staff+TAB), '
             'voice (single staff with pYIN), song (voice + accompaniment), '
             'auto (detect). Default: auto.',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory. Defaults to same directory as input file.',
    )
    parser.add_argument(
        '--formats', '-f',
        nargs='+',
        choices=sorted(ALL_OUTPUT_FORMATS),
        default=list(ALL_OUTPUT_FORMATS),
        help='Output formats. Default: pdf musicxml midi.',
    )
    parser.add_argument(
        '--separate', '-s',
        action='store_true',
        help='Use Demucs to separate sources before transcription '
             '(useful for mixed recordings).',
    )
    parser.add_argument(
        '--title', '-t',
        type=str,
        default=None,
        help='Custom title for the score. Defaults to filename.',
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output.',
    )
    parser.add_argument(
        '--confidence', '-c',
        type=float,
        default=0.5,
        help='Confidence threshold for note detection (0.0-1.0). '
             'Higher = fewer notes, more accurate. Default: 0.5.',
    )
    parser.add_argument(
        '--bpm',
        type=float,
        default=0,
        help='Override auto-detected BPM. 0 = auto-detect. Default: 0.',
    )
    parser.add_argument(
        '--key',
        type=str,
        default=None,
        help='Override auto-detected key signature (e.g. "C major", "A minor").',
    )

    args = parser.parse_args()

    # Expand glob patterns and collect files
    expanded = []
    for pattern in args.input:
        matches = glob.glob(pattern)
        if matches:
            expanded.extend(matches)
        elif os.path.isfile(pattern):
            expanded.append(pattern)
        else:
            print(f"Warning: '{pattern}' not found, skipping.", file=sys.stderr)

    if not expanded:
        parser.error('No valid input files found.')

    # Filter supported formats
    valid_files = []
    for f in expanded:
        ext = Path(f).suffix.lower()
        if ext in SUPPORTED_FORMATS:
            valid_files.append(f)
        else:
            print(f"Warning: '{f}' format not supported, skipping.", file=sys.stderr)

    if not valid_files:
        parser.error('No supported audio files found.')

    args.input = valid_files
    args.formats = set(args.formats)

    return args


def transcribe_file(filepath, instrument, output_dir, formats, separate, title, quiet,
                    confidence_threshold=0.5, bpm_override=0, key_override=None):
    """Transcribe a single audio file to sheet music."""
    path = Path(filepath)
    base_name = path.stem
    score_title = title or base_name.replace('_', ' ').replace('-', ' ').title()

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
    else:
        out = path.parent

    results = {
        'file': str(path),
        'title': score_title,
        'outputs': [],
        'key': None,
        'bpm': None,
        'time_signature': None,
        'notes_detected': 0,
    }

    is_song = (instrument == 'song')

    if is_song:
        steps = [
            'Loading audio',
            'Separating sources (Demucs)',
            'Analyzing rhythm',
            'Detecting vocal melody (pYIN)',
            'Detecting accompaniment (basic-pitch)',
            'Validating vocals',
            'Validating accompaniment',
            'Building score',
        ]
    else:
        steps = [
            'Loading audio',
            'Analyzing rhythm',
            'Detecting pitches',
            'Validating transcription',
            'Building score',
        ]
    if 'pdf' in formats:
        steps.append('Exporting PDF')
    if 'musicxml' in formats:
        steps.append('Exporting MusicXML')
    if 'midi' in formats:
        steps.append('Exporting MIDI')

    progress = tqdm(steps, desc=f'  {base_name}', disable=quiet, leave=True)

    try:
        # --- Step 1: Load and preprocess audio ---
        progress.set_description(f'  {base_name} | Loading audio')
        processor = AudioProcessor()
        audio_data = processor.load(
            str(path),
            separate=(separate and not is_song),  # Song mode handles separation itself
            instrument=instrument,
        )
        progress.update(1)

        if is_song:
            # =============================================================
            # SONG MODE: Dual pipeline (vocals + accompaniment)
            # =============================================================

            # --- Step 2: Source separation ---
            progress.set_description(f'  {base_name} | Separating sources (~30-60s)')
            stems = processor.separate_sources_dual(audio_data)
            progress.update(1)

            # --- Step 3: Rhythm analysis on original mix (best signal) ---
            progress.set_description(f'  {base_name} | Analyzing rhythm')
            rhythm = RhythmAnalyzer()
            rhythm_info = rhythm.analyze(audio_data['waveform'], audio_data['sr'])
            if bpm_override > 0:
                rhythm_info['bpm'] = float(bpm_override)
                rhythm_info['quarter_duration'] = 60.0 / float(bpm_override)
            results['bpm'] = rhythm_info['bpm']
            results['time_signature'] = rhythm_info['time_signature']
            progress.update(1)

            # --- Step 4: Detect vocal melody with pYIN ---
            progress.set_description(f'  {base_name} | Detecting vocal melody')
            detector = PitchDetector()
            vocal_notes = detector.detect_pyin(
                stems['vocals'], stems['sr'],
                instrument='voice',
                confidence_threshold=confidence_threshold,
            )
            results['vocal_notes'] = len(vocal_notes)
            progress.update(1)

            # --- Step 5: Detect accompaniment with basic-pitch ---
            progress.set_description(f'  {base_name} | Detecting accompaniment')
            # Write accompaniment to temp file for basic-pitch
            import tempfile as _tmpfile
            import soundfile as _sf
            with _tmpfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_acc:
                _sf.write(tmp_acc.name, stems['accompaniment'], stems['sr'])
                acc_path = tmp_acc.name
            # Higher threshold for accompaniment to reduce vocal bleed artifacts
            acc_threshold = max(confidence_threshold, 0.6)
            accomp_notes = detector.detect_basic_pitch(
                acc_path, stems['sr'],
                confidence_threshold=acc_threshold,
            )
            os.unlink(acc_path)
            results['accompaniment_notes'] = len(accomp_notes)
            progress.update(1)

            if not vocal_notes and not accomp_notes:
                print(f"\n  Warning: No notes detected in '{path.name}'.",
                      file=sys.stderr)
                progress.close()
                return results

            # Quantize each stem
            vocal_notes = rhythm.quantize(vocal_notes, rhythm_info)
            accomp_notes = rhythm.quantize(accomp_notes, rhythm_info)

            # --- Step 6: Validate vocals ---
            progress.set_description(f'  {base_name} | Validating vocals')
            from src.transcription_validator import TranscriptionValidator
            validator = TranscriptionValidator(max_iterations=2)
            vocal_audio = {
                'waveform': stems['vocals'], 'sr': stems['sr'],
            }
            vocal_notes, vocal_report = validator.validate_and_correct(
                vocal_notes, vocal_audio, rhythm_info, instrument_mode='voice',
            )
            progress.update(1)

            # --- Step 7: Validate accompaniment ---
            progress.set_description(f'  {base_name} | Validating accompaniment')
            accomp_audio = {
                'waveform': stems['accompaniment'], 'sr': stems['sr'],
            }
            accomp_notes, accomp_report = validator.validate_and_correct(
                accomp_notes, accomp_audio, rhythm_info, instrument_mode='piano',
            )
            progress.update(1)

            # Combined results
            all_notes = vocal_notes + accomp_notes
            results['notes_detected'] = len(all_notes)
            results['validation'] = {
                'vocals': vocal_report.to_dict(),
                'accompaniment': accomp_report.to_dict(),
            }

            # --- Step 8: Build song score ---
            progress.set_description(f'  {base_name} | Building score')
            builder = ScoreBuilder()
            score = builder.build(
                notes=all_notes,
                instrument_mode='song',
                title=score_title,
                bpm=rhythm_info['bpm'],
                time_signature=rhythm_info['time_signature'],
                audio_data=audio_data,
                vocal_notes=vocal_notes,
                accompaniment_notes=accomp_notes,
            )

            notes = all_notes  # For export compatibility

        else:
            # =============================================================
            # STANDARD MODE: Single pipeline
            # =============================================================

            # --- Step 2: Analyze rhythm ---
            progress.set_description(f'  {base_name} | Analyzing rhythm')
            rhythm = RhythmAnalyzer()
            rhythm_info = rhythm.analyze(audio_data['waveform'], audio_data['sr'])
            if bpm_override > 0:
                rhythm_info['bpm'] = float(bpm_override)
                rhythm_info['quarter_duration'] = 60.0 / float(bpm_override)
            results['bpm'] = rhythm_info['bpm']
            results['time_signature'] = rhythm_info['time_signature']
            progress.update(1)

            # --- Step 3: Detect pitches ---
            progress.set_description(f'  {base_name} | Detecting pitches')
            detector = PitchDetector()
            notes = detector.detect(
                audio_data['waveform'],
                audio_data['sr'],
                instrument=instrument,
                filepath=audio_data['filepath'],
                confidence_threshold=confidence_threshold,
            )

            if not notes:
                print(f"\n  Warning: No notes detected in '{path.name}'. "
                      f"The audio may be silent or too noisy. Try --separate.",
                      file=sys.stderr)
                progress.close()
                return results

            results['notes_detected'] = len(notes)
            notes = rhythm.quantize(notes, rhythm_info)
            progress.update(1)

            # --- Step 4: Validate and auto-correct ---
            progress.set_description(f'  {base_name} | Validating transcription')
            from src.transcription_validator import TranscriptionValidator
            validator = TranscriptionValidator(max_iterations=2)
            notes, validation_report = validator.validate_and_correct(
                notes, audio_data, rhythm_info, instrument_mode=instrument,
            )
            results['notes_detected'] = len(notes)
            results['validation'] = validation_report.to_dict()
            progress.update(1)

            # --- Step 5: Build music21 score ---
            progress.set_description(f'  {base_name} | Building score')
            builder = ScoreBuilder()
            score = builder.build(
                notes=notes,
                instrument_mode=instrument,
                title=score_title,
                bpm=rhythm_info['bpm'],
                time_signature=rhythm_info['time_signature'],
                audio_data=audio_data,
            )

        # Apply key override if set
        if key_override:
            from music21 import key as m21key
            try:
                parts_str = key_override.split()
                ks = m21key.Key(parts_str[0], parts_str[1])
                for part in score.parts:
                    for existing_ks in part.getElementsByClass(m21key.Key):
                        part.remove(existing_ks)
                    part.insert(0, ks)
                builder.detected_key = key_override
            except Exception:
                pass

        results['key'] = builder.detected_key
        progress.update(1)

        # --- Step 5+: Export ---
        if 'musicxml' in formats:
            progress.set_description(f'  {base_name} | Exporting MusicXML')
            xml_path = out / f'{base_name}.musicxml'
            MusicXMLExporter.export(score, str(xml_path))
            results['outputs'].append(str(xml_path))
            progress.update(1)

        if 'midi' in formats:
            progress.set_description(f'  {base_name} | Exporting MIDI')
            midi_path = out / f'{base_name}.mid'
            MIDIExporter.export(score, str(midi_path))
            results['outputs'].append(str(midi_path))
            progress.update(1)

        if 'pdf' in formats:
            progress.set_description(f'  {base_name} | Exporting PDF')
            pdf_path = out / f'{base_name}.pdf'
            xml_for_pdf = out / f'{base_name}.musicxml'
            # Ensure MusicXML exists for PDF rendering
            if not xml_for_pdf.exists():
                MusicXMLExporter.export(score, str(xml_for_pdf))
            PDFExporter.export(str(xml_for_pdf), str(pdf_path), score=score)
            results['outputs'].append(str(pdf_path))
            # Clean up temp MusicXML if user didn't request it
            if 'musicxml' not in formats and xml_for_pdf.exists():
                xml_for_pdf.unlink()
            progress.update(1)

        progress.close()

    except Exception as e:
        progress.close()
        print(f"\n  Error processing '{path.name}': {e}", file=sys.stderr)

    return results


def print_summary(all_results):
    """Print a summary table of all transcription results."""
    print('\n' + '=' * 60)
    print('  TRANSCRIPTION SUMMARY')
    print('=' * 60)

    for r in all_results:
        print(f'\n  File: {r["file"]}')
        if r['key']:
            print(f'    Key:            {r["key"]}')
        if r['bpm']:
            print(f'    Tempo:          {r["bpm"]} BPM')
        if r['time_signature']:
            print(f'    Time Signature: {r["time_signature"]}')
        print(f'    Notes Detected: {r["notes_detected"]}')
        if r.get('validation'):
            v = r['validation']
            # Song mode has separate vocal/accompaniment reports
            if 'vocals' in v and 'accompaniment' in v:
                vv = v['vocals']
                va = v['accompaniment']
                print(f'    Vocals:         {r.get("vocal_notes", "?")} notes, '
                      f'{vv["overall_similarity_after"]:.0f}% similarity '
                      f'(+{vv["improvement"]:.1f}%)')
                print(f'    Accompaniment:  {r.get("accompaniment_notes", "?")} notes, '
                      f'{va["overall_similarity_after"]:.0f}% similarity '
                      f'(+{va["improvement"]:.1f}%)')
            else:
                print(f'    Validation:     {v["overall_similarity_after"]:.0f}% similarity '
                      f'(+{v["improvement"]:.1f}% improvement)')
                print(f'    Confirmed: {v["confirmed_notes"]} | '
                      f'Corrected: {v["corrected_notes"]} | '
                      f'Ghosts removed: {v["removed_ghost_notes"]}')
        if r['outputs']:
            print(f'    Outputs:')
            for o in r['outputs']:
                print(f'      -> {o}')
        else:
            print('    Outputs:        (none - no notes detected)')

    print('\n' + '=' * 60)


def main():
    args = parse_args()

    if not args.quiet:
        print('\n  Audio-to-Score Transcriber')
        print(f'  Instrument: {args.instrument}')
        print(f'  Formats:    {", ".join(sorted(args.formats))}')
        if args.separate:
            print('  Source separation: enabled')
        print(f'  Files:      {len(args.input)}')
        print()

    all_results = []

    for filepath in args.input:
        result = transcribe_file(
            filepath=filepath,
            instrument=args.instrument,
            output_dir=args.output,
            formats=args.formats,
            separate=args.separate,
            title=args.title,
            quiet=args.quiet,
            confidence_threshold=args.confidence,
            bpm_override=args.bpm,
            key_override=args.key,
        )
        all_results.append(result)

    if not args.quiet:
        print_summary(all_results)


if __name__ == '__main__':
    main()
