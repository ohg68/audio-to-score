"""
Audio-to-Score: Streamlit UI for transcribing audio to sheet music.
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

# Add project root to path so src imports work
sys.path.insert(0, str(Path(__file__).parent))

from transcribe import transcribe_file
from src.midi_synthesizer import MidiSynthesizer

SUPPORTED_FORMATS = ['wav', 'mp3', 'flac', 'ogg', 'm4a', 'aac']
INSTRUMENTS = {
    'auto': 'Auto-detect',
    'piano': 'Piano (grand staff)',
    'guitar': 'Guitar (staff + TAB)',
    'voice': 'Voice / Melody',
    'song': 'Song (voice + accompaniment)',
}

st.set_page_config(page_title='Audio to Score', page_icon='🎵', layout='wide')
st.title('Audio to Score')
st.caption('Transcribe audio files to professional sheet music (PDF, MusicXML, MIDI)')

# --- Sidebar controls ---
with st.sidebar:
    st.header('Settings')

    instrument = st.selectbox(
        'Instrument mode',
        options=list(INSTRUMENTS.keys()),
        format_func=lambda k: INSTRUMENTS[k],
        index=0,
    )

    if instrument == 'song':
        st.info('Song mode uses Demucs to separate voice from accompaniment. This takes ~30-60s extra.')

    formats = st.multiselect(
        'Output formats',
        options=['pdf', 'musicxml', 'midi'],
        default=['pdf', 'musicxml', 'midi'],
    )

    confidence = st.slider('Confidence threshold', 0.1, 1.0, 0.5, 0.05,
                           help='Higher = fewer notes, more accurate')

    bpm_override = st.number_input('BPM override (0 = auto-detect)', min_value=0, max_value=300, value=0)

    separate = st.checkbox('Source separation (Demucs)',
                          value=(instrument == 'song'),
                          disabled=(instrument == 'song'),
                          help='Separate mixed audio before transcription')

    custom_title = st.text_input('Custom title (optional)')

# --- Main area ---
uploaded = st.file_uploader(
    'Upload an audio file',
    type=SUPPORTED_FORMATS,
    help='Supported: WAV, MP3, FLAC, OGG, M4A, AAC',
)

if uploaded and st.button('Transcribe', type='primary', use_container_width=True):
    if not formats:
        st.error('Select at least one output format.')
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save uploaded file
            input_path = os.path.join(tmpdir, uploaded.name)
            with open(input_path, 'wb') as f:
                f.write(uploaded.getbuffer())

            output_dir = os.path.join(tmpdir, 'output')
            os.makedirs(output_dir)

            with st.spinner('Transcribing...'):
                import traceback, io
                err_buf = io.StringIO()
                try:
                    result = transcribe_file(
                        filepath=input_path,
                        instrument=instrument,
                        output_dir=output_dir,
                        formats=set(formats),
                        separate=separate,
                        title=custom_title or None,
                        quiet=True,
                        confidence_threshold=confidence,
                        bpm_override=bpm_override,
                    )
                except Exception as e:
                    traceback.print_exc(file=err_buf)
                    result = {'notes_detected': 0, 'error': str(e), 'traceback': err_buf.getvalue()}

            # --- Show results ---
            if result.get('notes_detected', 0) == 0:
                msg = 'No notes detected. Try lowering the confidence threshold or enabling source separation.'
                if result.get('error'):
                    msg = f"Error: {result['error']}"
                st.warning(msg)
                if result.get('traceback'):
                    with st.expander('Error details'):
                        st.code(result['traceback'])
            else:
                st.success(f"Transcription complete — {result['notes_detected']} notes detected")

                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('BPM', f"{result.get('bpm', '?'):.0f}" if result.get('bpm') else '?')
                with col2:
                    st.metric('Key', result.get('key', '?'))
                with col3:
                    st.metric('Time Signature', result.get('time_signature', '?'))

                # Song mode: show per-stem metrics
                if instrument == 'song' and result.get('validation'):
                    v = result['validation']
                    if 'vocals' in v and 'accompaniment' in v:
                        vcol, acol = st.columns(2)
                        with vcol:
                            st.metric('Vocal notes', result.get('vocal_notes', '?'))
                        with acol:
                            st.metric('Accompaniment notes', result.get('accompaniment_notes', '?'))

                # Downloads
                st.divider()
                base_name = Path(uploaded.name).stem

                for output_path in result.get('outputs', []):
                    p = Path(output_path)
                    if p.exists():
                        ext = p.suffix.lstrip('.')
                        label = {'pdf': 'PDF', 'musicxml': 'MusicXML', 'mid': 'MIDI'}.get(ext, ext.upper())
                        mime = {
                            'pdf': 'application/pdf',
                            'musicxml': 'application/vnd.recordare.musicxml+xml',
                            'mid': 'audio/midi',
                        }.get(ext, 'application/octet-stream')

                        with open(output_path, 'rb') as f:
                            st.download_button(
                                f'Download {label}',
                                data=f.read(),
                                file_name=p.name,
                                mime=mime,
                            )

                        # Show PDF inline
                        if ext == 'pdf':
                            with open(output_path, 'rb') as f:
                                pdf_bytes = f.read()
                            import base64
                            b64 = base64.b64encode(pdf_bytes).decode()
                            st.markdown(
                                f'<iframe src="data:application/pdf;base64,{b64}" '
                                f'width="100%" height="800" type="application/pdf"></iframe>',
                                unsafe_allow_html=True,
                            )

                # MIDI Playback
                midi_files = [
                    p for p in result.get('outputs', [])
                    if Path(p).suffix == '.mid' and Path(p).exists()
                ]
                if midi_files:
                    st.divider()
                    st.subheader('Playback')
                    with st.spinner('Synthesizing MIDI...'):
                        synth = MidiSynthesizer()
                        audio = synth.synthesize_from_midi(midi_files[0])
                        wav_path = os.path.join(tmpdir, 'playback.wav')
                        MidiSynthesizer.save_wav(audio, 44100, wav_path)
                        with open(wav_path, 'rb') as f:
                            wav_bytes = f.read()
                    st.audio(wav_bytes, format='audio/wav')
