"""
PDF export from MusicXML using multiple backends.

Render chain (tries in order):
  1. verovio (Python, cross-platform) + svglib/reportlab or cairosvg
  2. LilyPond CLI (via music21 conversion, with syntax post-processing)
  3. MuseScore CLI

Renders A4 sheet music with:
  - 3-4 systems per page
  - Centered title, tempo, time signature
  - Page numbers in footer
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# LilyPond syntax replacements for v2.20+ / v2.24+ compatibility
_LILYPOND_SYNTAX_FIXES = [
    # RemoveEmptyStaffContext → RemoveAllEmptyStaves (deprecated in 2.18)
    (r'\\RemoveEmptyStaffContext', r'\\RemoveAllEmptyStaves'),
    # Old context property syntax
    (r"\\override\s+VerticalAxisGroup\s+#'remove-first\s*=\s*##t",
     r'\\RemoveAllEmptyStaves'),
    # Remove empty \with {} blocks that cause parse errors
    (r'\\with\s*\{\s*\}', ''),
    # Remove lilypond-book-preamble (breaks standalone PDF layout)
    (r'\\include\s+"lilypond-book-preamble\.ly"\s*\n?', ''),
]

# LilyPond paper/layout block injected after \header for proper A4 sheet music
_LILYPOND_LAYOUT_BLOCK = r"""
\paper {
  #(set-paper-size "a4")
  top-margin = 15\mm
  bottom-margin = 15\mm
  left-margin = 15\mm
  right-margin = 15\mm
  indent = 10\mm
  system-system-spacing.basic-distance = #18
  system-system-spacing.minimum-distance = #14
  system-system-spacing.padding = #3
  score-system-spacing.basic-distance = #20
  markup-system-spacing.basic-distance = #12
}

\layout {
  \context {
    \Score
    \override SpacingSpanner.common-shortest-duration = #(ly:make-moment 1/8)
  }
}
"""


class PDFExporter:
    """Export MusicXML to PDF via verovio, LilyPond, or MuseScore CLI."""

    @staticmethod
    def export(musicxml_path, pdf_path, score=None):
        """
        Render a MusicXML file to PDF.

        Parameters
        ----------
        musicxml_path : str
            Path to input MusicXML file.
        pdf_path : str
            Path for output PDF file.
        score : music21.stream.Score, optional
            The score object (used for LilyPond export).
        """
        musicxml_path = str(musicxml_path)
        pdf_path = str(pdf_path)

        if not os.path.exists(musicxml_path):
            raise FileNotFoundError(f"MusicXML file not found: {musicxml_path}")

        # Try verovio first
        if PDFExporter._try_verovio(musicxml_path, pdf_path):
            return

        # Try LilyPond
        if PDFExporter._try_lilypond(musicxml_path, pdf_path, score):
            return

        # Fallback to MuseScore CLI
        if PDFExporter._try_musescore(musicxml_path, pdf_path):
            return

        raise RuntimeError(
            "No PDF renderer available.\n\n"
            "Install one of:\n"
            "  1. verovio + svglib + reportlab: pip install verovio svglib reportlab\n"
            "  2. LilyPond: brew install lilypond (macOS) / apt install lilypond (Linux)\n"
            "  3. MuseScore 4: https://musescore.org/download\n"
        )

    @staticmethod
    def _try_verovio(musicxml_path, pdf_path):
        """Render PDF using the verovio Python package."""
        try:
            import verovio
        except ImportError:
            return False

        try:
            tk = verovio.toolkit()
            options = {
                'pageWidth': 2100,
                'pageHeight': 2970,
                'pageMarginTop': 120,
                'pageMarginBottom': 120,
                'pageMarginLeft': 100,
                'pageMarginRight': 80,
                'spacingSystem': 18,
                'spacingStaff': 12,
                'scale': 40,
                'adjustPageHeight': False,
                'font': 'Bravura',
                'footer': 'auto',
                'breaks': 'auto',
                'condense': 'auto',
            }
            tk.setOptions(options)

            with open(musicxml_path, 'r', encoding='utf-8') as f:
                musicxml_content = f.read()

            if not tk.loadData(musicxml_content):
                return False

            page_count = tk.getPageCount()
            if page_count == 0:
                return False

            # Render SVG pages and convert to PDF
            svg_pages = []
            for page_num in range(1, page_count + 1):
                svg = tk.renderToSVG(page_num)
                if svg:
                    svg_pages.append(svg)

            if not svg_pages:
                return False

            # Try cairosvg first (best quality for music glyphs)
            try:
                import cairosvg

                with tempfile.TemporaryDirectory() as tmpdir:
                    page_pdfs = []
                    for i, svg_str in enumerate(svg_pages):
                        page_pdf = os.path.join(tmpdir, f'page_{i}.pdf')
                        cairosvg.svg2pdf(
                            bytestring=svg_str.encode('utf-8'),
                            write_to=page_pdf,
                        )
                        if os.path.exists(page_pdf) and os.path.getsize(page_pdf) > 0:
                            page_pdfs.append(page_pdf)

                    if page_pdfs:
                        if len(page_pdfs) == 1:
                            shutil.copy2(page_pdfs[0], pdf_path)
                        else:
                            PDFExporter._merge_pdfs(page_pdfs, pdf_path)
                        return True
            except ImportError:
                pass

            # Try svglib + reportlab (pure Python fallback)
            try:
                from svglib.svglib import svg2rlg
                from reportlab.graphics import renderPDF

                with tempfile.TemporaryDirectory() as tmpdir:
                    page_pdfs = []
                    for i, svg_str in enumerate(svg_pages):
                        svg_file = os.path.join(tmpdir, f'page_{i}.svg')
                        with open(svg_file, 'w', encoding='utf-8') as f:
                            f.write(svg_str)
                        drawing = svg2rlg(svg_file)
                        if drawing:
                            page_pdf = os.path.join(tmpdir, f'page_{i}.pdf')
                            renderPDF.drawToFile(drawing, page_pdf)
                            if os.path.exists(page_pdf) and os.path.getsize(page_pdf) > 0:
                                page_pdfs.append(page_pdf)

                    if page_pdfs:
                        if len(page_pdfs) == 1:
                            shutil.copy2(page_pdfs[0], pdf_path)
                        else:
                            PDFExporter._merge_pdfs(page_pdfs, pdf_path)
                        return True
            except ImportError:
                pass

            # Try rsvg-convert or inkscape as last resort
            with tempfile.TemporaryDirectory() as tmpdir:
                page_pdfs = []
                for i, svg_str in enumerate(svg_pages):
                    svg_file = os.path.join(tmpdir, f'page_{i}.svg')
                    with open(svg_file, 'w', encoding='utf-8') as f:
                        f.write(svg_str)

                    for tool, args_fn in [
                        ('rsvg-convert', lambda s, p: ['-f', 'pdf', '-o', p, s]),
                        ('inkscape', lambda s, p: [s, '--export-type=pdf', f'--export-filename={p}']),
                    ]:
                        tool_path = shutil.which(tool)
                        if tool_path:
                            try:
                                page_pdf = os.path.join(tmpdir, f'page_{i}.pdf')
                                subprocess.run(
                                    [tool_path] + args_fn(svg_file, page_pdf),
                                    capture_output=True, check=True,
                                )
                                if os.path.exists(page_pdf) and os.path.getsize(page_pdf) > 0:
                                    page_pdfs.append(page_pdf)
                                    break
                            except (subprocess.CalledProcessError, FileNotFoundError):
                                continue

                if page_pdfs:
                    if len(page_pdfs) == 1:
                        shutil.copy2(page_pdfs[0], pdf_path)
                    else:
                        PDFExporter._merge_pdfs(page_pdfs, pdf_path)
                    return True

            return False

        except Exception as e:
            print(f"  Warning: verovio rendering failed: {e}", file=sys.stderr)
            return False

    @staticmethod
    def _fix_lilypond_syntax(ly_path):
        """Post-process a .ly file to fix deprecated syntax and inject layout for LilyPond 2.20+/2.24+."""
        with open(ly_path, 'r', encoding='utf-8') as f:
            content = f.read()

        for pattern, replacement in _LILYPOND_SYNTAX_FIXES:
            content = re.sub(pattern, replacement, content)

        # Inject paper/layout block after \header{...} if not already present
        if r'\paper' not in content:
            header_match = re.search(r'(\\header\s*\{[^}]*\})', content, re.DOTALL)
            if header_match:
                insert_pos = header_match.end()
                content = content[:insert_pos] + _LILYPOND_LAYOUT_BLOCK + content[insert_pos:]
            else:
                # No header — inject at the top after \version line
                version_match = re.search(r'(\\version\s+"[^"]+"\s*\n)', content)
                if version_match:
                    insert_pos = version_match.end()
                    content = content[:insert_pos] + _LILYPOND_LAYOUT_BLOCK + content[insert_pos:]

        with open(ly_path, 'w', encoding='utf-8') as f:
            f.write(content)

    @staticmethod
    def _try_lilypond(musicxml_path, pdf_path, score=None):
        """
        Render PDF using LilyPond.

        Converts MusicXML to LilyPond format via music21, post-processes
        deprecated syntax, then runs lilypond CLI.
        """
        lilypond_bin = shutil.which('lilypond')
        if not lilypond_bin:
            return False

        try:
            from music21 import converter

            if score is None:
                score = converter.parse(musicxml_path)

            with tempfile.TemporaryDirectory() as tmpdir:
                ly_path = os.path.join(tmpdir, 'score.ly')

                # Generate .ly file via music21
                ly_content = score.write('lily', fp=ly_path)
                if ly_content and os.path.exists(str(ly_content)):
                    ly_path = str(ly_content)

                if not os.path.exists(ly_path):
                    return False

                # Fix deprecated syntax for modern LilyPond
                PDFExporter._fix_lilypond_syntax(ly_path)

                # Run lilypond CLI
                result = subprocess.run(
                    [
                        lilypond_bin,
                        f'--output={os.path.join(tmpdir, "score")}',
                        '--pdf',
                        ly_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=tmpdir,
                )

                generated_pdf = os.path.join(tmpdir, 'score.pdf')
                if os.path.exists(generated_pdf):
                    shutil.move(generated_pdf, pdf_path)
                    return True

                # Log errors for debugging
                if result.stderr:
                    print(f"  LilyPond stderr: {result.stderr[:500]}", file=sys.stderr)

        except Exception as e:
            print(f"  Warning: LilyPond rendering failed: {e}", file=sys.stderr)

        return False

    @staticmethod
    def _try_musescore(musicxml_path, pdf_path):
        """Render PDF using MuseScore CLI."""
        mscore = PDFExporter._find_musescore()
        if not mscore:
            return False

        try:
            subprocess.run(
                [mscore, '-o', pdf_path, musicxml_path],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            return os.path.exists(pdf_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"  Warning: MuseScore failed: {e}", file=sys.stderr)
            return False

    @staticmethod
    def _merge_pdfs(pdf_paths, output_path):
        """Merge multiple single-page PDFs into one file."""
        try:
            from pypdf import PdfWriter
            writer = PdfWriter()
            for p in pdf_paths:
                writer.append(p)
            with open(output_path, 'wb') as f:
                writer.write(f)
        except ImportError:
            # pypdf not installed — fall back to first page only
            shutil.copy2(pdf_paths[0], output_path)

    @staticmethod
    def _find_musescore():
        """Find the MuseScore executable on the system."""
        for name in ['mscore', 'musescore', 'MuseScore4', 'MuseScore 4', 'mscore4']:
            path = shutil.which(name)
            if path:
                return path

        if sys.platform == 'darwin':
            for p in [
                '/Applications/MuseScore 4.app/Contents/MacOS/mscore',
                '/Applications/MuseScore 3.app/Contents/MacOS/mscore',
            ]:
                if os.path.exists(p):
                    return p

        elif sys.platform == 'win32':
            prog_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
            for p in [
                os.path.join(prog_files, 'MuseScore 4', 'bin', 'MuseScore4.exe'),
                os.path.join(prog_files, 'MuseScore 3', 'bin', 'MuseScore3.exe'),
            ]:
                if os.path.exists(p):
                    return p

        return None
