"""Build standalone HTML API/source documentation without executing pipeline stages.

Usage: python scripts/generate_docs.py

The FD001 stage scripts run analyses when imported, so importing them with
ordinary API documentation tools could retrain models or require NASA data.
This AST-based generator (similar in purpose to pdoc) safely documents each
module's docstrings, function signatures, comments and full source code.
"""
from __future__ import annotations
import ast
from html import escape
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'docs' / 'html'

def sources() -> list[Path]:
    """List all maintained Python source modules, including Stage 1–4 scripts."""
    files = [ROOT/'run_all.py', *sorted((ROOT/'scripts').glob('*.py'))]
    for stage in range(1,5):
        files += sorted((ROOT/f'stage{stage}').glob('*.py'))
    return [path for path in files if path.is_file()]

def filename(path: Path) -> str:
    """Create a stable HTML filename from a repository-relative Python path."""
    relative = path.relative_to(ROOT).with_suffix('').as_posix()
    return re.sub(r'[^A-Za-z0-9_-]', '_', relative) + '.html'

def page(title: str, body: str) -> str:
    """Wrap generated HTML content in accessible page chrome and styling."""
    style = ('body{font:15px/1.65 system-ui,Arial;max-width:1100px;'
             'margin:auto;padding:24px;background:#f7f9fb;color:#1d2c38}'
             'a{color:#176a9d}.card{background:white;border:1px solid #dbe4eb;'
             'border-radius:9px;padding:18px;margin:18px 0}'
             'pre{background:#152635;color:#f2f7fa;padding:18px;overflow:auto;'
             'border-radius:8px;font:12px/1.55 ui-monospace,monospace}'
             '.doc{background:#ecf3f8;color:#213c50;white-space:pre-wrap}'
             '.line{display:block;white-space:pre}.number{display:inline-block;'
             'width:4em;text-align:right;padding-right:1em;color:#8fa5b2}'
             'code{overflow-wrap:anywhere}h1{color:#14354c}')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title><style>{style}</style></head>'
            f'<body><p><a href="index.html">← API documentation index</a></p>'
            f'<h1>{escape(title)}</h1>{body}'
            '<footer><p>Generated from source code using Python AST. '
            'No training data was accessed.</p></footer></body></html>\n')

def render_source(path: Path) -> str:
    """Render an individual module's API summaries and line-numbered source."""
    text = path.read_text(encoding='utf-8')
    tree = ast.parse(text, filename=str(path))
    relative = path.relative_to(ROOT).as_posix()
    overview = ast.get_docstring(tree) or 'Sequential Python analysis module.'
    sections = [f'<section class="card"><h2>Module overview</h2>'
                f'<pre class="doc">{escape(overview)}</pre></section>']
    functions = [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
    sections.append('<section class="card"><h2>Functions and classes</h2>')
    if not functions:
        sections.append('<p>This is a sequential script. See its annotated source below.</p>')
    for item in functions:
        signature = item.name + (f'({ast.unparse(item.args)})' if hasattr(item,'args') else '')
        doc = ast.get_docstring(item) or '(No function-level docstring)'
        sections.append(f'<h3><code>{escape(signature)}</code></h3>'
                        f'<pre class="doc">{escape(doc)}</pre>')
    sections.append('</section><section class="card"><h2>Annotated source code</h2><pre>')
    for i,line in enumerate(text.splitlines(),1):
        sections.append(f'<span class="line" id="L{i}"><span class="number">{i}</span>{escape(line)}</span>')
    sections.append('</pre></section>')
    return page(relative, ''.join(sections))

def main() -> None:
    """Generate one HTML page per Python module and a linked API index."""
    OUTPUT.mkdir(parents=True,exist_ok=True)
    cards=[]
    for path in sources():
        name=filename(path)
        (OUTPUT/name).write_text(render_source(path),encoding='utf-8')
        cards.append(f'<li><a href="{name}"><code>{escape(path.relative_to(ROOT).as_posix())}</code></a></li>')
    index=('<section class="card"><p>HTML source-code documentation for '
           'MMA3001 NASA FD001. Each page includes the module documentation, '
           'function signatures, docstrings and full source comments.</p>'
           '<p>Rebuild with <code>python scripts/generate_docs.py</code>. '
           'Source files are parsed rather than imported to avoid unintentionally '
           'running training or opening test labels.</p><ul>'
           + ''.join(cards) + '</ul></section>')
    (OUTPUT/'index.html').write_text(page('MMA3001 Turbofan RUL — Python API',index),encoding='utf-8')
    print(f'Generated {len(cards)} API pages in {OUTPUT}')

if __name__=='__main__':
    main()
