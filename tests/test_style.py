import ast
import io
import tokenize
from pathlib import Path


def test_claude_md_respects_its_own_length_and_width_rules():
    # CLAUDE.md pins its own limits so the 80/100 numbers cannot drift between the two files.
    path = Path(__file__).resolve().parents[1] / "CLAUDE.md"
    lines = path.read_text().splitlines()
    assert len(lines) <= 80, f"CLAUDE.md has {len(lines)} lines, limit is 80"
    too_long = [(n, len(line)) for n, line in enumerate(lines, 1) if len(line) > 100]
    assert not too_long, f"CLAUDE.md lines over 100 chars: {too_long}"


def docstrings(source, tree):
    """Yields the line number, raw text and first source line of every docstring in a tree."""
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)):
            text = ast.get_docstring(node, clean=False)
            if text is not None:
                yield node.body[0].lineno, text, lines[node.body[0].lineno - 1]


def test_comments_and_docstrings_are_one_short_line():
    # CLAUDE.md pins the rule; walking the tree here keeps the two from drifting apart.
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in sorted(list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))):
        source = path.read_text()
        previous = None
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue
            alone = token.line.lstrip().startswith("#")
            if len(token.line.rstrip()) > 100:
                offenders.append(f"{path.name}:{token.start[0]} comment over 100 chars")
            if alone and previous is not None and token.start[0] == previous + 1:
                offenders.append(f"{path.name}:{token.start[0]} comment block should be one line")
            previous = token.start[0] if alone else None
        for number, text, first in docstrings(source, ast.parse(source)):
            if len(text.strip("\n").split("\n")) > 1:
                offenders.append(f"{path.name}:{number} docstring spans several lines")
            elif len(first.rstrip()) > 100:
                offenders.append(f"{path.name}:{number} docstring over 100 chars")
    assert not offenders, "one line of at most 100 chars, please:\n" + "\n".join(offenders)
