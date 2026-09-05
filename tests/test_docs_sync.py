"""Cheap checks that the documentation still describes the code.

Prose drifts silently.  Nothing fails when a dtype is added to the exporter
and not to the table, or when a ``throw`` grows a new message that no page
explains, or when a code sample's markers are renamed out from under a
``literalinclude`` -- the docs simply become wrong, and stay wrong until a
reader is misled by them.  These four checks cost milliseconds and catch the
kinds of drift that a human review reliably misses.

None of them tries to check that the prose is *good*, only that the nouns in
it still exist.  Where a rule cannot be enforced without guessing at intent
-- a marker that no page happens to use is perfectly legitimate -- it is
reported as a warning instead of a failure.
"""

from __future__ import annotations

import pathlib
import re
import tomllib
import warnings

import pytest

#: Directories that are generated, vendored, built, or not the subject.
#: ``tests`` is excluded because no page includes code from the suite, and
#: scanning it would find this file's own regexes and report them as
#: markers.
IGNORED = {
    "_build",
    "third_party",
    ".git",
    "build",
    ".venv",
    "__pycache__",
    "tests",
}

#: Files that can carry a ``docs: begin`` marker.
MARKER_SUFFIXES = {".c", ".cpp", ".hpp", ".py", ".sh", ".mk", ".txt", ".env"}

#: One string literal, escapes included.
LITERAL = re.compile(r'"((?:[^"\\]|\\.)*)"')

#: The escapes that appear in these messages.
UNESCAPE = {'\\"': '"', "\\\\": "\\", "\\n": "\n", "\\t": "\t"}

#: A ``literalinclude`` block and the options that follow it.
INCLUDE = re.compile(
    r"^```\{literalinclude\}[ \t]+(?P<path>\S+)[ \t]*\n"
    r"(?P<options>(?:[ \t]*:[^\n]*\n)*)",
    re.MULTILINE,
)

#: ``:start-after: docs: begin <name>``.
START_AFTER = re.compile(
    r"^[ \t]*:start-after:[ \t]*docs: begin (\S+)", re.MULTILINE
)

#: A ``docs: begin <name>`` marker in a source file.
MARKER = re.compile(r"docs: begin (\S+)")

#: Below this many characters a message prefix is a fragment like ``" ('"``
#: or ``"output "``, which would match half the page by accident and says
#: nothing about whether the message itself is documented.
MIN_DISTINCTIVE = 12


def unescape(literal):
    return re.sub(
        r"\\.", lambda m: UNESCAPE.get(m.group(0), m.group(0)), literal
    )


def sources(root):
    """Every file in the checkout that may carry a marker."""
    for path in sorted(pathlib.Path(root).rglob("*")):
        if not path.is_file():
            continue
        if IGNORED & set(path.parts):
            continue
        if path.suffix in MARKER_SUFFIXES:
            yield path


def pages(root):
    """Every documentation page that is a source, not a build product."""
    for path in sorted(pathlib.Path(root, "docs").rglob("*.md")):
        if not IGNORED & set(path.parts):
            yield path


def throw_arguments(text):
    """The argument list of every ``throw`` expression in @p text."""
    for match in re.finditer(r"\bthrow\b", text):
        depth, start, index = 0, None, match.end()
        while index < len(text):
            char = text[index]
            if char == "(":
                depth += 1
                if start is None:
                    start = index
            elif char == ")":
                depth -= 1
                if depth == 0:
                    yield text[start + 1 : index]
                    break
            elif char == ";" and depth == 0:
                break
            index += 1


def message_prefixes(root):
    """The documentable literal from each ``throw`` in the library.

    The first string literal in the expression, which is the message's
    literal prefix before it is concatenated with a path, an index or a
    plugin's own words.  ``throw LoadError(path + " is not valid JSON: ")``
    contributes the second half, since that is the part a reader can grep
    for; anything shorter than :data:`MIN_DISTINCTIVE` is dropped as a
    fragment rather than a message.
    """
    found = {}
    for path in sorted(pathlib.Path(root, "src/pjrt_exec").glob("*.cpp")):
        for expression in throw_arguments(path.read_text()):
            literals = LITERAL.findall(expression)
            if not literals:
                continue
            prefix = unescape(literals[0]).strip()
            if len(prefix) >= MIN_DISTINCTIVE:
                found.setdefault(prefix, path.name)
    return found


@pytest.fixture(scope="module")
def jax2exec():
    return pytest.importorskip("jax2exec")


def test_every_supported_dtype_is_in_the_exporting_guide(repo, jax2exec):
    """The dtype table is the contract; the guide is where it is read.

    A type added to the exporter and not to the table is a type nobody
    knows they may use.
    """
    guide = pathlib.Path(repo.root, "docs/guides/exporting.md").read_text()
    missing = [name for name in jax2exec.SUPPORTED_DTYPES if name not in guide]
    assert not missing, f"not in docs/guides/exporting.md: {missing}"


def test_every_throw_message_has_a_row_in_the_debugging_guide(repo):
    """Every distinct failure the loader can report is documented.

    The table in docs/guides/debugging.md is the first thing anyone hits
    when a load fails, and a message that is not in it sends the reader to
    the source instead.
    """
    guide = pathlib.Path(repo.root, "docs/guides/debugging.md").read_text()
    prefixes = message_prefixes(repo.root)
    assert prefixes, "no throw messages found; did the sources move?"
    missing = sorted(
        f"{origin}: {prefix!r}"
        for prefix, origin in prefixes.items()
        if prefix not in guide
    )
    assert not missing, (
        "no row in docs/guides/debugging.md for:\n  " + "\n  ".join(missing)
    )


def test_every_included_marker_exists(repo):
    """Each ``literalinclude`` resolves to a file and finds its marker.

    A renamed marker does not break the docs build loudly; Sphinx emits a
    warning and renders an empty block, so the page silently loses the code
    it was written around.
    """
    broken = []
    referenced = set()
    for page in pages(repo.root):
        text = page.read_text()
        for block in INCLUDE.finditer(text):
            target = (page.parent / block.group("path")).resolve()
            for name in START_AFTER.findall(block.group("options")):
                referenced.add((target, name))
                where = page.relative_to(repo.root)
                if not target.is_file():
                    broken.append(f"{where}: no such file {block.group(1)}")
                elif f"docs: begin {name}" not in target.read_text():
                    broken.append(
                        f"{where}: no 'docs: begin {name}' in "
                        f"{block.group('path')}"
                    )
    assert referenced, "no literalinclude markers found; did docs/ move?"
    assert not broken, "\n".join(broken)


def test_unused_markers_are_only_reported(repo):
    """Markers no page includes are a warning, not a failure.

    A marker kept for a page that has not been written yet, or one left
    behind by a page that was restructured, is untidy rather than wrong --
    and failing on it would push someone into deleting the marker instead
    of writing the page.
    """
    referenced = set()
    for page in pages(repo.root):
        for block in INCLUDE.finditer(page.read_text()):
            target = (page.parent / block.group("path")).resolve()
            for name in START_AFTER.findall(block.group("options")):
                referenced.add((target, name))

    unused = []
    for path in sources(repo.root):
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        for name in MARKER.findall(text):
            if (path.resolve(), name) not in referenced:
                unused.append(f"{path.relative_to(repo.root)}: {name}")

    if unused:
        warnings.warn(
            "docs markers no page includes:\n  " + "\n  ".join(sorted(unused)),
            UserWarning,
            stacklevel=1,
        )


def test_the_pinned_jax_version_agrees_everywhere(repo, jax2exec):
    """``versions.env`` is the single source of truth; the others follow it.

    jax and jaxlib must match exactly, and the exporter refuses to run
    against a version it was not tested with, so a pin that drifts here
    turns into a refusal at export time rather than a subtle difference.
    """
    env_text = pathlib.Path(repo.root, "versions.env").read_text()
    versions = dict(
        line.split("=", 1)
        for line in env_text.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    pinned = versions["JAX_VERSION"].strip()
    assert versions["JAXLIB_VERSION"].strip() == pinned

    with pathlib.Path(repo.root, "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    dependencies = pyproject["project"]["dependencies"]
    assert f"jax=={pinned}" in dependencies, dependencies
    assert f"jaxlib=={pinned}" in dependencies, dependencies

    assert jax2exec.SUPPORTED_JAX == pinned
