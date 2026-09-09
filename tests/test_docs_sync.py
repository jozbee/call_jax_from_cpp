"""Cheap checks that the documentation still describes the code.

Nothing fails when a dtype is added to the exporter and not to the table,
when a ``throw`` grows a message no page explains, or when a marker is
renamed out from under a ``literalinclude``; the docs simply become wrong.
None of this checks that the prose is *good*, only that the nouns in it
still exist.  Where a rule cannot be enforced without guessing at intent it
warns instead of failing.
"""

from __future__ import annotations

import pathlib
import re
import tomllib
import warnings

import pytest

#: Generated, vendored, built, or not the subject.  ``tests`` because a scan
#: of it would find this file's own regexes and report them as markers.
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

#: Below this many characters a message prefix is a fragment like ``"output "``
#: that would match half the page by accident.
MIN_DISTINCTIVE = 12

#: A fenced block and everything in it.  Code samples quote real numbers, and
#: none of those is the page making a claim of its own.
FENCE = re.compile(
    r"^([ \t]*)(```+|~~~+).*?^[ \t]*\2[ \t]*$", re.MULTILINE | re.DOTALL
)

#: The two homes of a measured figure.
FIGURE_HOMES = ("docs/developer/", "docs/benchmarks.md")

#: A duration: ``2027 us``, ``1.5 ms``, and the micro-sign spelling of ``us``.
DURATION = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?\s*(?:µs|us|ms)\b")

#: A ratio: ``2.4x``, ``4x``.  The ``\b`` excludes a size like ``1024x768``.
RATIO = re.compile(r"(?<![\dx.])\d+(?:\.\d+)?x\b")

#: How far below the H1 the assumes line may sit: room for a label or a
#: directive above it, not enough to drift out of the first screenful.
ASSUMES_WINDOW = 6

#: The opening line of the ``{glossary}`` directive, which picks its block out
#: of :data:`FENCE`'s matches.  Inside it a term is flush left and its
#: definition indented.
GLOSSARY_DIRECTIVE = "{glossary}"

#: A ``{term}`` reference in either spelling: ``{term}`arena``` or
#: ``{term}`arenas <arena>```.
TERM_ROLE = re.compile(r"\{term\}`(?P<text>[^`]*)`")

#: The ``title <target>`` half of a reference, when it has one.
TERM_TARGET = re.compile(r"^.*?<(?P<target>[^<>]*)>$", re.DOTALL)

#: Matches that are not measurements of this project, and why.  Keyed by text
#: rather than line number, because a line number goes stale when a paragraph
#: moves.  Add an entry only with the reason it is not a result.
NOT_A_MEASUREMENT = {
    ">2x": "the threshold in the rule, not a result",
}


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
    """The first string literal of each ``throw`` in the library: the part of
    the message a reader can grep for, before a path or an index is appended.
    Anything shorter than :data:`MIN_DISTINCTIVE` is a fragment, not a
    message, and is dropped.
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
    """A type added to the exporter and not to the table is a type nobody
    knows they may use.
    """
    guide = pathlib.Path(repo.root, "docs/guides/exporting.md").read_text()
    missing = [name for name in jax2exec.SUPPORTED_DTYPES if name not in guide]
    assert not missing, f"not in docs/guides/exporting.md: {missing}"


def test_every_throw_message_has_a_row_in_the_debugging_guide(repo):
    """Every distinct failure the loader can report has a row in the table
    that is the first thing anyone hits when a load fails.
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


def included_markers(root):
    """Each ``literalinclude``: page, path as written, target, marker."""
    for page in pages(root):
        for block in INCLUDE.finditer(page.read_text()):
            spelled = block.group("path")
            target = (page.parent / spelled).resolve()
            for name in START_AFTER.findall(block.group("options")):
                yield page, spelled, target, name


def test_every_included_marker_exists(repo):
    """Each ``literalinclude`` resolves to a file and finds its marker.  A
    renamed marker only warns in Sphinx and renders an empty block.
    """
    broken = []
    found = 0
    for page, spelled, target, name in included_markers(repo.root):
        found += 1
        where = page.relative_to(repo.root)
        if not target.is_file():
            broken.append(f"{where}: no such file {spelled}")
        elif f"docs: begin {name}" not in target.read_text():
            broken.append(f"{where}: no 'docs: begin {name}' in {spelled}")
    assert found, "no literalinclude markers found; did docs/ move?"
    assert not broken, "\n".join(broken)


def test_unused_markers_are_only_reported(repo):
    """A marker no page includes is untidy, not wrong; failing on it would
    push someone into deleting the marker instead of writing the page.
    """
    referenced = {
        (target, name)
        for _page, _spelled, target, name in included_markers(repo.root)
    }

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


def prose(text):
    """@p text with its fenced blocks blanked out, line numbers preserved."""
    return FENCE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def test_measured_figures_stay_under_developer_and_benchmarks(repo):
    """A latency or a ratio outside its home is a figure without a host.

    Every other page states the *shape* of a result and links to the page
    holding the figure, because a number reprinted away from the machine it
    was measured on cannot be checked, compared or refuted later.  Asserted,
    not warned: a warning about a figure is itself the thing that gets copied
    forward.  Prose only: fenced blocks are blanked first.  The two
    ``{include}`` pages are a fence each, so what those files say is governed
    where they live.
    """
    hits = []
    for page in pages(repo.root):
        where = page.relative_to(repo.root).as_posix()
        if where.startswith(FIGURE_HOMES):
            continue
        lines = prose(page.read_text()).splitlines()
        for number, line in enumerate(lines, 1):
            for pattern in (DURATION, RATIO):
                for match in pattern.finditer(line):
                    found = match.group(0)
                    # The match itself must be inside the excused text, so a
                    # real figure sharing a line with one is still reported.
                    excused = any(
                        text in line and found in text
                        for text in NOT_A_MEASUREMENT
                    )
                    if not excused:
                        hits.append(f"{where}:{number}: {found!r}")

    assert not hits, (
        "measured figures belong under docs/developer/ or on "
        "docs/benchmarks.md, beside the host they were measured on.  State "
        "the shape of the result here and link to the page that holds the "
        "number, or add the match to NOT_A_MEASUREMENT with the reason it "
        "is not a result:\n  " + "\n  ".join(hits)
    )


def entry_pages(root):
    """Every guide and numbered example page: what a reader lands on."""
    docs = pathlib.Path(root, "docs")
    return sorted(docs.glob("guides/*.md")) + sorted(
        docs.glob("examples/0*.md")
    )


def test_every_guide_and_example_page_states_what_it_assumes(repo):
    """A reader arrives from a search engine, not from the page before, so
    every guide and example opens with the italic *Assumes* line.  A
    convention rather than a mechanism, which is why it is asserted.
    """
    missing = []
    for page in entry_pages(repo.root):
        lines = page.read_text().splitlines()
        heading = next(
            (n for n, line in enumerate(lines) if line.startswith("# ")), None
        )
        if heading is None:
            missing.append(f"{page.relative_to(repo.root)}: no H1")
            continue
        window = lines[heading + 1 : heading + 1 + ASSUMES_WINDOW]
        if not any(line.startswith("*Assumes") for line in window):
            missing.append(f"{page.relative_to(repo.root)}")

    assert not missing, (
        "every guide and example page opens with an italic line starting "
        f"'*Assumes' within {ASSUMES_WINDOW} lines of its H1, saying what "
        "the page takes for granted and where the rest is.  Missing from:"
        "\n  " + "\n  ".join(missing)
    )


def test_glossary_terms_are_used(repo):
    """A term nothing links is a definition nobody reaches: either a link
    never made or an entry that outlived its page.  Reported, not failed,
    since a term added ahead of its page is legitimate.  Case-insensitive,
    as Sphinx's lookup is, and the ``title <target>`` form is compared on
    its target.
    """
    glossary = pathlib.Path(repo.root, "docs/background/glossary.md")
    blocks = [
        block.group(0)
        for block in FENCE.finditer(glossary.read_text())
        if GLOSSARY_DIRECTIVE in block.group(0).splitlines()[0]
    ]
    assert blocks, f"no {GLOSSARY_DIRECTIVE} directive in {glossary.name}"

    terms = [
        line.strip()
        for block in blocks
        for line in block.splitlines()[1:-1]
        if line.strip() and not line[:1].isspace()
    ]
    assert terms, "the glossary defines no terms; did the directive change?"

    referenced = set()
    for page in pages(repo.root):
        for reference in TERM_ROLE.finditer(page.read_text()):
            text = reference.group("text").strip()
            explicit = TERM_TARGET.match(text)
            if explicit:
                text = explicit.group("target").strip()
            referenced.add(text.lower())

    unused = [term for term in terms if term.lower() not in referenced]
    if unused:
        warnings.warn(
            "glossary terms no page references with {term}:\n  "
            + "\n  ".join(unused),
            UserWarning,
            stacklevel=1,
        )


def test_no_substitution_is_wrapped_in_inline_code(repo):
    """MyST does not substitute inside inline code, and the result is valid
    markdown, so the build stays green while the page tells a reader the
    branch is ``{{ xla_fork_branch }}``.  ``docs/conf.py`` publishes a
    ``<key>_code`` variant of every value for this.
    """
    wrapped = re.compile(r"`\{\{\s*[a-z_]+\s*\}\}`")
    offenders = [
        f"{path.relative_to(repo.root)}:{number}: {line.strip()}"
        for path in pages(repo.root)
        for number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        )
        if wrapped.search(line)
    ]
    assert not offenders, (
        "substitutions inside inline code render literally; use the _code "
        "variant, e.g. {{ xla_commit_code }}:\n  " + "\n  ".join(offenders)
    )


def test_the_pinned_jax_version_agrees_everywhere(repo, jax2exec):
    """``versions.env`` is the single source of truth; the others follow it.
    The exporter refuses a version it was not tested with, so a pin that
    drifts turns into a refusal at export time.
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
