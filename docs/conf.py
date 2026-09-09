"""Sphinx configuration for the call_jax_from_cpp documentation.

Two things a stock ``conf.py`` does not do: Doxygen runs at import time, so
the XML Breathe reads always matches the headers on disk (``sphinx-autobuild``
re-imports this module, which is what regenerates it); and every pinned
version is read out of ``versions.env`` and injected as a MyST substitution,
so no page hand-types one. The build runs with ``-W``: a warning is a failure.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent

# Extensions written for this site only, not installed as packages.
sys.path.insert(0, str(DOCS_DIR / "_ext"))

# -- Project ----------------------------------------------------------------

project = "call_jax_from_cpp"
author = "Brent Koogler"

# The release is whatever the installed jax2exec says it is, so it cannot
# drift from the package.
try:
    release = _package_version("jax2exec")
except PackageNotFoundError as exc:  # pragma: no cover - environment error
    raise RuntimeError(
        "the jax2exec distribution is not installed, so the docs cannot "
        "determine the release. Run `uv sync --extra docs` from the repository "
        "root and build with `uv run`."
    ) from exc
version = release


# -- versions.env -----------------------------------------------------------


def _read_versions_env(path: Path) -> dict[str, str]:
    """Parse ``versions.env`` into ``{lower_case_key: value}``.

    Flat ``KEY=value`` lines with ``#`` comments; anything else is skipped.
    """
    if not path.is_file():
        raise RuntimeError(
            f"{path} is missing. It is the single source of truth for every "
            "pinned version, and the docs refuse to hand-type one."
        )
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, sep, value = stripped.partition("=")
        if not sep:
            continue
        values[key.strip().lower()] = value.strip()
    return values


_versions = _read_versions_env(REPO_ROOT / "versions.env")


# -- Doxygen ----------------------------------------------------------------

DOXYGEN_XML = DOCS_DIR / "_build" / "doxygen" / "xml"


def _run_doxygen() -> None:
    """Regenerate the Doxygen XML that Breathe reads.

    Called at module level rather than from ``builder-inited`` so the XML
    exists before any extension is set up.
    """
    doxygen = shutil.which("doxygen")
    if doxygen is None:
        raise RuntimeError(
            "doxygen is not on PATH, and the C++ API pages are generated from "
            "its XML.\n"
            "  Debian/Ubuntu:  sudo apt-get install doxygen\n"
            "  Arch:           sudo pacman -S doxygen\n"
            "  macOS:          brew install doxygen\n"
            "  or build in the container, which already has it:\n"
            "    docker compose -f docker/compose.yml run --rm dev make docs"
        )
    # Doxygen creates OUTPUT_DIRECTORY but not its parents.
    DOXYGEN_XML.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([doxygen, "Doxyfile"], cwd=DOCS_DIR, check=False)
    if result.returncode != 0:
        # Doxygen has already printed the file and line; a CalledProcessError
        # traceback would bury it under a Sphinx stack trace.
        raise RuntimeError(
            f"doxygen exited {result.returncode}; the C++ headers above have a "
            "documentation defect. Fix the header -- an undocumented parameter "
            "on a documented function is the usual cause -- rather than "
            "relaxing WARN_AS_ERROR in docs/Doxyfile."
        )


_run_doxygen()

# -- Extensions -------------------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",  # writes .nojekyll, without which _static 404s
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
    "breathe",
    "cpp_autolink",  # docs/_ext: links qualified names inside code blocks
]

# A pjrt::/cjfc:: name in a code block with no reference entry fails the
# build, so a new helper shown on a page must be documented.
cpp_autolink_strict_prefixes = ["pjrt", "cjfc"]
cpp_autolink_ignore = [
    r"^pjrt::detail::",
    r"^cjfc::detail::",
    r"^cjfc::workload::detail::",
]

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

# -- MyST -------------------------------------------------------------------

myst_enable_extensions = [
    "attrs_block",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "dollarmath",
]

# Anchors down to h3, so a guide can link to a subsection of another guide.
myst_heading_anchors = 3

# {{ jax_version }} in any page. MyST does not substitute inside inline code,
# so every value also has a <key>_code variant carrying its own backticks;
# tests/test_docs_sync.py rejects a backtick-wrapped substitution.
myst_substitutions = dict(_versions)
myst_substitutions.update({f"{k}_code": f"`{v}`" for k, v in _versions.items()})

# CHANGELOG.md and CONTRIBUTING.md are included from the repository root with a
# standalone document's heading levels; the "non-consecutive header level"
# warning is structural.
suppress_warnings = ["myst.header"]

# -- Python API -------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Stub pages are hand-written; nothing is generated into the source tree.
autosummary_generate = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# The return type is already in the signature.
napoleon_use_rtype = False

# Under nitpicky every docstring type must resolve: without preprocessing,
# "int, optional" looks up "optional" as a class. The aliases map the short
# spellings the docstrings use to the names the domain indexes.
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Any": "typing.Any",
    "Iterable": "collections.abc.Iterable",
    "Mapping": "collections.abc.Mapping",
    "Path": "pathlib.Path",
    "Sequence": "collections.abc.Sequence",
    "callable": "collections.abc.Callable",
}

# -- C++ API ----------------------------------------------------------------

breathe_projects = {"pjrt_exec": str(DOXYGEN_XML)}
breathe_default_project = "pjrt_exec"
# Empty on purpose: every page names the members it documents, so a dropped
# member is a build failure rather than a quiet disappearance.
breathe_default_members = ()
breathe_show_include = False

# Trims "pjrt::" from the C++ index so entries sort under their own names.
cpp_index_common_prefix = ["pjrt::rt::", "pjrt::", "cjfc::workload::", "cjfc::"]

# -- Cross-project references -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
# An unreachable inventory is a warning, hence a failed build under -W. If
# someone else's outage starts failing CI, vendor the objects.inv files and
# point the second element of each tuple at the local copy.
intersphinx_timeout = 10

# The Linux Foundation's real-time wiki answers 403 to anything that is not a
# browser.
linkcheck_ignore = [
    r"https://wiki\.linuxfoundation\.org/realtime/.*",
]

# A dead cross-reference fails the build. The ignore list is what Breathe
# emits references for and this site does not document: the PJRT C API,
# libstdc++, nlohmann, and the namespaces themselves, which Breathe names in
# declarations but never declares as targets.
nitpicky = True
nitpick_ignore_regex = [
    ("cpp:identifier", r"^PJRT_.*"),
    ("cpp:identifier", r"^std::.*"),
    ("cpp:type", r"^PJRT_.*"),
    ("cpp:type", r"^std::.*"),
    ("cpp:identifier", r"^(pjrt|pjrt::rt|pjrt::detail|cjfc|cjfc::workload)$"),
    ("cpp:identifier", r"^nlohmann(::.*)?$"),
    ("cpp:type", r"^nlohmann(::.*)?$"),
]

# -- HTML -------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "call_jax_from_cpp"
html_baseurl = "https://jozbee.github.io/call_jax_from_cpp/"
html_static_path = ["_static"]
html_css_files = ["style.css"]
# One favicon: the SVG carries a media query. Two logos: the theme toggle sets
# data-theme rather than the OS preference, so a media query cannot follow it.
html_favicon = "_static/favicon.svg"
# Unlicense: a copyright line in the footer would be a false claim.
html_show_copyright = False

# Only keys the theme knows: an unknown one warns, and -W makes that fatal.
html_theme_options = {
    "repository_url": "https://github.com/jozbee/call_jax_from_cpp",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_navbar_depth": 1,
    "navigation_with_keys": False,
    "home_page_in_toc": False,
    "article_header_start": [
        "toggle-primary-sidebar.html",
        "breadcrumbs.html",
    ],
    "logo": {
        "image_light": "_static/logo-light.svg",
        "image_dark": "_static/logo-dark.svg",
        "alt_text": "call_jax_from_cpp",
    },
}
