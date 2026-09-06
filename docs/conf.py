"""Sphinx configuration for the call_jax_from_cpp documentation.

Two things happen here that a stock ``conf.py`` does not do, and both are
deliberate:

* Doxygen runs at import time, below, so the XML Breathe reads always matches
  the headers on disk.  ``sphinx-autobuild`` re-imports this module on every
  rebuild, which is what makes ``--watch ../include`` regenerate the XML.
* Every version this project pins is read out of ``versions.env`` and injected
  as a MyST substitution.  No page hand-types a version number; a bump edits
  one file.

The build runs with ``-W``, so a warning is a failure.  That is the point: an
orphan page, a broken cross-reference, or an unknown theme option should stop
the site from shipping rather than quietly degrade it.
"""

from __future__ import annotations

import shutil
import subprocess
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

DOCS_DIR = Path(__file__).parent.resolve()
REPO_ROOT = DOCS_DIR.parent

# -- Project ----------------------------------------------------------------

project = "call_jax_from_cpp"
author = "Brent Koogler"

# The release is whatever the installed jax2exec says it is, so the docs cannot
# drift from the package the way a hard-coded string does.  A missing
# distribution means the docs are being built against a tree that was never
# installed, which is worth stopping for.
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

    The file is a flat ``KEY=value`` list with ``#`` comments -- deliberately
    simple, because the Makefile, CMake and CI all source it too.  Anything
    that is not a ``KEY=value`` line is skipped rather than guessed at.
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

    This runs at module level rather than from a ``builder-inited`` handler so
    that the XML exists before any extension is set up, and so that
    ``sphinx-autobuild`` -- which re-imports this file on each rebuild --
    picks up header edits.
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
    # Doxygen creates OUTPUT_DIRECTORY but not its parents, and _build is
    # usually absent on a clean checkout.
    DOXYGEN_XML.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([doxygen, "Doxyfile"], cwd=DOCS_DIR, check=False)
    if result.returncode != 0:
        # The Doxyfile sets WARN_AS_ERROR = FAIL_ON_WARNINGS, so this is
        # usually a documentation defect in a header rather than a broken
        # Doxyfile: an undocumented parameter, an unknown command, a reference
        # that does not resolve. Doxygen has already printed the file and line
        # above; re-raising with a bare CalledProcessError traceback would bury
        # it under a Sphinx stack trace.
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

# Every pinned version, spelled once. Use as {{ jax_version }} in any page.
myst_substitutions = dict(_versions)

# CHANGELOG.md and CONTRIBUTING.md are included from the repository root, where
# their heading levels are the levels of a standalone document rather than of a
# page inside this tree. The resulting "non-consecutive header level" warnings
# are structural and expected; the alternative is to duplicate both files.
suppress_warnings = ["myst.header"]

# -- Python API -------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"
# Stub pages are written by hand and checked in; nothing is generated into the
# source tree at build time.
autosummary_generate = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# The return type is already in the signature and in the typehints; repeating
# it as a ":rtype:" line adds a row that says the same thing twice.
napoleon_use_rtype = False

# -- C++ API ----------------------------------------------------------------

breathe_projects = {"pjrt_exec": str(DOXYGEN_XML)}
breathe_default_project = "pjrt_exec"
# Empty on purpose: every page names the members it documents. A class that
# grows a member does not silently grow the page, and a member that is dropped
# is a build failure instead of a quiet disappearance.
breathe_default_members = ()
breathe_show_include = False

# Trims "pjrt::" from the C++ index so entries sort under their own names.
cpp_index_common_prefix = ["pjrt::rt::", "pjrt::"]

# -- Cross-project references -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
# An inventory that cannot be fetched is a warning, and -W turns that into a
# failed build -- so a CI run can fail on someone else's outage. The mitigation
# if that starts happening is to vendor the objects.inv files under
# docs/_inventory/ and point the second element of each tuple at the local
# copy; the fetch then never leaves the machine.
intersphinx_timeout = 10

# The Linux Foundation's real-time wiki answers 403 to anything that is not a
# browser, so linkcheck would fail on a page that is there. Everything else
# is checked.
linkcheck_ignore = [
    r"https://wiki\.linuxfoundation\.org/realtime/.*",
]

# nitpicky mode is off until the C++ pages are written, because Breathe emits a
# cross-reference for every type it sees -- including PJRT C API structs and
# libstdc++ types that have no target here. The ignore list below is ready for
# the day it is turned on.
nitpicky = False
nitpick_ignore_regex = [
    ("cpp:identifier", r"^PJRT_.*"),
    ("cpp:identifier", r"^std::.*"),
    ("cpp:type", r"^PJRT_.*"),
    ("cpp:type", r"^std::.*"),
]

# -- HTML -------------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = "call_jax_from_cpp"
html_baseurl = "https://jozbee.github.io/call_jax_from_cpp/"
html_static_path = ["_static"]
html_css_files = ["style.css"]
# The favicon follows the browser's colour scheme on its own (a media query
# inside the SVG); the sidebar logo cannot, because the site's theme toggle
# sets data-theme rather than the OS preference, so it is two files.
html_favicon = "_static/favicon.svg"
# The project is released under the Unlicense -- it is in the public domain --
# so a copyright line in the footer would be a false claim.
html_show_copyright = False

# Only keys the theme actually knows: an unknown one warns, and -W makes that
# fatal.
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
