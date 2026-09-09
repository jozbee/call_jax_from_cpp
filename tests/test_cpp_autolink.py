"""The code-block autolinker: what it links, and what it refuses to.

Loaded from ``docs/_ext`` by path, because the docs extra is not installed
for the fast suite.  Everything under test is a pure function of a fake
index, so no site is needed.  The refusals are what is defended: a wrong
link in a code block sends a lost reader to the wrong page and fails no
build.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "docs" / "_ext" / "cpp_autolink.py"


def _load():
    spec = importlib.util.spec_from_file_location("cpp_autolink", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves annotations through sys.modules, so register first.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


autolink = _load()


def obj(name, objtype, docname, anchor):
    """One ``get_objects()`` tuple; the display name is the last component."""
    return (name, re.split(r"::|\.", name)[-1], objtype, docname, anchor, 1)


#: The shape the C++ domain yields.  Two ``report`` functions on purpose: the
#: ambiguity is what stops ``guard.report(`` from linking.
CPP_OBJECTS = [
    obj("pjrt::Function", "class", "api/cpp/function", "_CPPv4F"),
    obj("pjrt::Function::call", "function", "api/cpp/function", "_CPPv4C"),
    obj("pjrt::rt::harden_malloc", "function", "api/cpp/rt", "_CPPv4HM"),
    obj("pjrt::rt::lock_memory", "function", "api/cpp/rt", "_CPPv4LM"),
    obj("pjrt::rt::Status", "struct", "api/cpp/rt", "_CPPv4S"),
    obj("pjrt::LoadPolicy", "enum", "api/cpp/runtime", "_CPPv4LP"),
    obj(
        "pjrt::LoadPolicy::BinaryOnly",
        "enumerator",
        "api/cpp/runtime",
        "_CPPv4BO",
    ),
    obj(
        "pjrt::LatencyRecorder::record",
        "function",
        "api/cpp/latency",
        "_CPPv4R",
    ),
    obj(
        "pjrt::LatencyRecorder::report",
        "function",
        "api/cpp/latency",
        "_CPPv4LRR",
    ),
    obj(
        "cjfc::AllocReport::report",
        "function",
        "api/cpp/examples",
        "_CPPv4ARR",
    ),
    obj("cjfc::workload::feedback", "function", "api/cpp/examples", "_CPPv4FB"),
    obj("std::vector::size", "function", "api/cpp/x", "_CPPv4SZ"),
    obj("pjrt::Function::x", "functionParam", "api/cpp/function", "_CPPv4X"),
]

PY_OBJECTS = [
    obj("jax2exec.export", "function", "api/python", "jax2exec.export"),
    obj("jax2exec.check", "function", "api/python", "jax2exec.check"),
]


@pytest.fixture(scope="module")
def index():
    return autolink.build_index(CPP_OBJECTS, "::")


@pytest.fixture(scope="module")
def py_index():
    return autolink.build_index(PY_OBJECTS, ".")


def link(target):
    return f"{target.docname}.html#{target.anchor}"


def python_block(code_html):
    return f'<div class="highlight-python notranslate"><div class="highlight"><pre>{code_html}</pre></div>\n</div>'


def names(*parts, sep='<span class="o">::</span>'):
    return sep.join(f'<span class="n">{p}</span>' for p in parts)


def rewrite(html, index, **kwargs):
    kwargs.setdefault("link", link)
    return autolink.rewrite_block(html, index, **kwargs)


def hrefs(html):
    return re.findall(
        r'<a class="cpp-autolink" href="([^"]+)" title="([^"]+)"', html
    )


def test_qualified_exact_links(index):
    out = rewrite(names("pjrt", "rt", "harden_malloc"), index)
    assert hrefs(out) == [
        ("api/cpp/rt.html#_CPPv4HM", "pjrt::rt::harden_malloc")
    ]


def test_unique_suffix_links_alias(index):
    """Inside ``namespace pjrt`` or after a ``using`` the code a page shows is
    not fully qualified; a suffix matching exactly one entity links.
    """
    out = rewrite(names("rt", "lock_memory"), index)
    assert hrefs(out) == [("api/cpp/rt.html#_CPPv4LM", "pjrt::rt::lock_memory")]

    out = rewrite(names("workload", "feedback"), index)
    assert hrefs(out) == [
        ("api/cpp/examples.html#_CPPv4FB", "cjfc::workload::feedback")
    ]


def test_longest_prefix_wins(index):
    """The enumerator, not the enum; the struct, not the namespace."""
    out = rewrite(names("pjrt", "LoadPolicy", "BinaryOnly"), index)
    assert hrefs(out) == [
        ("api/cpp/runtime.html#_CPPv4BO", "pjrt::LoadPolicy::BinaryOnly")
    ]

    # Only the documented prefix is wrapped; the undocumented tail is left.
    out = rewrite(names("pjrt", "rt", "Status", "ok"), index)
    assert hrefs(out) == [("api/cpp/rt.html#_CPPv4S", "pjrt::rt::Status")]
    assert out.endswith('<span class="o">::</span><span class="n">ok</span>')


def test_member_access_links_unique_function(index):
    out = rewrite(
        '<span class="n">f</span><span class="p">.</span>'
        '<span class="n">call</span><span class="p">(</span>',
        index,
    )
    assert hrefs(out) == [
        ("api/cpp/function.html#_CPPv4C", "pjrt::Function::call")
    ]

    out = rewrite(
        '<span class="n">s</span><span class="p">.</span>'
        '<span class="n">rec</span><span class="o">-&gt;</span>'
        '<span class="n">record</span><span class="p">(</span>',
        index,
    )
    assert hrefs(out) == [
        ("api/cpp/latency.html#_CPPv4R", "pjrt::LatencyRecorder::record")
    ]


def test_skip_set_and_ambiguity_do_not_link(index):
    """Three refusals, each for a different reason."""
    # In SKIP_MEMBERS: std-shaped, and a link would usually be a lie.
    out = rewrite(
        '<span class="n">v</span><span class="p">.</span>'
        '<span class="n">size</span><span class="p">(</span>',
        index,
    )
    assert hrefs(out) == []

    # Two documented functions end in "report".
    out = rewrite(
        '<span class="n">guard</span><span class="p">.</span>'
        '<span class="n">report</span><span class="p">(</span>',
        index,
    )
    assert hrefs(out) == []

    # A single unqualified identifier is never a link.
    out = rewrite('<span class="n">call</span>', index)
    assert hrefs(out) == []


def test_strict_prefix_warns_and_ignore_silences(index):
    warnings = []
    rewrite(
        names("pjrt", "nothing", "here"),
        index,
        strict_prefixes=["pjrt", "cjfc"],
        warn=warnings.append,
    )
    assert len(warnings) == 1
    assert "pjrt::nothing::here" in warnings[0]

    # Not a strict prefix: an example's own namespace is not the reference's.
    warnings.clear()
    rewrite(
        names("basic", "foo"),
        index,
        strict_prefixes=["pjrt", "cjfc"],
        warn=warnings.append,
    )
    assert warnings == []

    # Strict, but explicitly ignored.
    warnings.clear()
    rewrite(
        names("pjrt", "detail", "x"),
        index,
        strict_prefixes=["pjrt", "cjfc"],
        ignore=[r"^pjrt::detail::"],
        warn=warnings.append,
    )
    assert warnings == []


def test_html_outside_code_blocks_is_untouched(index, py_index):
    prose = (
        "<p>Call <code>pjrt::rt::harden_malloc</code> before the "
        "<code>Runtime</code>.</p>\n"
    )
    console = (
        '<div class="highlight-console notranslate"><div class="highlight">'
        f"<pre>{names('pjrt', 'rt', 'harden_malloc')}</pre></div>\n</div>"
    )
    body = prose + console
    out = autolink.rewrite_body(body, index, py_index, link=link)
    assert out == body


def test_python_dotted_run_links(index, py_index):
    body = python_block(
        names("jax2exec", "export", sep='<span class="o">.</span>')
    )
    out = autolink.rewrite_body(body, index, py_index, link=link)
    assert hrefs(out) == [
        ("api/python.html#jax2exec.export", "jax2exec.export")
    ]
