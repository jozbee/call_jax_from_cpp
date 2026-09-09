"""Turn qualified C++ names inside code blocks into links to the reference.

After Sphinx has rendered a page, every qualified name and every unique member
call in a C++ or Python highlight block is wrapped in an ``<a>`` pointing at
the entry the reference already has for it.  The targets come out of the
Sphinx domains, so a link cannot go stale, and a name under a *strict prefix*
(``pjrt::``, ``cjfc::``) with no reference entry is a warning, which ``-W``
makes a build failure: a new helper shown on a page must be documented.

Resolution is conservative, because a wrong link is worse than no link.  A
run of components links only when its longest prefix resolves exactly or to
exactly one documented suffix, so ``cjfc::json::object`` links only
``cjfc::json``.  A member call links only when exactly one documented function
in the project ends in that name, and never one of `SKIP_MEMBERS`.  A single
unqualified identifier never links.

Everything above `setup` is a pure function of its arguments and imports no
Sphinx, so it is tested without building a site.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

#: Member names common on standard-library types, where a link would usually
#: be a lie.  ``report`` is absent on purpose: the ambiguity rule already
#: refuses it, and listing it here would hide the day that stops being true.
SKIP_MEMBERS = frozenset(
    {
        "size",
        "data",
        "empty",
        "begin",
        "end",
        "clear",
        "reserve",
        "push_back",
        "name",
        "get",
        "at",
        "front",
        "back",
        "capacity",
        "count",
        "insert",
        "erase",
        "find",
        "reset",
        "release",
    }
)

#: Domain object types that are not entities anyone can link to.
_NOT_ENTITIES = frozenset({"functionParam", "templateParam"})

# What Pygments emits: a qualified name is a run of name spans (``n``, ``nf``,
# ``nc``, ``nn``, ``nl``) separated by an operator span holding ``::``; a
# member access is a ``p`` span holding ``.`` or an ``o`` span holding ``->``.
BLOCK = re.compile(
    r'<div class="highlight-(cpp|c\+\+|python)[^"]*">.*?</pre></div>\s*</div>',
    re.DOTALL,
)
NAME = r'<span class="n[a-z]?">([A-Za-z_]\w*)</span>'
SEP = r'<span class="o">::</span>'
RUN = re.compile(rf"(?<!{SEP}){NAME}(?:{SEP}{NAME})+")
MEMBER = re.compile(
    rf'(?<=<span class="p">\.</span>){NAME}'
    rf'|(?<=<span class="o">-&gt;</span>){NAME}'
)
PYRUN = re.compile(rf'{NAME}(?:<span class="o">\.</span>{NAME})+')

_CPP_SEP_HTML = '<span class="o">::</span>'
_PY_SEP_HTML = '<span class="o">.</span>'
_ONE_NAME = re.compile(NAME)


@dataclass(frozen=True)
class Target:
    """One entity a name can link to."""

    qualified: str
    docname: str
    anchor: str


@dataclass
class Index:
    """The domain's objects, in the three shapes resolution needs."""

    #: Fully qualified name to target.
    by_name: dict[str, Target] = field(default_factory=dict)
    #: Every proper suffix of a qualified name to the targets that end in it.
    by_suffix: dict[str, list[Target]] = field(default_factory=dict)
    #: Last component of a documented function to the functions with it.
    funcs_by_last: dict[str, list[Target]] = field(default_factory=dict)


def build_index(objects: Iterable[Sequence], sep: str = "::") -> Index:
    """Index a domain's ``get_objects()`` output.

    Parameters
    ----------
    objects : iterable
        ``(name, dispname, objtype, docname, anchor, priority)`` tuples, as
        the C++ and Python domains yield them.
    sep : str, optional
        The domain's qualification separator: ``"::"`` or ``"."``.

    Returns
    -------
    Index
        Overloads collapse to one target per qualified name, the first anchor
        winning, because they render as one entry on the page anyway.
    """
    index = Index()
    for entry in objects:
        name, _dispname, objtype, docname, anchor, *_ = entry
        if objtype in _NOT_ENTITIES or not name:
            continue
        if name in index.by_name:
            continue
        target = Target(name, docname, anchor)
        index.by_name[name] = target

        components = name.split(sep)
        for i in range(1, len(components)):
            index.by_suffix.setdefault(sep.join(components[i:]), []).append(
                target
            )
        if objtype == "function":
            index.funcs_by_last.setdefault(components[-1], []).append(target)
    return index


def resolve_qualified(
    index: Index, parts: Sequence[str], sep: str = "::"
) -> tuple[Target | None, int]:
    """Resolve the longest linkable prefix of a qualified name.

    Returns
    -------
    tuple
        ``(target, n)``: the entity, and how many of ``parts`` it consumed.
        ``(None, 0)`` when nothing resolves.  A prefix of one component is
        never tried, so a bare identifier cannot link here.
    """
    for n in range(len(parts), 1, -1):
        candidate = sep.join(parts[:n])
        target = index.by_name.get(candidate)
        if target is not None:
            return target, n
        matches = index.by_suffix.get(candidate)
        if matches is not None and len(matches) == 1:
            return matches[0], n
    return None, 0


def resolve_member(index: Index, name: str) -> Target | None:
    """The one documented function ending in ``name``, if there is exactly one.

    Ambiguity is treated as a refusal rather than as a choice: two functions
    called ``report`` mean the reader cannot be told which one this is.
    """
    if name in SKIP_MEMBERS:
        return None
    matches = index.funcs_by_last.get(name)
    if matches is None or len(matches) != 1:
        return None
    return matches[0]


def _anchor(href: str, qualified: str, inner: str) -> str:
    return (
        f'<a class="cpp-autolink" href="{href}" title="{qualified}">{inner}</a>'
    )


def _names_in(pieces: Sequence[str]) -> list[str] | None:
    parts = []
    for piece in pieces:
        found = _ONE_NAME.fullmatch(piece)
        if found is None:
            return None
        parts.append(found.group(1))
    return parts


def _rewrite_run(
    match: re.Match,
    index: Index,
    *,
    sep: str,
    sep_html: str,
    link: Callable[[Target], str | None],
    strict_prefixes: Sequence[str],
    ignore: Sequence[re.Pattern],
    warn: Callable[[str], None] | None,
) -> str:
    text = match.group(0)
    pieces = text.split(sep_html)
    parts = _names_in(pieces)
    if parts is None:
        return text

    target, n = resolve_qualified(index, parts, sep)
    if target is None:
        qualified = sep.join(parts)
        if (
            warn is not None
            and parts[0] in strict_prefixes
            and not any(pattern.search(qualified) for pattern in ignore)
        ):
            warn(
                f"{qualified} is spelled in a code block but has no entry in "
                "the reference; add a directive for it to the reference page, "
                "or an expression to cpp_autolink_ignore"
            )
        return text

    href = link(target)
    if href is None:
        return text
    head = _anchor(href, target.qualified, sep_html.join(pieces[:n]))
    tail = pieces[n:]
    return head if not tail else head + sep_html + sep_html.join(tail)


def _rewrite_member(
    match: re.Match, index: Index, link: Callable[[Target], str | None]
) -> str:
    name = match.group(1) or match.group(2)
    target = resolve_member(index, name)
    if target is None:
        return match.group(0)
    href = link(target)
    if href is None:
        return match.group(0)
    return _anchor(href, target.qualified, match.group(0))


def rewrite_block(
    html: str,
    index: Index,
    *,
    link: Callable[[Target], str | None],
    strict_prefixes: Sequence[str] = (),
    ignore: Sequence = (),
    warn: Callable[[str], None] | None = None,
) -> str:
    """Link the qualified names and member calls in one C++ highlight block."""
    patterns = [
        p if isinstance(p, re.Pattern) else re.compile(p) for p in ignore
    ]
    html = RUN.sub(
        lambda m: _rewrite_run(
            m,
            index,
            sep="::",
            sep_html=_CPP_SEP_HTML,
            link=link,
            strict_prefixes=strict_prefixes,
            ignore=patterns,
            warn=warn,
        ),
        html,
    )
    return MEMBER.sub(lambda m: _rewrite_member(m, index, link), html)


def _rewrite_python_block(
    html: str, index: Index, *, link: Callable[[Target], str | None]
) -> str:
    """Link dotted runs in a Python highlight block.

    Exact matches only, and never strict: a Python block on this site shows
    JAX and NumPy as often as it shows ``jax2exec``, and warning about every
    name that belongs to another project would be noise.
    """
    return PYRUN.sub(
        lambda m: _rewrite_run(
            m,
            index,
            sep=".",
            sep_html=_PY_SEP_HTML,
            link=link,
            strict_prefixes=(),
            ignore=(),
            warn=None,
        ),
        html,
    )


def rewrite_body(
    body: str,
    cpp_index: Index,
    py_index: Index,
    *,
    link: Callable[[Target], str | None],
    strict_prefixes: Sequence[str] = (),
    ignore: Sequence = (),
    warn: Callable[[str], None] | None = None,
) -> str:
    """Rewrite every C++ and Python highlight block in a rendered page.

    Everything outside such a block -- prose, tables, a ``console`` or ``text``
    block -- is returned byte for byte as it came in.
    """

    def one(match: re.Match) -> str:
        language = match.group(1)
        if language == "python":
            return _rewrite_python_block(match.group(0), py_index, link=link)
        return rewrite_block(
            match.group(0),
            cpp_index,
            link=link,
            strict_prefixes=strict_prefixes,
            ignore=ignore,
            warn=warn,
        )

    return BLOCK.sub(one, body)


def setup(app):
    """Register the ``html-page-context`` handler that does the rewriting."""
    from sphinx.errors import SphinxError
    from sphinx.util import logging as sphinx_logging

    logger = sphinx_logging.getLogger(__name__)

    app.add_config_value(
        "cpp_autolink_strict_prefixes", ["pjrt", "cjfc"], "html"
    )
    app.add_config_value("cpp_autolink_ignore", [], "html")

    # Built once per environment: the domains are stable once reading is done,
    # and keying on the env keeps sphinx-autobuild from serving a stale index.
    cache: dict[int, tuple[Index, Index]] = {}

    def indexes(env) -> tuple[Index, Index]:
        key = id(env)
        if key not in cache:
            cache.clear()
            cache[key] = (
                build_index(env.get_domain("cpp").get_objects(), "::"),
                build_index(env.get_domain("py").get_objects(), "."),
            )
        return cache[key]

    def on_page_context(app, pagename, templatename, context, doctree):
        body = context.get("body")
        if not body:
            return
        cpp_index, py_index = indexes(app.env)

        def link(target: Target) -> str | None:
            try:
                uri = app.builder.get_relative_uri(pagename, target.docname)
            except SphinxError:
                return None
            return f"{uri}#{target.anchor}"

        def warn(message: str) -> None:
            logger.warning(message, location=pagename, type="cpp_autolink")

        context["body"] = rewrite_body(
            body,
            cpp_index,
            py_index,
            link=link,
            strict_prefixes=app.config.cpp_autolink_strict_prefixes,
            ignore=app.config.cpp_autolink_ignore,
            warn=warn,
        )

    app.connect("html-page-context", on_page_context)
    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
