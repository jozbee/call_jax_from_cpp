"""Every place a JAX version is written down has to say the same thing.

``versions.env`` is the single source of truth, but ``pyproject.toml`` has to
repeat the pin, the environment has to contain that release, and the
exporter records what it was tested against.  Drift between them is what a
half-finished JAX bump looks like, and a ``.binpb`` is not portable across
one.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: ``name==version``, the only requirement form these two pins may take.
_PIN = re.compile(r"^\s*([A-Za-z0-9._-]+)\s*==\s*([^\s;#]+)")


def versions_env() -> dict[str, str]:
    """Parse ``versions.env`` into a dict, ignoring comments and blanks."""
    values: dict[str, str] = {}
    text = (REPO_ROOT / "versions.env").read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def pyproject() -> dict:
    """Parse ``pyproject.toml``."""
    return tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )


def exact_pins() -> dict[str, str]:
    """The ``==`` pins from ``[project].dependencies``."""
    pins = {}
    for requirement in pyproject()["project"]["dependencies"]:
        match = _PIN.match(requirement)
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


@pytest.mark.parametrize(
    ("package", "key"),
    [("jax", "JAX_VERSION"), ("jaxlib", "JAXLIB_VERSION")],
)
def test_pyproject_pins_match_versions_env(package, key):
    """The packaging pin and the build's source of truth agree."""
    environment = versions_env()
    assert key in environment, f"versions.env has no {key}"
    pins = exact_pins()
    assert package in pins, (
        f"pyproject.toml does not pin {package} exactly; jax and jaxlib must "
        "match each other and versions.env to the patch release"
    )
    assert pins[package] == environment[key]


def test_jax_and_jaxlib_are_pinned_to_the_same_release():
    """jaxlib must match jax exactly; the CPU plugin is built for that pair."""
    environment = versions_env()
    assert environment["JAX_VERSION"] == environment["JAXLIB_VERSION"]


@pytest.mark.parametrize(
    ("module_name", "key"),
    [("jax", "JAX_VERSION"), ("jaxlib", "JAXLIB_VERSION")],
)
def test_the_installed_release_is_the_pinned_one(module_name, key):
    """An environment a release behind produces artifacts the published
    plugin cannot load, and the failure arrives from the C++ side with
    nothing in it pointing back to here."""
    module = pytest.importorskip(
        module_name, reason=f"{module_name} is not installed; run `uv sync`"
    )
    assert module.__version__ == versions_env()[key]


def test_the_exporter_agrees_about_the_supported_release():
    """``SUPPORTED_JAX`` is what the exporter warns against, so it moves with
    the pin rather than after it."""
    jax2exec = pytest.importorskip("jax2exec")
    pytest.importorskip("jax", reason="SUPPORTED_JAX imports the exporter")
    assert jax2exec.SUPPORTED_JAX == versions_env()["JAX_VERSION"]


def test_the_tool_version_in_every_sidecar_is_the_package_version():
    """``generator.version`` traces an artifact back to the exporter that
    produced it, which only works if it is the version that shipped."""
    jax2exec = pytest.importorskip("jax2exec")
    assert jax2exec.__version__ == pyproject()["project"]["version"]
