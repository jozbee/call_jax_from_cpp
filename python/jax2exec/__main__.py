"""Run the package as a command: ``python -m jax2exec check <base>``."""

from __future__ import annotations

import sys

from ._cli import main

if __name__ == "__main__":
    sys.exit(main())
