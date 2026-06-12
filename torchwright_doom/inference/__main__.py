"""Entrypoint for ``python -m torchwright_doom.inference``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
