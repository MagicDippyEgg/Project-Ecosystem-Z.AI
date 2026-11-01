"""PyInstaller entry point.

A thin launcher so PyInstaller treats the app as a package import
rather than a standalone script (which breaks ``from .gui...`` relative
imports inside ``ecosystem.app``).
"""

from ecosystem.app import main

if __name__ == "__main__":
    raise SystemExit(main())
