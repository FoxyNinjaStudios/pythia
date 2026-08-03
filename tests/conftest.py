"""Pytest bootstrap: put the ./src source root on sys.path.

Only main.py and server.py live at the repo root; every other module (paths,
sam_wrapper, the sam3d_objects package, …) lives under ./src. Add it to the
import path so tests can import those modules directly.
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
