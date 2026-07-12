"""Standalone canonical sources copied byte-for-byte into published bundles.

``pretty_text.py`` ships as the bundle's ``tools/pretty_text.py`` and
``txt_to_png.py`` as ``tools/txt_to_png.py``. Both are importable in-tree —
the formatter wrapper and staged bundle validation exercise the exact code a
downloader receives — but they must import nothing beyond the Python
standard library, because a downloaded bundle cannot import project code.
Exact-copy and parity tests guard the duplication.
"""
