"""EUNOCURA backend package.

The first thing this package does is make stdout/stderr UTF-8 safe. Several services
log short status lines containing non-ASCII characters; on a Windows console using
cp1252 those prints raise UnicodeEncodeError *at import time*, which would take the
whole API down before it ever starts. Reconfiguring here fixes every import path at
once (server, scripts, tests) without touching the log messages themselves.
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        # Streams can be replaced by pytest/captured buffers; not worth failing over.
        pass
