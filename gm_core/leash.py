# Vendored from Imagination Studio by keys+Wasserman (drowzeys/Imagination-Studio-by-keys-Wasserman),
# app/bolt_leash.py at 912374e1b6111fdab5d55947cc695c9c303f83b7. Apache-2.0: LICENSES/Imagination-Studio-LICENSE-Apache-2.0 and LICENSES/Imagination-Studio-NOTICE.
# Unchanged except this header.
"""The leash of a Studio-managed model server (app/bolt.py): runs the server as its child and ends it when the app's
pipe closes, so a server never outlives the app that started it, as Drama 3's and MiniMax Music 3's engines end at the
end of their input.

    python3 -I bolt_leash.py -- server args...

The server's output goes to this process's stderr (the app reads it). When stdin reaches its end (the app closed it,
or the app died) or this process is asked to stop (SIGTERM, SIGINT), the server gets SIGTERM, then SIGKILL after
LEASH_GRACE_S seconds. The exit status is the server's.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading

GRACE_S = float(os.environ.get("LEASH_GRACE_S", 20))


def main(argv: list[str]) -> int:
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if not argv:
        print("usage: bolt_leash.py -- server args...", file=sys.stderr)
        return 2
    child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=sys.stderr, stderr=sys.stderr)
    ending = threading.Event()

    def end(*_) -> None:
        if ending.is_set():
            return
        ending.set()
        if child.poll() is None:
            try:
                child.terminate()
                child.wait(timeout=GRACE_S)
            except subprocess.TimeoutExpired:
                child.kill()
            except OSError:
                pass

    def watch_stdin() -> None:
        try:
            while sys.stdin.buffer.read(4096):
                pass
        except (OSError, ValueError):
            pass
        end()

    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=end, daemon=True).start())
    signal.signal(signal.SIGINT, lambda *_: threading.Thread(target=end, daemon=True).start())
    threading.Thread(target=watch_stdin, daemon=True).start()
    code = child.wait()
    ending.set()
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
