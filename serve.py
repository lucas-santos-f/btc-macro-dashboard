"""Local server for the dashboard.

    python serve.py                       # port 8765, opens the browser
    python serve.py --port 9000 --no-browser

Deliberately boring: binds to 127.0.0.1 only (not reachable from the LAN or
the internet regardless of firewall), serves static files from web/ and
nothing else. No POST route, no webhook, nothing a third party can trigger.
"""

import argparse
import http.server
import sys
import webbrowser
from functools import partial
from pathlib import Path

HOST = "127.0.0.1"  # never 0.0.0.0
WEB = Path(__file__).resolve().parent / "web"


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        # everything is local, plotly is vendored; block any external resource
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'",
        )
        super().end_headers()

    def log_request(self, code="-", size="-"):
        if not str(code).startswith("2"):  # only log what's actually wrong
            sys.stderr.write(f"  {code}  {self.requestline}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not (WEB / "data.json").exists():
        print("web/data.json not found, run:  python fetch_data.py", file=sys.stderr)
        return 1

    try:
        srv = http.server.ThreadingHTTPServer((HOST, args.port), partial(Handler, directory=str(WEB)))
    except OSError as e:
        # EADDRINUSE is 48 on macOS, 98 on Linux, 10048 on Windows
        if e.errno in (48, 98, 10048):
            print(f"Port {args.port} is taken, try:  python serve.py --port {args.port + 1}", file=sys.stderr)
            return 1
        raise

    url = f"http://{HOST}:{args.port}/"
    print(f"\n  dashboard at {url}  (this machine only)")
    print("  Ctrl+C to stop\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  bye")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
