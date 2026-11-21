"""A stand-in child engine for the lifecycle tests: answers /health (and /v1/models) with 200 on --port after
--delay seconds. No model, no GPU."""

import argparse
import http.server
import time


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path in ("/health", "/v1/models"):
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_):
        pass


def serve(port: int) -> http.server.ThreadingHTTPServer:
    return http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    time.sleep(args.delay)
    print("fake engine listening", flush=True)
    serve(args.port).serve_forever()
