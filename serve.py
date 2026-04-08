"""No-cache HTTP server for NautiCAI frontend dev."""
import http.server, functools

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress logs

if __name__ == "__main__":
    http.server.test(HandlerClass=NoCacheHandler, port=3000, bind="0.0.0.0")
