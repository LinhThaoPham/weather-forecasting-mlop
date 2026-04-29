"""Simple HTTP server to serve the HTML dashboard on Cloud Run."""
import os
import http.server
import socketserver

PORT = int(os.environ.get("PORT", 8080))
DIRECTORY = os.path.join(os.path.dirname(__file__))


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # CORS headers for API calls
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()


if __name__ == "__main__":
    with socketserver.TCPServer(("0.0.0.0", PORT), DashboardHandler) as httpd:
        print(f"[OK] Dashboard serving at http://0.0.0.0:{PORT}")
        httpd.serve_forever()
