import http.server
import socketserver
import os

# Serve from 'output' folder (relative path, but DON'T change directory)
DIRECTORY = "output"

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'x-api-key,Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.end_headers()

if __name__ == "__main__":
    PORT = 8000

    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🚀 Serving CORS files from: {os.path.abspath(DIRECTORY)} at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")