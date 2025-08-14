import http.server
import socketserver
import os
import socket

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

def find_available_port(start_port=8000, max_attempts=100):
    port = start_port
    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            port += 1
    raise OSError(f"No available port found in range {start_port}-{start_port + max_attempts}")

if __name__ == "__main__":
    PORT = find_available_port()
    
    with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
        print(f"🚀 Serving CORS files from: {os.path.abspath(DIRECTORY)} at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped.")