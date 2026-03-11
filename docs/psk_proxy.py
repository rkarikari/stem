#!/usr/bin/env python3
"""
PSKReporter local proxy for radio_range.html
Run this script, then use radio_range.html normally.
The HTML page will fetch PSK spots via http://localhost:7373

Usage:
    python3 psk_proxy.py
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request, urllib.parse, sys

PORT = 7373
PSK_HOST = "https://retrieve.pskreporter.info"

class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = PSK_HOST + self.path
        print(f"[proxy] GET {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "radio_range/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            print(f"[proxy] OK {len(data)} bytes")
            self.send_response(200)
            self.send_header("Content-Type", "text/xml; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET")
            self.end_headers()
            self.wfile.write(data)
        except Exception as e:
            print(f"[proxy] ERROR {e}")
            self.send_response(500)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(str(e).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()

    def log_message(self, fmt, *args):
        pass  # silence default access log

print(f"PSKReporter proxy running on http://localhost:{PORT}")
print(f"Open radio_range.html in your browser — PSK spots will work automatically.")
print(f"Press Ctrl+C to stop.\n")
try:
    HTTPServer(("localhost", PORT), ProxyHandler).serve_forever()
except KeyboardInterrupt:
    print("\nProxy stopped.")
    sys.exit(0)
