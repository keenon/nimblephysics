from http.server import HTTPServer, SimpleHTTPRequestHandler
from http import HTTPStatus
import os
import pathlib


file_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'web_gui')
global_json = "{ hello: world }"


class LocalHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=file_path, **kwargs)

    def do_GET(self):
        if self.path == '/json':
            resp = global_json.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-type", "application/json")
            self.send_header("Content-Length", len(resp))
            self.end_headers()
            self.wfile.write(resp)
        else:
            super().do_GET()


def dart_serve_web_gui(json: str, port=8000):
    global global_json
    global_json = json
    server_address = ('', port)
    httpd = HTTPServer(server_address, LocalHTTPRequestHandler)
    print('Web GUI serving on http://localhost:'+str(port))
    httpd.serve_forever()


if __name__ == '__main__':
    dart_serve_web_gui('{}')
