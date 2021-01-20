from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from http import HTTPStatus
import os
import pathlib
import diffdart as dart
import random
import typing
import threading


file_path = os.path.join(pathlib.Path(__file__).parent.absolute(), 'web_gui')


def createRequestHandler():
    """
    This creates a request handler that can serve the raw web GUI files, in
    addition to a configuration string of JSON.
    """
    class LocalHTTPRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=file_path, **kwargs)

        def do_GET(self):
            """
            if self.path == '/json':
                resp = jsonConfig.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-type", "application/json")
                self.send_header("Content-Length", len(resp))
                self.end_headers()
                self.wfile.write(resp)
            else:
                super().do_GET()
            """
            super().do_GET()
    return LocalHTTPRequestHandler


class DartGUI:
    def __init__(self):
        self.guiServer = dart.server.GUIWebsocketServer()

    def serve(self, port):
        self.guiServer.serve(8070)
        server_address = ('', port)
        self.httpd = ThreadingHTTPServer(server_address, createRequestHandler())
        print('Web GUI serving optimization solution on http://localhost:'+str(port))
        t = threading.Thread(None, self.httpd.serve_forever)
        t.daemon = True
        t.start()

    def stateMachine(self) -> dart.server.GUIWebsocketServer:
        return self.guiServer

    def stopServing(self):
        self.guiServer.stopServing()
        self.httpd.shutdown()
