from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from http import HTTPStatus
import os
import pathlib
import nimblephysics as nimble
import random
import typing
import threading
from typing import List
import torch
import numpy as np


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


class NimbleGUI:
  def __init__(self, worldToCopy: nimble.simulation.World):
    self.world = worldToCopy.clone()
    self.guiServer = nimble.server.GUIWebsocketServer()
    self.guiServer.renderWorld(self.world)
    # Set up the realtime animation
    self.ticker = nimble.realtime.Ticker(self.world.getTimeStep() * 10)
    self.ticker.registerTickListener(self._onTick)
    self.guiServer.registerConnectionListener(self._onConnect)

    self.looping = False
    self.posMatrixToLoop = np.zeros((self.world.getNumDofs(), 0))
    self.i = 0

  def serve(self, port):
    self.guiServer.serve(8070)
    server_address = ('', port)
    self.httpd = ThreadingHTTPServer(server_address, createRequestHandler())
    print('Web GUI serving on http://localhost:'+str(port))
    t = threading.Thread(None, self.httpd.serve_forever)
    t.daemon = True
    t.start()

  def stopServing(self):
    self.guiServer.stopServing()
    self.httpd.shutdown()

  def displayState(self, state: torch.Tensor):
    self.looping = False
    self.world.setState(state.detach().numpy())
    self.guiServer.renderWorld(self.world)

  def loopStates(self, states: List[torch.Tensor]):
    self.looping = True
    self.statesToLoop = states
    dofs = self.world.getNumDofs()
    poses = np.zeros((dofs, len(states)))
    for i in range(len(states)):
      # Take the top-half of each state vector, since this is the position component
      poses[:, i] = states[i].detach().numpy()[:dofs]
    self.guiServer.renderTrajectoryLines(self.world, poses)
    self.posMatrixToLoop = poses

  def loopPosMatrix(self, poses: np.ndarray):
    self.looping = True
    self.guiServer.renderTrajectoryLines(self.world, poses)
    # It's important to make a copy, because otherwise we get a reference to internal C++ memory that gets cleared
    self.posMatrixToLoop = np.copy(poses)

  def stopLooping(self):
    self.looping = False

  def nativeAPI(self) -> nimble.server.GUIWebsocketServer:
    return self.guiServer

  def blockWhileServing(self):
    self.guiServer.blockWhileServing()

  def _onTick(self, now):
    if self.looping:
      if self.i < np.shape(self.posMatrixToLoop)[1]:
        self.world.setPositions(self.posMatrixToLoop[:, self.i])
        self.guiServer.renderWorld(self.world)
        self.i += 1
      else:
        self.i = 0

  def _onConnect(self):
    self.ticker.start()
