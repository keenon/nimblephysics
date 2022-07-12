import nimblephysics_libs._nimblephysics as nimble
import os
import numpy as np


class MarkerSetTool:
    gui: nimble.server.GUIWebsocketServer
    osimPath: str
    osimFile: nimble.biomechanics.OpenSimFile

    def __init__(self, nativeGUI: nimble.server.GUIWebsocketServer):
        self.gui = nativeGUI

    def loadModel(self, path: str):
        self.osimPath = path
        self.osimFile = nimble.biomechanics.OpenSimParser.parseOsim(path)
        self.gui.renderBasis()
        self.gui.renderSkeleton(self.osimFile.skeleton)
        self.gui.createSphere("Marker1", 0.05, [0, 0, 0], [1, 0, 0, 1])
        self.gui.setObjectTooltip("Marker1", "MKR1")
        self.gui.registerDragListener(
            "Marker1",
            lambda pos: self.gui.setObjectPosition("Marker1", pos),
            lambda: print('drag finished!'))
        self.gui.registerTooltipChangeListener("Marker1", lambda newTooltip:
                                               self.gui.setObjectTooltip(
                                                   "Marker1", newTooltip)
                                               )
        self.gui.createButton(
            "create_marker",
            "Create Marker",
            [50, 100],
            [150, 30],
            lambda: print("Clicked!"))

    def saveModel(self, path: str):
        markers: Dict[str, (str, np.ndarray)] = {
            'test': ('pelvis', [0, 1, 2])
        }
        isAnatomical: Dict[str, bool] = {
            'test': True
        }
        nimble.biomechanics.OpenSimParser.replaceOsimMarkers(
            self.osimPath, markers, isAnatomical, path)


if __name__ == "__main__":
    nativeGUI = nimble.server.GUIWebsocketServer()
    nativeGUI.serve(8070)
    tool = MarkerSetTool(nativeGUI)
    tool.loadModel(os.path.abspath(
        "./models/rajagopal_data/Rajagopal2015.osim"))
    tool.saveModel(os.path.abspath(
        "./models/rajagopal_data/Rajagopal2015_newMarkers.osim"))
    nativeGUI.blockWhileServing()
