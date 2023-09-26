"""
This provides a native WebSocket server infrastructure.
"""
from __future__ import annotations
import _nimblephysics.dynamics
import _nimblephysics.simulation
import numpy
import typing
__all__ = ['GUIRecording', 'GUIStateMachine', 'GUIWebsocketServer']
class GUIRecording(GUIStateMachine):
    def __init__(self) -> None:
        ...
    def getFrameJson(self, frame: int) -> str:
        ...
    def getFramesJson(self, startFrame: int = ...) -> str:
        ...
    def getNumFrames(self) -> int:
        ...
    def saveFrame(self) -> None:
        ...
    def writeFrameJson(self, path: str, frame: int) -> None:
        ...
    def writeFramesJson(self, path: str, startFrame: int = ...) -> None:
        ...
class GUIStateMachine:
    def __init__(self) -> None:
        ...
    def clear(self) -> None:
        ...
    def clearBodyWrench(self, body: _nimblephysics.dynamics.BodyNode, prefix: str = ...) -> None:
        ...
    def createBox(self, key: str, size: numpy.ndarray[numpy.float64[3, 1]] = ..., pos: numpy.ndarray[numpy.float64[3, 1]] = ..., euler: numpy.ndarray[numpy.float64[3, 1]] = ..., color: numpy.ndarray[numpy.float64[4, 1]] = ..., layer: str = ..., castShadows: bool = ..., receiveShadows: bool = ...) -> None:
        ...
    def createButton(self, key: str, label: str, fromTopLeft: numpy.ndarray[numpy.int32[2, 1]], size: numpy.ndarray[numpy.int32[2, 1]], onClick: typing.Callable[[], None], layer: str = ...) -> None:
        ...
    def createCapsule(self, key: str, radius: float, height: float, pos: numpy.ndarray[numpy.float64[3, 1]] = ..., euler: numpy.ndarray[numpy.float64[3, 1]] = ..., color: numpy.ndarray[numpy.float64[4, 1]] = ..., layer: str = ..., castShadows: bool = ..., receiveShadows: bool = ...) -> None:
        ...
    def createCone(self, key: str, radius: float, height: float, pos: numpy.ndarray[numpy.float64[3, 1]] = ..., euler: numpy.ndarray[numpy.float64[3, 1]] = ..., color: numpy.ndarray[numpy.float64[4, 1]] = ..., layer: str = ..., castShadows: bool = ..., receiveShadows: bool = ...) -> None:
        ...
    def createCylinder(self, key: str, radius: float, height: float, pos: numpy.ndarray[numpy.float64[3, 1]] = ..., euler: numpy.ndarray[numpy.float64[3, 1]] = ..., color: numpy.ndarray[numpy.float64[4, 1]] = ..., layer: str = ..., castShadows: bool = ..., receiveShadows: bool = ...) -> None:
        ...
    def createLayer(self, key: str, color: numpy.ndarray[numpy.float64[4, 1]] = ..., defaultShow: bool = ...) -> None:
        ...
    def createLine(self, key: str, points: list[numpy.ndarray[numpy.float64[3, 1]]], color: numpy.ndarray[numpy.float64[4, 1]] = ..., layer: str = ..., width: list[float] = ...) -> None:
        ...
    def createMeshFromShape(self, key: str, mesh: _nimblephysics.dynamics.MeshShape, pos: numpy.ndarray[numpy.float64[3, 1]] = ..., euler: numpy.ndarray[numpy.float64[3, 1]] = ..., scale: numpy.ndarray[numpy.float64[3, 1]] = ..., color: numpy.ndarray[numpy.float64[4, 1]] = ..., layer: str = ..., castShadows: bool = ..., receiveShadows: bool = ...) -> None:
        ...
    def createPlot(self, key: str, fromTopLeft: numpy.ndarray[numpy.int32[2, 1]], size: numpy.ndarray[numpy.int32[2, 1]], xs: list[float], minX: float, maxX: float, ys: list[float], minY: float, maxY: float, plotType: str, layer: str = ...) -> None:
        ...
    def createRichPlot(self, key: str, fromTopLeft: numpy.ndarray[numpy.int32[2, 1]], size: numpy.ndarray[numpy.int32[2, 1]], minX: float, maxX: float, minY: float, maxY: float, title: str, xAxisLabel: str, yAxisLabel: str, layer: str = ...) -> None:
        ...
    def createSlider(self, key: str, fromTopLeft: numpy.ndarray[numpy.int32[2, 1]], size: numpy.ndarray[numpy.int32[2, 1]], min: float, max: float, value: float, onlyInts: bool, horizontal: bool, onChange: typing.Callable[[float], None], layer: str = ...) -> None:
        ...
    def createSphere(self, key: str, radii: numpy.ndarray[numpy.float64[3, 1]] = ..., pos: numpy.ndarray[numpy.float64[3, 1]] = ..., color: numpy.ndarray[numpy.float64[4, 1]] = ..., layer: str = ..., castShadows: bool = ..., receiveShadows: bool = ...) -> None:
        ...
    def createText(self, key: str, contents: str, fromTopLeft: numpy.ndarray[numpy.int32[2, 1]], size: numpy.ndarray[numpy.int32[2, 1]], layer: str = ...) -> None:
        ...
    def deleteObject(self, key: str) -> None:
        ...
    def deleteObjectWarning(self, key: str, warningKey: str) -> None:
        ...
    def deleteUIElement(self, key: str) -> None:
        ...
    def getObjectColor(self, key: str) -> numpy.ndarray[numpy.float64[4, 1]]:
        ...
    def getObjectPosition(self, key: str) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    def getObjectRotation(self, key: str) -> numpy.ndarray[numpy.float64[3, 1]]:
        ...
    def renderArrow(self, start: numpy.ndarray[numpy.float64[3, 1]], end: numpy.ndarray[numpy.float64[3, 1]], bodyRadius: float, tipRadius: float, color: numpy.ndarray[numpy.float64[4, 1]] = ..., prefix: str = ..., layer: str = ...) -> None:
        ...
    def renderBasis(self, scale: float = ..., prefix: str = ..., pos: numpy.ndarray[numpy.float64[3, 1]] = ..., euler: numpy.ndarray[numpy.float64[3, 1]] = ..., layer: str = ...) -> None:
        ...
    def renderBodyWrench(self, body: _nimblephysics.dynamics.BodyNode, wrench: numpy.ndarray[numpy.float64[6, 1]], scaleFactor: float = ..., prefix: str = ..., layer: str = ...) -> None:
        ...
    def renderMovingBodyNodeVertices(self, body: _nimblephysics.dynamics.BodyNode, scaleFactor: float = ..., prefix: str = ..., layer: str = ...) -> None:
        ...
    def renderSkeleton(self, skeleton: _nimblephysics.dynamics.Skeleton, prefix: str = ..., overrideColor: numpy.ndarray[numpy.float64[4, 1]] = ..., layer: str = ...) -> None:
        ...
    def renderTrajectoryLines(self, world: _nimblephysics.simulation.World, positions: numpy.ndarray[numpy.float64[m, n]], prefix: str = ..., layer: str = ...) -> None:
        ...
    def renderWorld(self, world: _nimblephysics.simulation.World, prefix: str = ..., renderForces: bool = ..., renderForceMagnitudes: bool = ..., layer: str = ...) -> None:
        ...
    def setButtonLabel(self, key: str, label: str) -> None:
        ...
    def setFramesPerSecond(self, framesPerSecond: int) -> None:
        ...
    def setObjectColor(self, key: str, color: numpy.ndarray[numpy.float64[4, 1]]) -> None:
        ...
    def setObjectPosition(self, key: str, position: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    def setObjectRotation(self, key: str, euler: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    def setObjectScale(self, key: str, scale: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        ...
    def setObjectTooltip(self, key: str, tooltip: str) -> None:
        ...
    def setObjectWarning(self, key: str, warningKey: str, warning: str, layer: str) -> None:
        ...
    def setPlotData(self, key: str, xs: list[float], minX: float, maxX: float, ys: list[float], minY: float, maxY: float) -> None:
        ...
    def setRichPlotBounds(self, key: str, minX: float, maxX: float, minY: float, maxY: float) -> None:
        ...
    def setRichPlotData(self, key: str, name: str, color: str, plotType: str, xs: list[float], ys: list[float]) -> None:
        ...
    def setSliderMax(self, key: str, value: float) -> None:
        ...
    def setSliderMin(self, key: str, value: float) -> None:
        ...
    def setSliderValue(self, key: str, value: float) -> None:
        ...
    def setSpanWarning(self, startTimestep: int, endTimestep: int, warningKey: str, warning: str, layer: str) -> None:
        ...
    def setTextContents(self, key: str, contents: str) -> None:
        ...
    def setUIElementPosition(self, key: str, position: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        ...
    def setUIElementSize(self, key: str, size: numpy.ndarray[numpy.int32[2, 1]]) -> None:
        ...
class GUIWebsocketServer(GUIStateMachine):
    def __init__(self) -> None:
        ...
    def blockWhileServing(self) -> None:
        ...
    def clear(self) -> None:
        ...
    def flush(self) -> None:
        ...
    def getKeysDown(self) -> set[str]:
        ...
    def getScreenSize(self) -> numpy.ndarray[numpy.int32[2, 1]]:
        ...
    def isKeyDown(self, key: str) -> bool:
        ...
    def isServing(self) -> bool:
        ...
    def registerConnectionListener(self, listener: typing.Callable[[], None]) -> None:
        ...
    def registerDragListener(self, key: str, listener: typing.Callable[[numpy.ndarray[numpy.float64[3, 1]]], None], endDrag: typing.Callable[[], None]) -> GUIWebsocketServer:
        ...
    def registerKeydownListener(self, listener: typing.Callable[[str], None]) -> None:
        ...
    def registerKeyupListener(self, listener: typing.Callable[[str], None]) -> None:
        ...
    def registerScreenResizeListener(self, listener: typing.Callable[[numpy.ndarray[numpy.int32[2, 1]]], None]) -> None:
        ...
    def registerShutdownListener(self, listener: typing.Callable[[], None]) -> None:
        ...
    def registerTooltipChangeListener(self, key: str, listener: typing.Callable[[str], None]) -> GUIWebsocketServer:
        ...
    def serve(self, port: int) -> None:
        ...
    def stopServing(self) -> None:
        ...