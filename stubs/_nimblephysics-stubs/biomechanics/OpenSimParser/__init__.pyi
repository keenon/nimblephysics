from __future__ import annotations
import _nimblephysics.biomechanics.OpenSimParser
import typing
import _nimblephysics.biomechanics
import _nimblephysics.common
import _nimblephysics.dynamics
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "convertOsimToMJCF",
    "convertOsimToSDF",
    "filterJustMarkers",
    "getScaleAndMarkerOffsets",
    "loadGRF",
    "loadMot",
    "loadMotAtLowestMarkerRMSERotation",
    "loadTRC",
    "moveOsimMarkers",
    "parseOsim",
    "rationalizeJoints",
    "replaceOsimInertia",
    "replaceOsimMarkers",
    "saveIDMot",
    "saveMot",
    "saveOsimInverseDynamicsProcessedForcesXMLFile",
    "saveOsimInverseDynamicsRawForcesXMLFile",
    "saveOsimInverseDynamicsXMLFile",
    "saveOsimInverseKinematicsXMLFile",
    "saveOsimScalingXMLFile",
    "saveProcessedGRFMot",
    "saveRawGRFMot",
    "saveTRC"
]


def convertOsimToMJCF(uri: _nimblephysics.common.Uri, outputPath: str, mergeBodiesInto: typing.Dict[str, str]) -> bool:
    pass
def convertOsimToSDF(uri: _nimblephysics.common.Uri, outputPath: str, mergeBodiesInto: typing.Dict[str, str]) -> bool:
    pass
def filterJustMarkers(inputPath: _nimblephysics.common.Uri, outputPath: str) -> None:
    pass
def getScaleAndMarkerOffsets(standardSkeleton: _nimblephysics.biomechanics.OpenSimFile, scaledSkeleton: _nimblephysics.biomechanics.OpenSimFile) -> _nimblephysics.biomechanics.OpenSimScaleAndMarkerOffsets:
    pass
def loadGRF(path: str, targetFramesPerSecond: int = 100) -> typing.List[_nimblephysics.biomechanics.ForcePlate]:
    pass
def loadMot(skel: _nimblephysics.dynamics.Skeleton, path: str) -> _nimblephysics.biomechanics.OpenSimMot:
    pass
def loadMotAtLowestMarkerRMSERotation(osim: _nimblephysics.biomechanics.OpenSimFile, path: str, c3d: _nimblephysics.biomechanics.C3D) -> _nimblephysics.biomechanics.OpenSimMot:
    pass
def loadTRC(path: str) -> _nimblephysics.biomechanics.OpenSimTRC:
    pass
def moveOsimMarkers(inputPath: _nimblephysics.common.Uri, bodyScales: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]], markerOffsets: typing.Dict[str, typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], outputPath: str) -> None:
    pass
def parseOsim(path: str) -> _nimblephysics.biomechanics.OpenSimFile:
    pass
def rationalizeJoints(inputPath: _nimblephysics.common.Uri, outputPath: str) -> None:
    pass
def replaceOsimInertia(inputPath: _nimblephysics.common.Uri, skel: _nimblephysics.dynamics.Skeleton, outputPath: str) -> None:
    pass
def replaceOsimMarkers(inputPath: _nimblephysics.common.Uri, markers: typing.Dict[str, typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], isAnatomical: typing.Dict[str, bool], outputPath: str) -> None:
    pass
def saveIDMot(skel: _nimblephysics.dynamics.Skeleton, outputPath: str, timestamps: typing.List[float], forcePlates: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
    pass
def saveMot(skel: _nimblephysics.dynamics.Skeleton, path: str, timestamps: typing.List[float], poses: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
    pass
def saveOsimInverseDynamicsProcessedForcesXMLFile(subjectName: str, contactBodies: typing.List[_nimblephysics.dynamics.BodyNode], grfForcePath: str, forcesOutputPath: str) -> None:
    pass
def saveOsimInverseDynamicsRawForcesXMLFile(subjectName: str, skel: _nimblephysics.dynamics.Skeleton, poses: numpy.ndarray[numpy.float64, _Shape[m, n]], forcePlates: typing.List[_nimblephysics.biomechanics.ForcePlate], grfForcePath: str, forcesOutputPath: str) -> None:
    pass
def saveOsimInverseDynamicsXMLFile(subjectName: str, osimInputModelPath: str, osimInputMotPath: str, osimForcesXmlPath: str, osimOutputStoPath: str, osimOutputBodyForcesStoPath: str, idInstructionsOutputPath: str, startTime: float, endTime: float) -> None:
    pass
def saveOsimInverseKinematicsXMLFile(subjectName: str, markerNames: typing.List[str], osimInputModelPath: str, osimInputTrcPath: str, osimOutputMotPath: str, ikInstructionsOutputPath: str) -> None:
    pass
def saveOsimScalingXMLFile(subjectName: str, skel: _nimblephysics.dynamics.Skeleton, massKg: float, heightM: float, osimInputPath: str, osimInputMarkersPath: str, osimOutputPath: str, scalingInstructionsOutputPath: str) -> None:
    pass
def saveProcessedGRFMot(outputPath: str, timestamps: typing.List[float], bodyNodes: typing.List[_nimblephysics.dynamics.BodyNode], groundLevel: float, wrenches: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
    pass
def saveRawGRFMot(outputPath: str, timestamps: typing.List[float], forcePlates: typing.List[_nimblephysics.biomechanics.ForcePlate]) -> None:
    pass
def saveTRC(path: str, timestamps: typing.List[float], markerTimestamps: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
    pass
