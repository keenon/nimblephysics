from __future__ import annotations
import nimblephysics_libs._nimblephysics.biomechanics.OpenSimParser
import typing
import nimblephysics_libs._nimblephysics.biomechanics
import nimblephysics_libs._nimblephysics.common
import nimblephysics_libs._nimblephysics.dynamics
import numpy
_Shape = typing.Tuple[int, ...]

__all__ = [
    "appendMocoTrajectoryAndSaveCSV",
    "convertOsimToMJCF",
    "convertOsimToSDF",
    "filterJustMarkers",
    "getScaleAndMarkerOffsets",
    "hasArms",
    "hasTorso",
    "isArmBodyHeuristic",
    "isTorsoBodyHeuristic",
    "loadGRF",
    "loadMocoTrajectory",
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
    "saveTRC",
    "translateOsimMarkers"
]


def appendMocoTrajectoryAndSaveCSV(inputPath: str, mocoTraj: nimblephysics_libs._nimblephysics.biomechanics.OpenSimMocoTrajectory, outputPath: str) -> None:
    pass
def convertOsimToMJCF(uri: nimblephysics_libs._nimblephysics.common.Uri, outputPath: str, mergeBodiesInto: typing.Dict[str, str]) -> bool:
    pass
def convertOsimToSDF(uri: nimblephysics_libs._nimblephysics.common.Uri, outputPath: str, mergeBodiesInto: typing.Dict[str, str]) -> bool:
    pass
def filterJustMarkers(inputPath: nimblephysics_libs._nimblephysics.common.Uri, outputPath: str) -> None:
    pass
def getScaleAndMarkerOffsets(standardSkeleton: nimblephysics_libs._nimblephysics.biomechanics.OpenSimFile, scaledSkeleton: nimblephysics_libs._nimblephysics.biomechanics.OpenSimFile) -> nimblephysics_libs._nimblephysics.biomechanics.OpenSimScaleAndMarkerOffsets:
    pass
def hasArms(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> bool:
    pass
def hasTorso(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton) -> bool:
    pass
def isArmBodyHeuristic(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, bodyName: str) -> bool:
    pass
def isTorsoBodyHeuristic(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, bodyName: str) -> bool:
    pass
def loadGRF(path: str, targetTimestamps: typing.List[float] = []) -> typing.List[nimblephysics_libs._nimblephysics.biomechanics.ForcePlate]:
    pass
def loadMocoTrajectory(path: str) -> nimblephysics_libs._nimblephysics.biomechanics.OpenSimMocoTrajectory:
    pass
def loadMot(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, path: str) -> nimblephysics_libs._nimblephysics.biomechanics.OpenSimMot:
    pass
def loadMotAtLowestMarkerRMSERotation(osim: nimblephysics_libs._nimblephysics.biomechanics.OpenSimFile, path: str, c3d: nimblephysics_libs._nimblephysics.biomechanics.C3D) -> nimblephysics_libs._nimblephysics.biomechanics.OpenSimMot:
    pass
def loadTRC(path: str) -> nimblephysics_libs._nimblephysics.biomechanics.OpenSimTRC:
    pass
def moveOsimMarkers(inputPath: nimblephysics_libs._nimblephysics.common.Uri, bodyScales: typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]], markerOffsets: typing.Dict[str, typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], outputPath: str) -> None:
    pass
def parseOsim(path: str, geometryFolder: str = '', ignoreGeometry: bool = False) -> nimblephysics_libs._nimblephysics.biomechanics.OpenSimFile:
    pass
def rationalizeJoints(inputPath: nimblephysics_libs._nimblephysics.common.Uri, outputPath: str) -> None:
    pass
def replaceOsimInertia(inputPath: nimblephysics_libs._nimblephysics.common.Uri, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, outputPath: str) -> None:
    pass
def replaceOsimMarkers(inputPath: nimblephysics_libs._nimblephysics.common.Uri, markers: typing.Dict[str, typing.Tuple[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]], isAnatomical: typing.Dict[str, bool], outputPath: str) -> None:
    pass
def saveIDMot(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, outputPath: str, timestamps: typing.List[float], forcePlates: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
    pass
def saveMot(skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, path: str, timestamps: typing.List[float], poses: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
    pass
def saveOsimInverseDynamicsProcessedForcesXMLFile(subjectName: str, contactBodies: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode], grfForcePath: str, forcesOutputPath: str) -> None:
    pass
def saveOsimInverseDynamicsRawForcesXMLFile(subjectName: str, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, poses: numpy.ndarray[numpy.float64, _Shape[m, n]], forcePlates: typing.List[nimblephysics_libs._nimblephysics.biomechanics.ForcePlate], grfForcePath: str, forcesOutputPath: str) -> None:
    pass
def saveOsimInverseDynamicsXMLFile(subjectName: str, osimInputModelPath: str, osimInputMotPath: str, osimForcesXmlPath: str, osimOutputStoPath: str, osimOutputBodyForcesStoPath: str, idInstructionsOutputPath: str, startTime: float, endTime: float) -> None:
    pass
def saveOsimInverseKinematicsXMLFile(subjectName: str, markerNames: typing.List[str], osimInputModelPath: str, osimInputTrcPath: str, osimOutputMotPath: str, ikInstructionsOutputPath: str) -> None:
    pass
def saveOsimScalingXMLFile(subjectName: str, skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, massKg: float, heightM: float, osimInputPath: str, osimInputMarkersPath: str, osimOutputPath: str, scalingInstructionsOutputPath: str) -> None:
    pass
def saveProcessedGRFMot(outputPath: str, timestamps: typing.List[float], bodyNodes: typing.List[nimblephysics_libs._nimblephysics.dynamics.BodyNode], skel: nimblephysics_libs._nimblephysics.dynamics.Skeleton, poses: numpy.ndarray[numpy.float64, _Shape[m, n]], forcePlates: typing.List[nimblephysics_libs._nimblephysics.biomechanics.ForcePlate], wrenches: numpy.ndarray[numpy.float64, _Shape[m, n]]) -> None:
    pass
def saveRawGRFMot(outputPath: str, timestamps: typing.List[float], forcePlates: typing.List[nimblephysics_libs._nimblephysics.biomechanics.ForcePlate]) -> None:
    pass
def saveTRC(path: str, timestamps: typing.List[float], markerTimestamps: typing.List[typing.Dict[str, numpy.ndarray[numpy.float64, _Shape[3, 1]]]]) -> None:
    pass
def translateOsimMarkers(originalModelPath: nimblephysics_libs._nimblephysics.common.Uri, targetModelPath: nimblephysics_libs._nimblephysics.common.Uri, outputPath: str, verbose: bool = False) -> typing.Tuple[typing.List[str], typing.List[str]]:
    pass
