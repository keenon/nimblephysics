"""This provides performance measurement utilities, to aid performance optimization work."""
from __future__ import annotations
import nimblephysics_libs._nimblephysics.performance
import typing

__all__ = [
    "FinalizedPerformanceLog",
    "PerformanceLog"
]


class FinalizedPerformanceLog():
    def prettyPrint(self) -> str: ...
    def toJson(self) -> str: ...
    pass
class PerformanceLog():
    def finalize(self) -> typing.Dict[str, FinalizedPerformanceLog]: ...
    pass
