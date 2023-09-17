"""
This provides performance measurement utilities, to aid performance optimization work.
"""
from __future__ import annotations
__all__ = ['FinalizedPerformanceLog', 'PerformanceLog']
class FinalizedPerformanceLog:
    def prettyPrint(self) -> str:
        ...
    def toJson(self) -> str:
        ...
class PerformanceLog:
    def finalize(self) -> dict[str, FinalizedPerformanceLog]:
        ...
