from __future__ import annotations
import typing
__all__ = ['Composite', 'Observer', 'ResourceRetriever', 'Subject', 'Uri', 'UriComponent']
class Composite:
    def __init__(self) -> None:
        ...
class Observer:
    pass
class ResourceRetriever:
    pass
class Subject:
    pass
class Uri:
    mAuthority: UriComponent
    mFragment: UriComponent
    mPath: UriComponent
    mQuery: UriComponent
    mScheme: UriComponent
    @staticmethod
    def createFromPath(path: str) -> Uri:
        ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: str, relative: str) -> Uri:
        ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: str, relative: str, strict: bool) -> Uri:
        ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: Uri, relative: str) -> Uri:
        ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: Uri, relative: str, strict: bool) -> Uri:
        ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: Uri, relative: Uri) -> Uri:
        ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: Uri, relative: Uri, strict: bool) -> Uri:
        ...
    @staticmethod
    def createFromString(input: str) -> Uri:
        ...
    @staticmethod
    def createFromStringOrPath(input: str) -> Uri:
        ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: str, relative: str) -> str:
        ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: str, relative: str, strict: bool) -> str:
        ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: Uri, relative: str) -> str:
        ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: Uri, relative: str, strict: bool) -> str:
        ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: Uri, relative: Uri) -> str:
        ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: Uri, relative: Uri, strict: bool) -> str:
        ...
    @staticmethod
    def getUri(input: str) -> str:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, input: str) -> None:
        ...
    @typing.overload
    def __init__(self, input: str) -> None:
        ...
    def clear(self) -> None:
        ...
    def fromPath(self, path: str) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: str, relative: str) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: str, relative: str, strict: bool) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: str, relative: str) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: str, relative: str, strict: bool) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: str) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: str, strict: bool) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: str) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: str, strict: bool) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: Uri) -> bool:
        ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: Uri, strict: bool) -> bool:
        ...
    def fromString(self, input: str) -> bool:
        ...
    def fromStringOrPath(self, input: str) -> bool:
        ...
    def getFilesystemPath(self) -> str:
        ...
    def getPath(self) -> str:
        ...
    def toString(self) -> str:
        ...
class UriComponent:
    def __init__(self) -> None:
        ...
    def get(self) -> str:
        ...
    def getOrDefault(self, orDefault: str) -> str:
        ...
