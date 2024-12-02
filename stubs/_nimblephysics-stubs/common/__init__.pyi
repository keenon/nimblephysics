from __future__ import annotations
import nimblephysics_libs._nimblephysics.common
import typing

__all__ = [
    "Composite",
    "Observer",
    "ResourceRetriever",
    "Subject",
    "Uri",
    "UriComponent"
]


class Composite():
    def __init__(self) -> None: ...
    pass
class Observer():
    pass
class ResourceRetriever():
    pass
class Subject():
    pass
class Uri():
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, input: str) -> None: ...
    def clear(self) -> None: ...
    @staticmethod
    def createFromPath(path: str) -> Uri: ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: str, relative: str) -> Uri: ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: str, relative: str, strict: bool) -> Uri: ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: Uri, relative: str) -> Uri: ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: Uri, relative: str, strict: bool) -> Uri: ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: Uri, relative: Uri) -> Uri: ...
    @staticmethod
    @typing.overload
    def createFromRelativeUri(base: Uri, relative: Uri, strict: bool) -> Uri: ...
    @staticmethod
    def createFromString(input: str) -> Uri: ...
    @staticmethod
    def createFromStringOrPath(input: str) -> Uri: ...
    def fromPath(self, path: str) -> bool: ...
    @typing.overload
    def fromRelativeUri(self, base: str, relative: str) -> bool: ...
    @typing.overload
    def fromRelativeUri(self, base: str, relative: str, strict: bool) -> bool: ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: str) -> bool: ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: str, strict: bool) -> bool: ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: Uri) -> bool: ...
    @typing.overload
    def fromRelativeUri(self, base: Uri, relative: Uri, strict: bool) -> bool: ...
    def fromString(self, input: str) -> bool: ...
    def fromStringOrPath(self, input: str) -> bool: ...
    def getFilesystemPath(self) -> str: ...
    def getPath(self) -> str: ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: str, relative: str) -> str: ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: str, relative: str, strict: bool) -> str: ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: Uri, relative: str) -> str: ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: Uri, relative: str, strict: bool) -> str: ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: Uri, relative: Uri) -> str: ...
    @staticmethod
    @typing.overload
    def getRelativeUri(base: Uri, relative: Uri, strict: bool) -> str: ...
    @staticmethod
    def getUri(input: str) -> str: ...
    def toString(self) -> str: ...
    @property
    def mAuthority(self) -> UriComponent:
        """
        :type: UriComponent
        """
    @mAuthority.setter
    def mAuthority(self, arg0: UriComponent) -> None:
        pass
    @property
    def mFragment(self) -> UriComponent:
        """
        :type: UriComponent
        """
    @mFragment.setter
    def mFragment(self, arg0: UriComponent) -> None:
        pass
    @property
    def mPath(self) -> UriComponent:
        """
        :type: UriComponent
        """
    @mPath.setter
    def mPath(self, arg0: UriComponent) -> None:
        pass
    @property
    def mQuery(self) -> UriComponent:
        """
        :type: UriComponent
        """
    @mQuery.setter
    def mQuery(self, arg0: UriComponent) -> None:
        pass
    @property
    def mScheme(self) -> UriComponent:
        """
        :type: UriComponent
        """
    @mScheme.setter
    def mScheme(self, arg0: UriComponent) -> None:
        pass
    pass
class UriComponent():
    def __init__(self) -> None: ...
    def get(self) -> str: ...
    def getOrDefault(self, orDefault: str) -> str: ...
    pass
