import numpy as np
import pathlib
from pathlib import Path

from neuropy.utils.mixins.dict_representable import DictRepresentable
from neuropy.utils.mixins.file_representable import FileRepresentable
from neuropy.utils.mixins.print_helpers import SimplePrintable


class DataWriter(FileRepresentable, DictRepresentable, SimplePrintable):
    def __init__(self, metadata=None) -> None:

        self._filename = None

        if metadata is not None:
            assert isinstance(metadata, dict), "Only dictionary accepted as metadata"
            self._metadata: dict = metadata
        else:
            self._metadata: dict = {}

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, d):
        """metadata compatibility"""
        if d is not None:
            assert isinstance(d, dict), "Only dictionary accepted"
            self._metadata = self._metadata | d

    ## DictRepresentable protocol:
    @staticmethod
    def from_dict(d):
        return NotImplementedError

    def to_dict(self, recurrsively=False):
        return NotImplementedError

    ## FileRepresentable protocol:
    @classmethod
    def from_file(cls, f):
        if f.is_file():
            dict_rep = None
            try:
                dict_rep = np.load(f, allow_pickle=True).item()
                # return dict_rep
            except NotImplementedError:
                print("Issue with pickled POSIX_PATH on windows for path {}, falling back to non-pickled version...".format(f))
                temp = pathlib.PosixPath
                # pathlib.PosixPath = pathlib.WindowsPath # Bad hack
                pathlib.PosixPath = pathlib.PurePosixPath # Bad hack
                dict_rep = np.load(f, allow_pickle=True).item()
                # d['filename']
                # print("Post hack decode: {}\n".format(d))
                # return dict_rep
            
            if dict_rep is not None:
                # Convert to object
                obj = cls.from_dict(dict_rep)
                obj.filename = f
                return obj
            return dict_rep
            
        else:
            return None
        
    @classmethod
    def to_file(cls, data: dict, f):
        if f is not None:
            assert isinstance(f, Path)
            np.save(f, data)
            print(f"{f.name} saved")
        else:
            print("filename can not be None")


    def save(self, fp):

        assert isinstance(fp, (str, Path)), "filename is invalid"
        data = self.to_dict()
        np.save(fp, data)
        print(f"{fp} saved")
