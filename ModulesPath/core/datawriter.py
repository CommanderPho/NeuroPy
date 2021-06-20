import numpy as np
from pathlib import Path


class DataWriter:
    def __init__(self, filename=None) -> None:

        if filename is not None:
            self.filename = Path(filename)
        else:
            self.filename = None

    def load(self):
        if self.filename.is_file():
            return np.load(self.filename, allow_pickle=True).item()
        else:
            return None

    def to_dict(self):
        return NotImplementedError

    def save(self):

        data = self.to_dict()
        if self.filename is not None:
            assert isinstance(self.filename, Path)
            np.save(self.filename, data)
            print("data saved")
        else:
            print("filename not understood")

    def delete_file(self):
        self.filename.unlink()

        print("file removed")

    def create_backup(self):
        pass
