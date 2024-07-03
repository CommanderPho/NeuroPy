"""
This type stub file was generated by pyright.
"""

from .datawriter import DataWriter

class Shank:
    def __init__(self) -> None:
        ...
    
    @staticmethod
    def auto_generate(columns=..., contacts_per_column=..., xpitch=..., ypitch=..., y_shift_per_column=..., channel_id=...): # -> Shank:
        ...
    
    @staticmethod
    def from_library(probe_name): # -> None:
        ...
    
    @staticmethod
    def set_contacts(positions, channel_ids): # -> None:
        ...
    
    @property
    def x(self): # -> None:
        ...
    
    @x.setter
    def x(self, arr): # -> None:
        ...
    
    @property
    def y(self): # -> None:
        ...
    
    @y.setter
    def y(self, arr): # -> None:
        ...
    
    @property
    def contact_id(self): # -> None:
        ...
    
    @property
    def channel_id(self): # -> None:
        ...
    
    @channel_id.setter
    def channel_id(self, chan_ids): # -> None:
        ...
    
    @property
    def connected(self): # -> None:
        ...
    
    @connected.setter
    def connected(self, arr): # -> None:
        ...
    
    @property
    def n_contacts(self): # -> int:
        ...
    
    def to_dict(self, recurrsively=...): # -> dict[str, Any]:
        ...
    
    def from_dict(self): # -> None:
        ...
    
    def set_disconnected_channels(self, channel_ids): # -> None:
        ...
    
    def to_dataframe(self):
        ...
    
    def move(self, translation): # -> None:
        ...
    


class Probe:
    def __init__(self, shanks, shank_pitch=...) -> None:
        ...
    
    @property
    def n_contacts(self): # -> int:
        ...
    
    @property
    def n_shanks(self):
        ...
    
    @property
    def shank_id(self):
        ...
    
    @property
    def x(self):
        ...
    
    @property
    def x_max(self):
        ...
    
    @property
    def y(self):
        ...
    
    @property
    def channel_id(self):
        ...
    
    @property
    def connected(self):
        ...
    
    def add_shanks(self, shanks: Shank, shank_pitch=...): # -> None:
        ...
    
    def to_dict(self, recurrsively=...):
        ...
    
    def to_dataframe(self):
        ...
    
    def move(self, translation): # -> None:
        ...
    


class ProbeGroup(DataWriter):
    def __init__(self, metadata=...) -> None:
        ...
    
    @property
    def x(self):
        ...
    
    @property
    def x_min(self):
        ...
    
    @property
    def x_max(self):
        ...
    
    @property
    def y(self):
        ...
    
    @property
    def y_min(self):
        ...
    
    @property
    def y_max(self):
        ...
    
    @property
    def n_contacts(self): # -> int:
        ...
    
    @property
    def channel_id(self):
        ...
    
    @property
    def shank_id(self):
        ...
    
    def get_channels(self, groupby=...):
        ...
    
    def get_shank_id_for_channels(self, channel_id):
        """Get shank ids for the channels.

        Parameters
        ----------
        channel_id : array
            channel_ids, can have repeated values

        Returns
        -------
        array
            shank_ids corresponding to the channels
        """
        ...
    
    def get_probe(self): # -> None:
        ...
    
    def get_connected_channels(self, groupby=...):
        ...
    
    @property
    def probe_id(self):
        ...
    
    @property
    def n_probes(self): # -> int:
        ...
    
    @property
    def n_shanks(self): # -> int:
        ...
    
    @property
    def get_disconnected(self):
        ...
    
    def add_probe(self, probe: Probe): # -> None:
        ...
    
    def to_dict(self, recurrsively=...): # -> dict[str, Any]:
        ...
    
    @staticmethod
    def from_dict(d: dict): # -> ProbeGroup:
        ...
    
    def to_dataframe(self):
        ...
    
    def remove_probes(self, probe_id=...): # -> None:
        ...
    


