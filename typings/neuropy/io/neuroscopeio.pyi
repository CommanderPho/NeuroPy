"""
This type stub file was generated by pyright.
"""

from .. import core

"""
This type stub file was generated by pyright.
"""
class NeuroscopeIO:
    def __init__(self, xml_filename) -> None:
        ...
    
    def __str__(self) -> str:
        ...
    
    def set_datetime(self, datetime_epoch):
        """Often a resulting recording file is creating after concatenating different blocks.
        This method takes Epoch array containing datetime.
        """
        ...
    
    def write_neurons(self, neurons: core.Neurons):
        """To view spikes in neuroscope, spikes are exported to .clu.1 and .res.1 files in the basepath.
        You can order the spikes in a way to view sequential activity in neuroscope.

        Parameters
        ----------
        spks : list
            list of spike times.
        """
        ...
    
    def write_epochs(self, epochs: core.Epoch, ext=...):
        ...
    
    def write_position(self, position: core.Position):
        ...
    
    def to_dict(self, recurrsively=...):
        ...
    


