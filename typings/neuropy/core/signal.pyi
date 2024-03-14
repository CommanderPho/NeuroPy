"""
This type stub file was generated by pyright.
"""

"""
This type stub file was generated by pyright.
"""
class Signal:
    def __init__(self, traces, sampling_rate, t_start=..., channel_id=...) -> None:
        ...
    
    @property
    def t_stop(self):
        ...
    
    @property
    def duration(self):
        ...
    
    @property
    def n_channels(self):
        ...
    
    @property
    def n_frames(self):
        ...
    
    @property
    def sampling_rate(self):
        ...
    
    @sampling_rate.setter
    def sampling_rate(self, srate):
        ...
    
    @property
    def time(self):
        ...
    
    def time_slice(self, channel_id=..., t_start=..., t_stop=...):
        ...
    


