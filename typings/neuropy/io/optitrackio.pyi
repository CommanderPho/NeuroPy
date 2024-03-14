"""
This type stub file was generated by pyright.
"""

"""
This type stub file was generated by pyright.
"""
def getSampleRate(fileName):
    """Get Sample rate from csv header file - not set at 120Hz"""
    ...

def getnframes(fileName):
    """Get nframes from csv header file"""
    ...

def getnframes_fbx(fileName):
    ...

def getunits(fileName):
    """determine if position data is in centimeters or meters"""
    ...

def posfromCSV(fileName):
    """Import position data from OptiTrack CSV file"""
    ...

def interp_missing_pos(x, y, z, t):
    """Interpolate missing data points"""
    ...

def posfromFBX(fileName):
    ...

def getStartTime(fileName):
    """Get optitrack start time"""
    ...

def timestamps_from_oe(rec_folder, data_type=...):
    """Gets timestamps for all recordings/experiments in a given recording folder. Assumes you have recorded
    in flat binary format in OpenEphys and left the directory structure intact. continuous data by default,
    set data_type='events' for TTL timestamps"""
    ...

def get_sync_info(_sync_file):
    ...

class OptitrackIO:
    def __init__(self, dirname, scale_factor=...) -> None:
        ...
    
    def get_position_at_datetimes(self, dt):
        ...
    
    def old_stuff(self):
        """get position data from files. All position related files should be in 'position' folder within basepath

        Parameters
        ----------
        method : str, optional
            method to grab file start times: "from_metadata" (default) grabs from metadata.csv file,
                                             "from_files" grabs from timestamps.npy files in open-ephys folders
        scale : float, optional
            scale the extracted coordinates, by default 1.0
        """
        ...
    


