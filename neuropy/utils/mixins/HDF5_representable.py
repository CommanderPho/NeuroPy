from pathlib import Path

# ==================================================================================================================== #
# 2023-07-30 HDF5 General Object Serialization Classes                                                                 #
# ==================================================================================================================== #


# Deserialization ____________________________________________________________________________________________________ #

def post_deserialize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    wrapper._is_post_deserialize = True
    return wrapper




class HDF_DeserializationMixin(AttrsBasedClassHelperMixin):
    def deserialize(self, *args, **kwargs):
        # Your deserialization logic here
        
        # Call post-deserialization methods
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_is_post_deserialize'):
                attr()


    @classmethod
    def read_hdf(cls, file_path, key: str, **kwargs):
        """ Reads the data from the key in the hdf5 file at file_path
        Usage:
            _reread_pos_obj = cls.read_hdf(hdf5_output_path, key='pos')
            _reread_pos_obj
        """
        raise NotImplementedError # implementor must override!

        # # Read the DataFrame using pandas
        # pos_df = pd.read_hdf(file_path, key=key)

        # # Open the file with h5py to read attributes
        # with h5py.File(file_path, 'r') as f:
        #     dataset = f[key]
        #     metadata = {
        #         'time_variable_name': dataset.attrs['time_variable_name'],
        #         'sampling_rate': dataset.attrs['sampling_rate'],
        #         't_start': dataset.attrs['t_start'],
        #         't_stop': dataset.attrs['t_stop'],
        #     }

        # # Reconstruct the object using the class constructor
        # _out = cls(pos_df=pos_df, metadata=metadata)
        # _out.filename = file_path # set the filename it was loaded from
        # return _out



""" Usage of DeserializationMixin

from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, computed_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin


class MyClass(DeserializationMixin):
    def __init__(self, value):
        self.value = value

    @post_deserialize
    def setup(self):
        print(f"Post-deserialization setup for value: {self.value}")

        
"""





class HDF_SerializationMixin(AttrsBasedClassHelperMixin):
    """
    Inherits `get_serialized_fields` from AttrsBasedClassHelperMixin
    """


    def to_hdf(self, file_path, key: str, **kwargs):
        """ Saves the object to key in the hdf5 file specified by file_path
        Usage:
            hdf5_output_path: Path = curr_active_pipeline.get_output_path().joinpath('test_data.h5')
            _pfnd_obj: PfND = long_one_step_decoder_1D.pf
            _pfnd_obj.to_hdf(hdf5_output_path, key='test_pfnd')
        """
        raise NotImplementedError # implementor must override!
        # self.position.to_hdf(file_path=file_path, key=f'{key}/pos')
        # if self.epochs is not None:
        #     self.epochs.to_hdf(file_path=file_path, key=f'{key}/epochs') #TODO 2023-07-30 11:13: - [ ] What if self.epochs is None?
        # else:
        #     # if self.epochs is None
        #     pass
        # self.spikes_df.spikes.to_hdf(file_path, key=f'{key}/spikes')

        # # Open the file with h5py to add attributes to the group. The pandas.HDFStore object doesn't provide a direct way to manipulate groups as objects, as it is primarily intended to work with datasets (i.e., pandas DataFrames)
        # with h5py.File(file_path, 'r+') as f:
        #     ## Unfortunately, you cannot directly assign a dictionary to the attrs attribute of an h5py group or dataset. The attrs attribute is an instance of a special class that behaves like a dictionary in some ways but not in others. You must assign attributes individually
        #     group = f[key]
        #     group.attrs['position_srate'] = self.position_srate
        #     group.attrs['ndim'] = self.ndim

        #     # can't just set the dict directly
        #     # group.attrs['config'] = str(self.config.to_dict())  # Store as string if it's a complex object
        #     # Manually set the config attributes
        #     config_dict = self.config.to_dict()
        #     group.attrs['config/speed_thresh'] = config_dict['speed_thresh']
        #     group.attrs['config/grid_bin'] = config_dict['grid_bin']
        #     group.attrs['config/grid_bin_bounds'] = config_dict['grid_bin_bounds']
        #     group.attrs['config/smooth'] = config_dict['smooth']
        #     group.attrs['config/frate_thresh'] = config_dict['frate_thresh']
            


# General/Combined ___________________________________________________________________________________________________ #

class HDFMixin(HDF_DeserializationMixin, HDF_SerializationMixin):
    # Common methods for serialization and deserialization
    pass

"""
from neuropy.utils.mixins.HDF5_representable import HDFMixin
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, serialized_field, computed_field
from neuropy.utils.mixins.HDF5_representable import HDF_DeserializationMixin, post_deserialize, HDF_SerializationMixin, HDFMixin

class SpecialClassHDFMixin(HDFMixin):
    # Custom methods for a specific class

class MyClass(SpecialClassHDFMixin):
    # MyClass definition

"""



# class HDF5_Initializable(FileInitializable):
# 	""" Implementors can be initialized from a file path (from which they are loaded)
# 	"""
# 	@classmethod
# 	def from_file(cls, f):
# 		assert isinstance(f, (str, Path))
# 		raise NotImplementedError


# class HDF5_Representable(HDF5_Initializable, FileRepresentable):
# 	""" Implementors can be loaded or saved to a file
# 	"""
# 	@classmethod
# 	def to_file(cls, data: dict, f):
# 		raise NotImplementedError

 
# 	def save(self):
# 		raise NotImplementedError