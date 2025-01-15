from dataclasses import dataclass
from attrs import define, field, Factory
from typing import Callable, List, Dict, Optional
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from neuropy.core.session.dataSession import DataSession
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta
from neuropy.utils.result_context import IdentifyingContext
from neuropy.core.parameters import ParametersContainer


@dataclass
class SessionFileSpec:
    """ Specifies a specification for a single file.
    Members:
        session_load_callback: Callable[[Path, DataSession], DataSession] a function that takes the path to load and the session to load it into and performs the operation
    
    Examples: 
        SessionFileSpec('{}.xml', session_name, 'The primary .xml configuration file'), SessionFileSpec('{}.neurons.npy', session_name, 'The numpy data file containing information about neural activity.')
        SessionFileSpec('{}.probegroup.npy', session_name, 'The numpy data file containing information about the spatial layout of recording probes')
        SessionFileSpec('{}.position.npy', session_name, 'The numpy data file containing the recorded animal positions (as generated by optitrack) over time.')
        SessionFileSpec('{}.paradigm.npy', session_name, 'The numpy data file containing the recording epochs. Each epoch is defined as a: (label:str, t_start: float (in seconds), t_end: float (in seconds))')
    """
    fileSpecString: str
    suggestedBaseName: str
    description: str
    session_load_callback: Callable[[Path, DataSession], DataSession]
    
    @property
    def filename(self):
        """The filename property."""
        return self.fileSpecString.format(self.suggestedBaseName)
    
    def resolved_path(self, parent_path, overrideBasename=None):
        """Gets the resolved Path given the parent_path """
        if overrideBasename is not None:
            self.suggestedBaseName = overrideBasename
        return parent_path.joinpath(self.filename)


class SessionFolderSpecError(Exception):
    """ An exception raised when a session folder spec requirement fails """
    def __init__(self, message, failed_spec_item):
        self.message = message
        self.failed_spec_item = failed_spec_item
    def __str__(self):
        return self.message

class RequiredFileError(SessionFolderSpecError):
    """ An exception raised when a required file is missing """
    def __init__(self, message, missing_file_spec):
        self.message = message
        self.failed_spec_item = missing_file_spec
    def __str__(self):
        return self.message
    
class RequiredValidationFailedError(SessionFolderSpecError):
    """ An exception raised when a required validation spec fails """
    def __init__(self, message, failed_validation):
        self.message = message
        self.failed_spec_item = failed_validation
    def __str__(self):
        return self.message
    

class SessionFolderSpec():
    """ Documents the required and optional files for a given session format """
    def __init__(self, required = [], optional = [], additional_validation_requirements=[]) -> None:
        # additiona_validation_requirements: a list of callbacks that are passed the proposed_session_path on self.validate(...) and return True/False. All must return true for validate to succeed.
        self.required_files = required
        self.optional_files = optional
        self.additional_validation_requirements = additional_validation_requirements
        
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__dict__};>"


    def resolved_paths(self, proposed_session_path):
        """ Gets whether the proposed_session_path meets the requirements and returns the resolved paths if it can.
            Does not check whether any of the files exist, it just builds the paths
            
        Returns:
            two dictionaries containing the resolved path:file_spec pairs
        """
        proposed_session_path = Path(proposed_session_path)
        # build absolute paths from the proposed_session_path and the files:
        resolved_required_filespecs_dict = {a_file_spec.resolved_path(proposed_session_path):a_file_spec for a_file_spec in self.required_files}
        resolved_optional_filespecs_dict = {a_file_spec.resolved_path(proposed_session_path):a_file_spec for a_file_spec in self.optional_files}
        return resolved_required_filespecs_dict, resolved_optional_filespecs_dict
        
    def validate(self, proposed_session_path):
        """Check whether the proposed_session_path meets this folder spec's requirements
        Args:
            proposed_session_path ([Path]): [description]

        Returns:
            [Bool]: [description]
        """
        resolved_required_filespecs_dict, resolved_optional_filespecs_dict = self.resolved_paths(proposed_session_path=proposed_session_path)
            
        meets_spec = False
        if not Path(proposed_session_path).exists():
            meets_spec = False # the path doesn't even exist, it can't be valid
        else:
            # the path exists:
            for a_required_filepath, a_file_spec in resolved_required_filespecs_dict.items():
                if not a_required_filepath.exists():
                    meets_spec = False                    
                    raise RequiredFileError(f'Required File: {a_required_filepath} does not exist.', (a_required_filepath, a_file_spec))
                    break
            for a_required_validation_function in self.additional_validation_requirements:
                if not a_required_validation_function(Path(proposed_session_path)):
                    # print('Required additional_validation_requirements[i]({}) returned False'.format(proposed_session_path))
                    meets_spec = False
                    raise RequiredValidationFailedError(f'Required additional_validation_requirements[i]({proposed_session_path}) returned False', a_required_validation_function)
                    break
                
            for an_optional_filepath, a_file_spec in resolved_optional_filespecs_dict.items():
                if not an_optional_filepath.exists():
                    print(f'WARNING: Optional File: "{an_optional_filepath}" does not exist. Continuing without it.')
                    
            meets_spec = True # otherwise it exists
            
        return meets_spec, resolved_required_filespecs_dict, resolved_optional_filespecs_dict
    

# session_name = '2006-6-07_11-26-53'
# SessionFolderSpec(required=['{}.xml'.format(session_name),
#                             '{}.spikeII.mat'.format(session_name), 
#                             '{}.position_info.mat'.format(session_name),
#                             '{}.epochs_info.mat'.format(session_name), 
# ])



@define(slots=False, repr=False, str=False)
class SessionConfig(SimplePrintable, metaclass=OrderedMeta):
    """A simple data structure that holds the information specifying a data session, such as the basepath, session_spec, and session_name
    
    TODO 2023-10-23 - Upgrade to attrs class
    
    
    from neuropy.core.session.Formats.SessionSpecifications import SessionConfig
    
    """
    basepath: Path = field()
    session_spec: SessionFolderSpec = field()
    session_name: str = field()
    session_context: IdentifyingContext = field()
    format_name: str = field()
    preprocessing_parameters: ParametersContainer = field()

    # Derived properties:
    absolute_start_timestamp: float = field(default=603785.852737) # init=False, 
    position_sampling_rate_Hz: float = field(default=29.96977250291495) # init=False, 

    microseconds_to_seconds_conversion_factor: float = field(default=1e-06)

    pix2cm: float = field(default=287.7697841726619) # init=False, 
    x_midpoint: float = field(default=143.8848920863310)
    loaded_track_limits: dict = field(default=Factory(dict))

    is_resolved: bool = field(default=False) # init=False, 
    resolved_required_filespecs_dict: dict = field(default=Factory(dict)) # , init=False
    resolved_optional_filespecs_dict: dict = field(default=Factory(dict)) # , init=False


    @property
    def resolved_required_file_specs(self):
        """The resolved_required_file_specs property."""
        return {a_filepath:(lambda sess, filepath=a_filepath: a_spec.session_load_callback(filepath, sess)) for a_filepath, a_spec in self.resolved_required_filespecs_dict.items()}
        
    @property
    def resolved_optional_file_specs(self):
        """The resolved_required_file_specs property."""
        return {a_filepath:(lambda sess, filepath=a_filepath: a_spec.session_load_callback(filepath, sess)) for a_filepath, a_spec in self.resolved_optional_filespecs_dict.items()}
    

    def __attrs_post_init__(self):
        # Computed variables:
        self.is_resolved, self.resolved_required_filespecs_dict, self.resolved_optional_filespecs_dict = self.session_spec.validate(self.basepath)


    def validate(self):
        """ re-validates the self.session_spec items and updates the resolved dicts """
        self.is_resolved, self.resolved_required_filespecs_dict, self.resolved_optional_filespecs_dict = self.session_spec.validate(self.basepath)


    def to_dict(self):
        out_dict = {a_key:str(a_value) for a_key, a_value in self.__dict__.items() if a_key in ['format_name', 'basepath', 'session_name', 'session_context', 'absolute_start_timestamp', 'position_sampling_rate_Hz', 'pix2cm', 'x_midpoint']}
        # need to flatten: 'resolved_required_filespecs_dict', 'resolved_optional_filespecs_dict':
        out_dict['resolved_required_filespecs_dict'] = [str(a_path) for a_path in self.resolved_required_file_specs.keys()]
        out_dict['resolved_optional_filespecs_dict'] = [str(a_path) for a_path in self.resolved_optional_file_specs.keys()]

        out_dict['loaded_track_limits'] = {str(k):v for k,v in self.loaded_track_limits.items()}

        return out_dict
    
    # Context and Description ____________________________________________________________________________________________ #
    def get_context(self):
        """ returns an IdentifyingContext for the session """
        if self.session_context is not None:
            return self.session_context
        else:
            from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
            ## Tries to get the appropriate class using its self.format_name and compute its context
            active_data_mode_registered_class = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()[self.format_name]
            self.session_context = active_data_mode_registered_class.parse_session_basepath_to_context(self.basepath)
            return self.session_context
            # return IdentifyingContext(format_name=self.format_name, session_name=self.session_name)

    def get_description(self, prefix_items=['sess'])->str:
        """ returns a simple text descriptor of the session
        Outputs:
            a str like 'sess_kdiba_2006-6-07_11-26-53'
        """
        ## Build a session descriptor string:
        session_descriptor_string = self.get_context().get_description(separator='_', include_property_names=False, replace_separator_in_property_names='-', prefix_items=prefix_items)
        return session_descriptor_string
    
    def __str__(self) -> str:
        return self.get_description()
    
    
    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes (_mapping and _keys_at_init). Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # state = self.to_dict().copy()
        # Remove the unpicklable entries.
        # del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes (i.e., _mapping and _keys_at_init).
        # print(f'SessionConfig.__setstate__(state: {state})')
        if 'session_context' not in state:
            from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
            # self.session_context = None
            state['session_context'] = None
            ## Tries to get the appropriate class using its self.format_name and compute its context
            active_data_mode_registered_class = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()[state['format_name']]
            state['session_context'] = active_data_mode_registered_class.parse_session_basepath_to_context(state['basepath'])

        if 'loaded_track_limits' not in state:
            state['loaded_track_limits'] = dict()
        
        
        self.__dict__.update(state)
        

        
