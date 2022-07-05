from dataclasses import dataclass
from typing import Callable
import warnings
import numpy as np
import pandas as pd
from pathlib import Path


# Local imports:
## Core:
# from .datawriter import DataWriter
# from .neurons import NeuronType, Neurons, BinnedSpiketrain, Mua
# from .probe import ProbeGroup
# from .position import Position
# from .epoch import Epoch #, NamedTimerange
# from .signal import Signal
# from .laps import Laps
# from .flattened_spiketrains import FlattenedSpiketrains

# from .. import DataWriter, NeuronType, Neurons, BinnedSpiketrain, Mua, ProbeGroup, Position, Epoch, Signal, Laps, FlattenedSpiketrains

from neuropy.core.session.dataSession import DataSession
from neuropy.utils.mixins.print_helpers import ProgressMessagePrinter, SimplePrintable, OrderedMeta


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
                    # print('WARNING: Optional File: {} does not exist.'.format(an_optional_file))
                    warnings.warn(f'WARNING: Optional File: {an_optional_filepath} does not exist. Continuing without it.')
                    
            meets_spec = True # otherwise it exists
            
        return meets_spec, resolved_required_filespecs_dict, resolved_optional_filespecs_dict
    

# session_name = '2006-6-07_11-26-53'
# SessionFolderSpec(required=['{}.xml'.format(session_name),
#                             '{}.spikeII.mat'.format(session_name), 
#                             '{}.position_info.mat'.format(session_name),
#                             '{}.epochs_info.mat'.format(session_name), 
# ])


class SessionConfig(SimplePrintable, metaclass=OrderedMeta):
    """A simple data structure that holds the information specifying a data session, such as the basepath, session_spec, and session_name
    """
    
    @property
    def resolved_required_file_specs(self):
        """The resolved_required_file_specs property."""
        return {a_filepath:(lambda sess, filepath=a_filepath: a_spec.session_load_callback(filepath, sess)) for a_filepath, a_spec in self.resolved_required_filespecs_dict.items()}
        
    @property
    def resolved_optional_file_specs(self):
        """The resolved_required_file_specs property."""
        return {a_filepath:(lambda sess, filepath=a_filepath: a_spec.session_load_callback(filepath, sess)) for a_filepath, a_spec in self.resolved_optional_filespecs_dict.items()}
    
    
    def __init__(self, basepath, session_spec, session_name):
        """[summary]
        Args:
            basepath (pathlib.Path): [description].
            session_spec (SessionFolderSpec): used to load the files
            session_name (str, optional): [description].
        """
        self.basepath = basepath
        self.session_name = session_name
        # Session spec:
        self.session_spec=session_spec
        self.is_resolved, self.resolved_required_filespecs_dict, self.resolved_optional_filespecs_dict = self.session_spec.validate(self.basepath)
        
    def validate(self):
        """ re-validates the self.session_spec items and updates the resolved dicts """
        self.is_resolved, self.resolved_required_filespecs_dict, self.resolved_optional_filespecs_dict = self.session_spec.validate(self.basepath)


    def to_dict(self):
        out_dict = {a_key:str(a_value) for a_key, a_value in self.__dict__.items() if a_key in ['basepath', 'session_name', 'absolute_start_timestamp', 'position_sampling_rate_Hz']}
        # need to flatten: 'resolved_required_filespecs_dict', 'resolved_optional_filespecs_dict':
        out_dict['resolved_required_filespecs_dict'] = [str(a_path) for a_path in self.resolved_required_file_specs.keys()]
        out_dict['resolved_optional_filespecs_dict'] = [str(a_path) for a_path in self.resolved_optional_file_specs.keys()]
        return out_dict
    
