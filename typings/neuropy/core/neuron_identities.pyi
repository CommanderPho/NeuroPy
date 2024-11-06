"""
This type stub file was generated by pyright.
"""

import pandas as pd
import tables as tb
from enum import Enum, unique
from functools import total_ordering
from typing import List
from pandas import CategoricalDtype
from neuropy.utils.mixins.HDF5_representable import HDFConvertableEnum
from neuropy.utils.mixins.print_helpers import SimplePrintable
from attrs import define
from neuropy.utils.mixins.indexing_helpers import UnpackableMixin

NeuronExtendedIdentityTuple = ...
@define(slots=False, eq=False)
class NeuronExtendedIdentity(UnpackableMixin):
    """ 
    from neuropy.core.neuron_identities import NeuronExtendedIdentity
    
    """
    shank: int = ...
    cluster: int = ...
    aclu: int = ...
    qclu: int = ...
    @property
    def id(self) -> int:
        """Provided for compatibility with NeuronExtendedIdentityTuple """
        ...
    
    @id.setter
    def id(self, value): # -> None:
        ...
    
    @classmethod
    def init_from_NeuronExtendedIdentityTuple(cls, a_tuple, qclu=...): # -> Self:
        """ # NeuronExtendedIdentityTuple """
        ...
    
    def __getstate__(self): # -> dict[str, Any]:
        ...
    
    def __setstate__(self, state): # -> None:
        ...
    


neuronTypesList: List[str] = ...
neuronTypesEnum = ...
class NeuronIdentityTable(tb.IsDescription):
    """ represents a single neuron in the scope of multiple sessions for use in a PyTables table or HDF5 output file """
    neuron_uid = ...
    session_uid = ...
    neuron_id = ...
    neuron_type = ...
    shank_index = ...
    cluster_index = ...
    qclu = ...


@pd.api.extensions.register_dataframe_accessor("neuron_identity")
class NeuronIdentityDataframeAccessor:
    """ Describes a dataframe with at least a neuron_id (aclu) column. Provides functionality regarding building globally (across-sessions) unique neuron identifiers.
    
    #TODO 2023-08-22 15:34: - [ ] Finish implementation. Purpose is to easily add across-session-unique neuron identifiers to a result dataframe (as many result dataframes have an 'aclu' column).
        - [ ] find already implemented 'aclu' conversions, like for the JonathanFiringRateResult (I think)
        
    """
    def __init__(self, pandas_obj) -> None:
        ...
    
    @property
    def neuron_ids(self):
        """ return the unique cell identifiers (given by the unique values of the 'aclu' column) for this DataFrame """
        ...
    
    @property
    def neuron_probe_tuple_ids(self): # -> list[NeuronExtendedIdentity]:
        """ returns a list of NeuronExtendedIdentityTuple tuples where the first element is the shank_id and the second is the cluster_id. Returned in the same order as self.neuron_ids """
        ...
    
    @property
    def n_neurons(self): # -> int:
        ...
    
    def extract_unique_neuron_identities(self):
        """ Tries to build information about the unique neuron identitiies from the (highly reundant) information in the spikes_df. """
        ...
    
    def make_neuron_indexed_df_global(self, curr_session_context: IdentifyingContext, add_expanded_session_context_keys: bool = ..., add_extended_aclu_identity_columns: bool = ..., inplace: bool = ...) -> pd.DataFrame:
        """ 2023-10-04 - Builds session-relative neuron identifiers, adding the global columns to the neuron_indexed_df

        Usage:
            from neuropy.core.neuron_identities import NeuronIdentityDataframeAccessor

            curr_session_context = curr_active_pipeline.get_session_context()
            input_df = neuron_replay_stats_df.copy()
            input_df = input_df.neuron_identity.make_neuron_indexed_df_global(curr_active_pipeline.get_session_context(), add_expanded_session_context_keys=False, add_extended_aclu_identity_columns=False)
            input_df

        """
        ...
    


class NeuronIdentity(SimplePrintable):
    """NeuronIdentity: A multi-facited identifier for a specific neuron/putative cell 
        Used to retain a the identity associated with a value or set of values even after filtering and such.

        cell_uid: (aclu) [2:65]
        shank_index: [1:12]
        cluster_index: [2:28]
        
        NOTE: also store 'color'
        ['shank','cluster']
    
    """
    @property
    def extended_identity_tuple(self): # -> NeuronExtendedIdentity:
        """The extended_identity_tuple property."""
        ...
    
    @extended_identity_tuple.setter
    def extended_identity_tuple(self, value): # -> None:
        ...
    
    @property
    def extended_id_string(self):
        """The extended_id_string property."""
        ...
    
    def __init__(self, cell_uid, shank_index, cluster_index, qclu, color=...) -> None:
        ...
    
    @classmethod
    def init_from_NeuronExtendedIdentity(cls, a_tuple: NeuronExtendedIdentity, a_color=...): # -> Self:
        """Iniitalizes from a NeuronExtendedIdentityTuple and optionally a color
        Args:
            a_tuple (NeuronExtendedIdentityTuple): [description]
            a_color ([type], optional): [description]. Defaults to None.
        """
        ...
    


class NeuronIdentityAccessingMixin:
    """ 
        Requires implementor overrides the neuron_ids property to provide an ordered list of unique cell identifiers (such as the 'aclu' values from a spikes_df)
        
        provides functions to map between unique cell_identifiers (cell_ids) and implementor specific indicies (neuron_IDXs)
    
        NOTE: 
            Based on how the neuron_IDXs are treated in self.get_neuron_id_and_idx(...), they are constrained to be:
                1. Monotonically increasing from 0 to (len(self.neuron_ids)-1)
                
                Note that if an implementor violates this definition, for example if they filtered out or excluded some of the neurons and so were left with fewer self.neuron_ids than were had when the self.neuron_IDXs were first built, then there can potentially be:
                    1. Indicies missing from the neuronIDXs (corresponding to the filtered out neuron_ids)
                    2. Too many indicies present in neuronIDXs (with the extras corresponding to the neuron_ids that were removed after the IDXs were built).
                    3. **IMPORTANT**: Values of neuronIDXs that are too large and would cause index out of bound errors when trying to get the corresponding to neuron_id value.
                    4. **CRITICAL**: VALUE SHIFTED reverse lookups! If any neuronIDX is removed with its corresponding neuron_id, it will cause all the neuron_IDXs after it to be 1 value too large and throw off reverse lookups. This is what's happening with the placefields/spikes getting shifted!
    
    
    CONCLUSIONS:
        Implementor must be sure to keep self.neuron_ids up-to-date with any other list of neuron_ids it might use (like the 'aclu' values from the spikes_df) AND be sure to not hold references to (or keep them up-to-date) the neuron_IDXs. Any time IDXs are used (such as those retrieved from the spikes_df's neuron_IDX column) they must be up-to-date to be referenced.
        
    """
    @property
    def neuron_ids(self):
        """ e.g. return np.array(active_epoch_placefields2D.cell_ids) """
        ...
    
    def get_neuron_id_and_idx(self, neuron_i=..., neuron_id=...): # -> tuple[int | None, Any | int | None]:
        """For a specified neuron_i (index) or neuron_id, returns the other quanity (or both)

        Args:
            neuron_i ([type], optional): [description]. Defaults to None.
            neuron_id ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        ...
    
    def find_cell_ids_from_neuron_IDXs(self, neuron_IDXs): # -> list[Any | int | None]:
        """Finds the cell original IDs from the cell IDXs (not IDs)
        Args:
            neuron_IDXs ([type]): [description]
        """
        ...
    
    def find_neuron_IDXs_from_cell_ids(self, cell_ids): # -> list[int | None]:
        """Finds the cell IDXs (not IDs) from the cell original IDs (cell_ids)
        Args:
            cell_ids ([type]): [description]
        """
        ...
    


@unique
class PlotStringBrevityModeEnum(HDFConvertableEnum, Enum):
    """An enum of different modes that specify how verbose/brief the rendered strings should be on a given plot.
    More verbose means longer ouptuts with fewer abbreviations. For very brief modes, less important elements may be omitted entirely
    """
    VERBOSE = ...
    DEFAULT = ...
    CONCISE = ...
    MINIMAL = ...
    NONE = ...
    @property
    def extended_identity_labels(self): # -> dict[str, str]:
        """The extended_identity_labels property."""
        ...
    
    def extended_identity_formatting_string(self, neuron_extended_id): # -> str:
        """The extended_identity_labels property."""
        ...
    
    @property
    def should_show_firing_rate_label(self): # -> bool:
        """ Whether the firing rate in Hz should be showed on the plot """
        ...
    
    @property
    def hdfcodingClassName(self) -> str:
        ...
    
    @classmethod
    def hdf_coding_ClassNames(cls): # -> NDArray[Any]:
        ...
    
    @classmethod
    def get_pandas_categories_type(cls) -> CategoricalDtype:
        ...
    
    @classmethod
    def convert_to_hdf(cls, value) -> str:
        ...
    
    @classmethod
    def from_hdf_coding_string(cls, string_value: str) -> PlotStringBrevityModeEnum:
        ...
    


def build_units_colormap(neuron_ids): # -> tuple[NDArray[Any], ndarray[Any, Any], ndarray[Any, Any], ListedColormap]:
    """ 
    Usage:
        from neuropy.core.neuron_identities import build_units_colormap
        
        pf_sort_ind, pf_colors, pf_colormap, pf_listed_colormap = build_units_colormap(good_placefield_neuronIDs)
    """
    ...

class NeuronIdentitiesDisplayerMixin:
    @property
    def neuron_ids(self):
        """ like self.neuron_ids """
        ...
    
    @property
    def neuron_extended_ids(self):
        """ list of NeuronExtendedIdentityTuple named tuples) like tuple_neuron_ids """
        ...
    
    @property
    def neuron_shank_ids(self): # -> list[Any]:
        ...
    
    @property
    def neuron_cluster_ids(self): # -> list[Any]:
        ...
    
    @property
    def neuron_qclu_ids(self): # -> list[Any]:
        ...
    
    def get_extended_neuron_id_string(self, neuron_i=..., neuron_id=...): # -> str:
        ...
    
    def other_neuron_id_string(self, neuron_i):
        ...
    


@total_ordering
@unique
class NeuronType(HDFConvertableEnum, Enum):
    """
    Kamran 2023-07-18:
        cluq=[1,2,4,9] all passed.
        3 were noisy
        [6,7]: double fields.
        5: interneurons

    Pho-Pre-2023-07-18:
        pyramidal: [-inf, 4)
        contaminated: [4, 7)
        interneurons: [7, +inf)
    """
    PYRAMIDAL = ...
    CONTAMINATED = ...
    INTERNEURONS = ...
    def describe(self): # -> None:
        ...
    
    def __eq__(self, other) -> bool:
        ...
    
    def __le__(self, other) -> bool:
        ...
    
    def __hash__(self) -> int:
        ...
    
    @property
    def shortClassName(self): # -> Any:
        ...
    
    @property
    def longClassName(self): # -> Any:
        ...
    
    @property
    def hdfcodingClassName(self) -> str:
        ...
    
    @classmethod
    def longClassNames(cls): # -> NDArray[Any]:
        ...
    
    @classmethod
    def shortClassNames(cls): # -> NDArray[Any]:
        ...
    
    @classmethod
    def bapunNpyFileStyleShortClassNames(cls): # -> NDArray[Any]:
        ...
    
    @classmethod
    def hdf_coding_ClassNames(cls): # -> NDArray[Any]:
        ...
    
    @classmethod
    def classCutoffValues(cls):
        ...
    
    @classmethod
    def classCutoffMap(cls) -> dict:
        """ For each qclu value in 0-10, return a str in ['pyr','cont','intr']
            Kamran 2023-07-18:
                cluq=[1,2,4,9] all passed.
                3 were noisy
                [6,7]: double fields.
                5: interneurons

        ## Post 2023-07-18:
            for i in np.arange(10):
                _out_map[i] = "cont" # initialize all to 'contaminated'/noisy
            for i in [1,2,4,6,7,9]:
                _out_map[i] = 'pyr' # pyramidal
            _out_map[5] = "intr" # interneurons

        ## Post 2023-07-31 - Excluding "double fields" qclues on Kamran's request
        ## 2023-12-07 12:22: - [ ] Kamran wants me to exclude [6, 7]
        ## 2023-12-07 20:59: - [ ] Updated to only include [1,2,4,9] as pyramidal
		## 2024-10-25 - Including [1,2,4,6,7,9]
        """
        ...
    
    @classmethod
    def from_short_string(cls, string_value) -> NeuronType:
        ...
    
    @classmethod
    def from_long_string(cls, string_value) -> NeuronType:
        ...
    
    @classmethod
    def from_string(cls, string_value) -> NeuronType:
        ...
    
    @classmethod
    def from_bapun_npy_style_string(cls, string_value) -> NeuronType:
        ...
    
    @classmethod
    def from_qclu_series(cls, qclu_Series): # -> NDArray[Any]:
        ...
    
    @classmethod
    def from_any_string_series(cls, neuron_types_strings): # -> NDArray[Any]:
        ...
    
    @classmethod
    def from_bapun_npy_style_series(cls, bapun_style_neuron_types): # -> NDArray[Any]:
        ...
    
    @classmethod
    def from_hdf_coding_style_series(cls, hdf_coding_neuron_types): # -> NDArray[Any]:
        ...
    
    @classmethod
    def classRenderColors(cls): # -> NDArray[Any]:
        """ colors used to render each type of neuron """
        ...
    
    @property
    def renderColor(self): # -> Any:
        ...
    
    @classmethod
    def get_pandas_categories_type(cls) -> CategoricalDtype:
        ...
    
    @classmethod
    def convert_to_hdf(cls, value) -> str:
        ...
    
    @classmethod
    def from_hdf_coding_string(cls, string_value: str) -> NeuronType:
        ...
    


