"""
This type stub file was generated by pyright.
"""

from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Tuple
from attrs import field

def keys_only_repr(instance): # -> str:
    """ specifies that this field only prints its .keys(), not its values.
    
    # Usage (within attrs class):
        computed_data: Optional[DynamicParameters] = serialized_field(default=None, repr=keys_only_repr)
        accumulated_errors: Optional[DynamicParameters] = non_serialized_field(default=Factory(DynamicParameters), is_computable=True, repr=keys_only_repr)
    
    """
    ...

@unique
class HDF_SerializationType(Enum):
    """ Specifies how a serialized field is stored, as an HDF5 Dataset or Attribute """
    DATASET = ...
    ATTRIBUTE = ...
    @property
    def required_tag(self): # -> Any:
        ...
    
    @classmethod
    def requiredClassTags(cls): # -> NDArray[Any]:
        ...
    


class AttrsBasedClassHelperMixin:
    """ heleprs for classes defined with `@define(slots=False, ...)` 
    
    from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define


    hdf_fields = BasePositionDecoder.get_serialized_dataset_fields('hdf')

    """
    @classmethod
    def get_fields_with_tag(cls, tag: str = ..., invert: bool = ...) -> Tuple[List, Callable]:
        ...
    
    @classmethod
    def get_serialized_fields(cls, serializationType: HDF_SerializationType, serialization_format: str = ...) -> Tuple[List, Callable]:
        """ general function for getting the list of fields with a certain serializationType as a list of attrs attributes and a filter to select them useful for attrs.asdict(...) filtering. """
        ...
    
    @classmethod
    def get_serialized_dataset_fields(cls, serialization_format: str = ...) -> Tuple[List, Callable]:
        ...
    
    @classmethod
    def get_serialized_attribute_fields(cls, serialization_format: str = ...) -> Tuple[List, Callable]:
        ...
    
    def to_dict(self) -> Dict:
        ...
    


custom_define = ...
def merge_metadata(default_metadata: Dict[str, Any], additional_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    ...

def non_serialized_field(default: Optional[Any] = ..., is_computable: bool = ..., metadata: Optional[Dict[str, Any]] = ..., **kwargs) -> field:
    ...

def serialized_field(default: Optional[Any] = ..., is_computable: bool = ..., serialization_fn: Optional[Callable] = ..., is_hdf_handled_custom: bool = ..., hdf_metadata: Optional[Dict] = ..., metadata: Optional[Dict[str, Any]] = ..., **kwargs) -> field:
    ...

def serialized_attribute_field(default: Optional[Any] = ..., is_computable: bool = ..., serialization_fn: Optional[Callable] = ..., metadata: Optional[Dict[str, Any]] = ..., **kwargs) -> field:
    """ marks a specific field to be serialized as an HDF5 attribute on the group for this object """
    ...

class SimpleFieldSizesReprMixin:
    """ Defines the __repr__ for implementors that only renders the implementors fields and their sizes

    from neuropy.utils.mixins.AttrsClassHelpers import SimpleFieldSizesReprMixin

    
    """
    def __repr__(self): # -> str:
        """ 2024-01-11 - Renders only the fields and their sizes
        """
        ...
    


_default_class_name_dict_replace = ...
def convert_attrs_inline_class_instance_to_normal_class_defn(an_instance, class_name_replace_dict=..., print_output: bool = ...) -> str:
    """ Converts an inline `attrs`-based class definition generated via `attrs.make_class(...)` to a full-class-defn-form class definition.
    
     For example:
        ```python
        HeuristicScoresTuple = attrs.make_class("HeuristicScoresTuple", {k:field() for k in ("longest_sequence_length", "longest_sequence_length_ratio", "direction_change_bin_ratio", "congruent_dir_bins_ratio", "total_congruent_direction_change", 
                                                                                            "total_variation", "integral_second_derivative", "stddev_of_diff",
                                                                                            "position_derivatives_df")}, bases=(UnpackableMixin, object,))
        ```

        ```python
        class HeuristicScoresTuple(UnpackableMixin, object,):
            longest_sequence_length = field()
            longest_sequence_length_ratio = field()
            # ... remainder of fields
        ```


    Usage:

        from neuropy.utils.mixins.AttrsClassHelpers import convert_attrs_inline_class_instance_to_normal_class_defn
        content = convert_attrs_inline_class_instance_to_normal_class_defn(an_instance=deepcopy(active_heuristic_scores))
        print(content)
    
    """
    ...

