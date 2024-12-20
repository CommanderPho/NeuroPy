"""
This type stub file was generated by pyright.
"""

from typing import Any, Dict, Optional

class GetAccessibleMixin:
    """ Implementors provide a default `get('an_attribute', a_default)` option so they can be accessed like dictionaries via passthrough instead of having to use getattr(....) 
    
    from neuropy.utils.mixins.gettable_mixin import GetAccessibleMixin
    
    History: 2024-10-23 11:36 Refactored from pyphocorehelpers.mixins.gettable_mixin.GetAccessibleMixin 

    """
    def get(self, attribute_name: str, default: Optional[Any] = ...) -> Optional[Any]:
        """ Use the getattr built-in function to retrieve attributes """
        ...
    


class KeypathsAccessibleMixin:
    """ implementors support benedict-like keypath indexing and updating

    from neuropy.utils.mixins.gettable_mixin import KeypathsAccessibleMixin
    

    """
    def get_by_keypath(self, keypath: str) -> Any:
        """Gets the value at the specified keypath.
        Usage:
            # Get a value using keypath
            value = params.get_by_keypath('directional_train_test_split.training_data_portion')
            print(value)  # Output: 0.8333333333333334
                        
        """
        ...
    
    def set_by_keypath(self, keypath: str, new_value: Any): # -> None:
        """Sets the value at the specified keypath.
        Usage:
            # Set a value using keypath
            params.set_by_keypath('directional_train_test_split.training_data_portion', 0.9)
        """
        ...
    
    def keypaths(self) -> list:
        """Returns all keypaths in the nested attrs classes.

        Usage:
            # Get all keypaths
            all_keypaths = params.keypaths()
            print(all_keypaths)     
        """
        ...
    
    @classmethod
    def keypath_dict_to_nested_dict(cls, keypath_dict: Dict[str, Any]) -> Dict:
        """ converts a flat dict of keypath:value -> nested dicts of keys """
        ...
    


