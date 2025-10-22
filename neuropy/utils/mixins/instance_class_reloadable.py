from typing import Dict, List, Tuple, Optional, Callable, Union, Any
from typing_extensions import TypeAlias
from nptyping import NDArray
from neuropy.utils.mixins.indexing_helpers import get_dict_subset


class InstanceClassReloadableMixin:
    """ Instances of an implementor class can be dynamically reloaded to the latest version of the class while running.

    from neuropy.utils.mixins.instance_class_reloadable import InstanceClassReloadableMixin


    """
    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains all our instance attributes. Always use the dict.copy() method to avoid modifying the original state.
        state = self.__dict__.copy()
        return state

    @classmethod
    def _reload_class(cls, an_instance, an_updated_class=None):
        """ specifically updates the instance after its class definition has been updated.
        """
        if an_updated_class is not None:
            an_updated_class = cls
        return an_updated_class(**get_dict_subset(an_instance.__getstate__(), subset_excludelist=['_VersionedResultMixin_version']))


    def _reload_class_defn(self, an_updated_class=None):
        """ specifically updates the instance after its class definition has been updated.
        """
        ## get newest class import
        if an_updated_class is not None:
            an_updated_class = self.__class__
        
        return an_updated_class(**get_dict_subset(self.__getstate__(), subset_excludelist=['_VersionedResultMixin_version']))










