""" result_context.py

The general goal of these context objects is to be able to generate distinct identifiers for a given result so that it can be serialized to disk and analyzed later.

To do this systematically, we need to be able to identify



For human purposes, it's also incredibly useful to be able to generate minimal-difference versions of these contexts.

For example, if you generated two figures, one with a spatial bin size of 0.5 and another of 1.0 when generating titles for the two plots those two levels of the spatial bin size variable are all that you'd want to keep straight. The rest of the variables aren't changing, and so the minimal-difference required to discriminate between the two cases is only the levels of that variables.
{'bin_size': [0.5, 1.0]}

Let's say you save these out to disk and then a week later see that applying the exact same analyses (with those same two levels) to a different dataset produces an unexpected result at a value of 0.5. You don't remember such a wonky looking graph of 0.5 for the previous dataset, and want to figure out what's going wrong. 

To do this you'll now want to compare the 4 graphs, meaning you'll need to keep track:

Now you have two different datasets: 'earlier' and 'now', each with a minimal-difference of:
    'earlier': {'bin_size': [0.5, 1.0]}
    'now': {'bin_size': [0.5, 1.0]}

To uniquely identify them you'll probably want to visualize them as:

'earlier' (row):    | 'bin_size'=0.5 | 'bin_size'=1.0 | 
'now' (row):        | 'bin_size'=0.5 | 'bin_size'=1.0 |

Ultimately you only have two axes over which things can be compared (rows and columns)
There's one hidden one: 'tabs', 'windows'

Humans need things with distinct, visual groupings. Inclusion Sets, Exceptions (a single outlier rendered in juxtaposition to an inclusion set of the norms)

"""

import copy
from benedict import benedict # https://github.com/fabiocaccamo/python-benedict#usage
from neuropy.utils.mixins.diffable import DiffableObject


class IdentifyingContext(DiffableObject, object):
    """ a general extnsible base context that allows additive member creation
    
        Should not hold any state or progress-related variables. 
    
    """
    def __init__(self, **kwargs):
        super(IdentifyingContext, self).__init__()
        ## Sets attributes dnymically:
        for name, value in kwargs.items():
            setattr(self, name, value)
        
    def add_context(self, collision_prefix:str, **additional_context_items):
        """ adds the additional_context_items to self 
        collision_prefix: only used when an attr name in additional_context_items already exists for this context 
        
        """
        for name, value in additional_context_items.items():
            # TODO: ensure no collision between attributes occur, and if they do rename them with an identifying prefix
            if hasattr(self, name):
                print(f'WARNING: namespace collision in add_context! attr with name {name} already exists!')
                ## TODO: rename the current attribute to be set by appending a prefix
                final_name = f'{collision_prefix}{name}'
            else:
                final_name = name
            # Set the new attr
            setattr(self, final_name, value)
        
        return self


    def adding_context(self, collision_prefix:str, **additional_context_items):
        """ returns a new IdentifyingContext that results from adding additional_context_items to a copy of self 
        collision_prefix: only used when an attr name in additional_context_items already exists for this context 
        
        """
        duplicate_ctxt = copy.deepcopy(self)
        
        for name, value in additional_context_items.items():
            # TODO: ensure no collision between attributes occur, and if they do rename them with an identifying prefix
            if hasattr(duplicate_ctxt, name):
                print(f'WARNING: namespace collision in add_context! attr with name {name} already exists!')
                ## TODO: rename the current attribute to be set by appending a prefix
                assert collision_prefix is not None, f"namespace collision in add_context! attr with name {name} already exists but collision_prefix is None!"
                final_name = f'{collision_prefix}{name}'
            else:
                final_name = name
            # Set the new attr
            setattr(duplicate_ctxt, final_name, value)
        
        return duplicate_ctxt
                       
                       
        
    def get_description(self, subset_whitelist=None, separator:str='_', include_property_names:bool=False, replace_separator_in_property_names:str='-', prefix_items=[], suffix_items=[])->str:
        """ returns a simple text descriptor of the context
        
        include_property_names: str - whether to include the keys/names of the properties in the output string or just the values
        replace_separator_in_property_names: str = replaces occurances of the separator with the str specified for the included property names. has no effect if include_property_names=False
        
        Outputs:
            a str like 'sess_kdiba_2006-6-07_11-26-53'
        """
        ## Build a session descriptor string:
        if include_property_names:
            descriptor_array = [[name.replace(separator, replace_separator_in_property_names), str(val)]  for name, val in self.to_dict(subset_whitelist=subset_whitelist).items()] # creates a list of [name, val] list items
            descriptor_array = [item for sublist in descriptor_array for item in sublist] # flat descriptor array
        else:
            descriptor_array = [str(val) for val in list(self.to_dict(subset_whitelist=subset_whitelist).values())] # ensures each value is a string
            
        if prefix_items is not None:
            descriptor_array.extend(prefix_items)
        if suffix_items is not None:
            descriptor_array.extend(suffix_items)
        
        descriptor_string = separator.join(descriptor_array)
        return descriptor_string
    
    def __str__(self) -> str:
        """ 'kdiba_2006-6-08_14-26-15_maze1_PYR' """
        return self.get_description()


    def __repr__(self) -> str:
        """ 
            "IdentifyingContext({'format_name': 'kdiba', 'session_name': '2006-6-08_14-26-15', 'filter_name': 'maze1_PYR'})" 
            "IdentifyingContext<('kdiba', '2006-6-08_14-26-15', 'maze1_PYR')>"
        """
        # return f"IdentifyingContext({self.get_description(include_property_names=True)})"
        # return f"IdentifyingContext({self.get_description(include_property_names=False)})"
        return f"IdentifyingContext<{self.as_tuple().__repr__()}>"
        # return f"IdentifyingContext({self.to_dict().__repr__()})"
    def __hash__(self):
        """ custom hash function that allows use in dictionary just based off of the values and not the object instance. """
        dict_rep = self.to_dict()
        member_names_tuple = list(dict_rep.keys())
        values_tuple = list(dict_rep.values())
        combined_tuple = tuple(member_names_tuple + values_tuple)
        return hash(combined_tuple)
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, IdentifyingContext):
            return self.to_dict() == other.to_dict() # Python's dicts use element-wise comparison by default, so this is what we want.
        return NotImplemented
    
    
    def to_dict(self, subset_whitelist=None):
        """ 
        Inputs:
            subset_whitelist:<list?> a list of keys that specify the subset of the keys to be returned. If None, all are returned.
        """
        if subset_whitelist is None:
            return benedict(self.__dict__)
        else:
            return benedict(self.__dict__).subset(subset_whitelist)

    @classmethod
    def init_from_dict(cls, a_dict):
        return cls(**a_dict) # expand the dict as input args.
        

    def as_tuple(self, subset_whitelist=None, drop_missing:bool=False):
        """ returns a tuple of just its values 
        Inputs:
            subset_whitelist:<list?> a list of keys that specify the subset of the keys to be returned. If None, all are returned.

        Usage:
        curr_sess_ctx_tuple = curr_sess_ctx.as_tuple(subset_whitelist=['format_name','animal','exper_name', 'session_name'])
        curr_sess_ctx_tuple # ('kdiba', 'gor01', 'one', '2006-6-07_11-26-53')

        """
        if drop_missing:
            return tuple([v for v in tuple(self.to_dict(subset_whitelist=subset_whitelist).values()) if v is not None]) # Drops all 'None' values in the tuple
        else:
            return tuple(self.to_dict(subset_whitelist=subset_whitelist).values())


    def has_keys(self, keys_list):
        """ returns a boolean array with each entry indicating whether that element in keys_list was found in the context """
        is_key_found = [(v is not None) for v in self.as_tuple(subset_whitelist=keys_list)]
        return is_key_found

    def check_keys(self, keys_list, debug_print=False):
        """ checks whether it has the keys or not
        Usage:
            all_keys_found, found_keys, missing_keys = curr_sess_ctx.check_keys(['format_name','animal','exper_name', 'session_name'], debug_print=False)
        """
        is_key_found = self.has_keys(keys_list)

        found_keys = [k for k, is_found in zip(keys_list, is_key_found) if is_found]
        missing_keys = [k for k, is_found in zip(keys_list, is_key_found) if not is_found]

        all_keys_found = (len(missing_keys) == 0)
        if not all_keys_found:
            if debug_print:
                print(f'missing {len(missing_keys)} keys: {missing_keys}')
        else:
            if debug_print:
                print(f'found all {len(found_keys)} keys: {found_keys}')
        return all_keys_found, found_keys, missing_keys


    ## For serialization/pickling:
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        # del state['file']
        return state

    def __setstate__(self, state):
        # Restore instance attributes.
        self.__dict__.update(state)
        
        
        
# class ResultContext(IdentifyingContext):
#     """result_context serves to uniquely identify the **context** of a given generic result.

#     Typically a result depends on several inputs:

#     - session: the context in which the original recordings were made.
#         Originally this includes the circumstances udner which the recording was performed (recording datetime, recording configuration (hardware, sampling rate, etc), experimenter name, animal identifer, etc)
#     - filter: the filtering performed to pre-process the loaded session data
#     - computation configuration: the specific computations performed and their parameters to transform the data to the result

#     Heuristically: If changing the value of that variable results in a changed result, it should be included in the result_context 
#     """
#     session_context: str = ''
#     filter_context: str = ''
#     computation_configuration: str = ''
    
 
#     def __init__(self, session_context, filter_context, computation_configuration):
#         super(ResultContext, self).__init__()
#         self.session_context = session_context
#         self.filter_context = filter_context
#         self.computation_configuration = computation_configuration
    
    