from copy import deepcopy
from functools import partial
import socket # for getting hostname
from typing import Union, List, Dict, Tuple, Set, Any, Optional, OrderedDict, Callable  # for OrderedMeta
from nptyping import NDArray

from datetime import datetime, date, timedelta # for `get_now_day_str`
import time # for `get_now_time_str`, `get_now_time_precise_str`
import numpy as np
import pandas as pd
from pandas.core.resample import TimedeltaIndexResampler

# Required for dbg_dump:
import sys
import pprint
import inspect
import ast

import site # Required for StackTraceFormatting
from os.path import join, abspath # Required for StackTraceFormatting
from traceback import extract_tb, format_list, format_exception_only # Required for StackTraceFormatting
from attrs import define, field, Factory

import re ## required for strip_type_str_to_classname(...)
from pathlib import Path
import logging

import itertools

# Required for proper print_object_memory_usage
import objsize # python -m pip install objsize==0.6.1

# from pyphocorehelpers.function_helpers import function_attributes # # function_attributes causes circular import issue :[
import numpy as np
import dask.array as da
from IPython.display import Image, display, HTML

from io import BytesIO
import base64
import ipywidgets as widgets



# @function_attributes(short_name=None, tags=['unused', 'repr', 'str', 'string_representation', 'preview'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-11-28 12:43', related_items=[])
def truncating_list_repr(items, max_full_display_n_items: int = 1000, truncated_num_edge_items: int = 3):
    """ If length is less than `max_full_display_n_items` return the full list 
    https://stackoverflow.com/questions/62884503/what-are-the-best-practices-for-repr-with-collection-class-python
    

    Usage:
        from neuropy.utils.print_helpers import truncating_list_repr

        short_list = [1, 2, 3]
        medium_list = [1, 2, 3, 4, 5, 6]
        long_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(truncating_list_repr(short_list, max_full_display_n_items = 5, truncated_num_edge_items = 3)) # '[1, 2, 3]'
        print(truncating_list_repr(medium_list, max_full_display_n_items = 5, truncated_num_edge_items = 3)) # [1, 2, 3, ..., 4, 5, 6]
        truncating_list_repr(long_list, max_full_display_n_items = 5, truncated_num_edge_items = 3) # '[ 1,  2,  3, ...,  8,  9, 10]'


    """
    if len(items) < max_full_display_n_items:
        return f"{items}"
    else:
        # Get the first and last three items
        if isinstance(items, dict):
            # Convert the dictionary to a list of tuples (key, value)
            dict_items = list(items.items())
            items_to_display = dict_items[:truncated_num_edge_items] + dict_items[-truncated_num_edge_items:]
        else:
            items_to_display = items[:truncated_num_edge_items] + items[-truncated_num_edge_items:]
        # Find the which item has the longest repr
        max_length_repr = max(items_to_display, key=lambda x: len(repr(x)))
        # Get the length of the item with the longest repr
        padding = len(repr(max_length_repr))
        # Create a list of the reprs of each item and apply the padding
        values = [repr(item).rjust(padding) for item in items_to_display]
        # Insert the '...' inbetween the 3rd and 4th item
        values.insert(truncated_num_edge_items, '...')
        # Convert the list to a string joined by commas
        array_as_string = ', '.join(values)
        return f"[{array_as_string}]"


class SimplePrintable:
    """Adds the default print method for classes that displays the class name and its dictionary.
    
    Shouldn't it define __str__(self) instead of __repr__(self)?
    """
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.__dict__};>"
    
class iPythonKeyCompletingMixin:
    """ Enables iPython key completion
    Requires Implementors to provide:
        self.keys()
    """
    def _ipython_key_completions_(self) -> List[Optional[str]]:
        return self.keys()
    

class PrettyPrintable(iPythonKeyCompletingMixin):
    def keys(self) -> List[Optional[str]]:
        return self.__dict__.keys()

    def _repr_pretty_(self, p, cycle=False):
        """The cycle parameter will be true if the representation recurses - e.g. if you put a container inside itself."""
        # p.text(self.__repr__() if not cycle else '...')
        p.text(self.__dict__.__repr__() if not cycle else "...")
        # return self.as_array().__repr__() # p.text(repr(self))

class WrappingMessagePrinter(object):
    """ 
    
    Examples:
        with WrappingMessagePrinter('Saving 2D Placefield image out to "{}"...'.format(active_plot_filepath), begin_line_ending='...', finished_message='done.'):
            for aFig in active_figures:
                aFig.savefig(active_plot_filepath)
    """
    def __init__(self, begin_string, begin_line_ending=' ', finished_message='done.', finished_line_ending='\n', returns_string:bool=False, enable_print:bool=True):
        self.begin_string = begin_string
        self.begin_line_ending = begin_line_ending
        self.finished_message = finished_message
        self.finished_line_ending = finished_line_ending
        
        self.returns_string = returns_string
        if self.returns_string:
            self.returned_string = ''
        else:
            self.returned_string = None    
        self.enable_print = enable_print
        
    def __enter__(self):
        self.returned_string = WrappingMessagePrinter.print_generic_progress_message(self.begin_string, self.begin_line_ending, self.returns_string, self.enable_print)
        # self.returned_string = WrappingMessagePrinter.print_file_progress_message(self.filepath, self.action, self.contents_description, self.print_line_ending, returns_string=self.returns_string)
        
    def __exit__(self, *args):
        if self.enable_print:
            print(self.finished_message, end=self.finished_line_ending)
        if self.returns_string:
            self.returned_string = f'{self.returned_string}{self.finished_message}{self.finished_line_ending}'
         
    @classmethod
    def print_generic_progress_message(cls, begin_string, begin_line_ending, returns_string, enable_print):
        if returns_string:
            out_string = f'{begin_string}...'
            if enable_print:
                print(out_string, end=begin_line_ending)
            return f'{out_string}{begin_line_ending}'
        else:
            if enable_print:
                print(f'{begin_string}...', end=begin_line_ending)



class CustomTreeFormatters:

    @classmethod
    def basic_custom_tree_formatter(cls, depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False, value_formatting_fn=None) -> str:
        """ For use with `print_keys_if_possible` to render a neat and pretty tree

            from neuropy.utils.print_helpers import CustomTreeFormatters

            
            print_keys_if_possible("sess.config.preprocessing_parameters", preprocessing_parameters_dict, custom_item_formatter=CustomTreeFormatters.basic_custom_tree_formatter)

        """
        prefix = '├── ' if depth_string else ''
        link_char = '│   ' if depth_string else '    '
        depth_string_with_link = depth_string + link_char
        formatted_string = f"{depth_string_with_link}{prefix}{curr_key}: {type_name}"
        if is_omitted_from_expansion:
            value_str = f"{(ANSI_COLOR_STRINGS.WARNING + ' (children omitted)' + ANSI_COLOR_STRINGS.ENDC)}"
        else:
            ## try to get the value:
            value_str = value_formatting_fn(curr_value)
            if (value_str is not None) and (len(value_str) > 0):
                value_str = f"{(ANSI_COLOR_STRINGS.WARNING + value_str + ANSI_COLOR_STRINGS.ENDC)}"
            else:
                value_str = ""
    
        if is_omitted_from_expansion:
            formatted_string += ' (children omitted)'
        else:
            ## try to get the value:
            if value_formatting_fn is None:
                # value_string_rep_fn = DocumentationFilePrinter.never_string_rep
                value_formatting_fn = DocumentationFilePrinter.string_rep_if_short_enough
            
            value_str = value_formatting_fn(curr_value)
            if (value_str is not None) and (len(value_str) > 0):
                ## add the value str
                formatted_string += f' {value_str}'

        return formatted_string


# ==================================================================================================================== #
# Category: Colored (ANSI-Formatted) Outputs:                                                                          #
# ==================================================================================================================== #
class ANSI_COLOR_STRINGS:
    """ Hardcoded ANSI-color strings. Can be used in print(...) like: `print(f"{bcolors.WARNING}Warning: No active frommets remain. Continue?{bcolors.ENDC}")` """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'

    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    LIGHTRED = '\033[91m'
    LIGHTGREEN = '\033[92m'
    LIGHTYELLOW = '\033[93m'
    LIGHTBLUE = '\033[94m'
    LIGHTMAGENTA = '\033[95m'
    LIGHTCYAN = '\033[96m'

    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ANSI_Coloring:
    """docstring for ANSI_Coloring."""
    def __init__(self, arg):
        super(ANSI_Coloring, self).__init__()

    @classmethod
    def ansi_highlight_only_suffix(cls, type_string, suffix_color=ANSI_COLOR_STRINGS.BOLD):
        """ From a FQDN-style type_string like 'pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult' generates a ansi-formatted string with the last suffix (the type name) colored. 
        Usage:
            type_string = 'pyphoplacecellanalysis.General.Model.ComputationResults.ComputationResult'
            ansi_highlighted_type_string = ansi_highlight_only_suffix(type_string)
            print(ansi_highlighted_type_string)
            >>> 'pyphoplacecellanalysis.General.Model.\x1b[93mComputationResult\x1b[0m'
        """
        return '.'.join(type_string.split('.')[:-2] + [(suffix_color + type_string.split('.')[-1] + ANSI_COLOR_STRINGS.ENDC)])


import io # used by DocumentationFilePrinter to capture stdout to a string buffer
from datetime import datetime # used by DocumentationFilePrinter to add the current date to the documentation header
from contextlib import redirect_stdout # used by DocumentationFilePrinter to capture print output
from ansi2html import Ansi2HTMLConverter # used by DocumentationFilePrinter to build html document from ansi-color coded version
from pathlib import Path
from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager # used by DocumentationFilePrinter to showing the output files

class DocumentationFilePrinter:
    """ Used to print and save readable data-structure documentation (in both plain and rich text) by wrapping `print_keys_if_possible(...)
    
        Usage:
            from neuropy.utils.print_helpers import DocumentationFilePrinter
            doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='ComputationResult')
            doc_printer.save_documentation('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'])

    """
    def __init__(self, doc_output_parent_folder='C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation', doc_name='ComputationResult', custom_plain_text_formatter=None, custom_rich_text_formatter=None, enable_print:bool=True):
        if not isinstance(doc_output_parent_folder, Path):
            doc_output_parent_folder = Path(doc_output_parent_folder)
        self.doc_output_parent_folder = doc_output_parent_folder
        self.doc_name = doc_name
        self.output_md_file = self.doc_output_parent_folder.joinpath(self.doc_name).with_suffix('.md')
        self.output_temp_ansi_file = self.doc_output_parent_folder.joinpath(self.doc_name).with_suffix('.ansi')
        self.output_html_file = self.doc_output_parent_folder.joinpath(self.doc_name).with_suffix('.html')
        
        self.md_string = ''
        self.ansi_string = ''
        self.html_string = ''
        
        self.enable_print = enable_print
        
        if custom_plain_text_formatter is None:
            self._custom_plain_text_formatter = DocumentationFilePrinter._default_plain_text_formatter
        if custom_rich_text_formatter is None:
            self._custom_rich_text_formatter = DocumentationFilePrinter._default_rich_text_formatter

        self.doc_header_string = f"{self.doc_name} - printed by print_keys_if_possible on {datetime.today().strftime('%Y-%m-%d')}\n===================================================================================================\n\n"
        self.doc_rich_header_string = f"{ANSI_COLOR_STRINGS.BOLD}{ANSI_COLOR_STRINGS.LIGHTRED}{self.doc_name}{ANSI_COLOR_STRINGS.ENDC} - printed by DocumentationFilePrinter on {datetime.today().strftime('%Y-%m-%d')}\n==================================================================================================="

        # ansi-to-html conversion:
        custom_css_content_dict = {'font-family':'"Lucida Console", "Courier New", monospace', 'font-size':'11px'} # 'font-family: "Lucida Console", "Courier New", monospace; font-size: 12px;'
        # self._asci_to_html_converter = Ansi2HTMLConverter(title=f'DOCS<{self.doc_name}>', dark_bg=False, custom_content_css_dict=custom_css_content_dict)
        # self._asci_to_html_converter = Ansi2HTMLConverter(title=f'DOCS<{self.doc_name}>', dark_bg=False, linkify=True, custom_bg='#FFFFFF', custom_fg='#FF0000', custom_content_css_dict=custom_css_content_dict)
        ## Light:
        # self._asci_to_html_converter = Ansi2HTMLConverter(title=f'DOCS<{self.doc_name}>', dark_bg=False, linkify=True, custom_bg='#FFFFFF', custom_fg=None, custom_content_css_dict=custom_css_content_dict)
        ## Dark (better for screen):
        self._asci_to_html_converter = Ansi2HTMLConverter(title=f'DOCS<{self.doc_name}>', dark_bg=True, custom_content_css_dict=custom_css_content_dict)

    def save_documentation(self, *args, skip_save_to_file=False, skip_print=False, custom_plain_text_formatter=None, custom_rich_text_formatter=None, **kwargs):
        """
            skip_print: if False, relies on self.enable_print's value to determine whether the output will be printed when this function is called
            
            Internally calls:
                print_keys_if_possible(*args, custom_rich_text_formatter=None, **kwargs) with custom_item_formatter= both plain and rich text formatters to print documentation
                saves to files unless skip_save_to_file=True
                
            Usage:
                doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='ComputationResult')
                doc_printer.save_documentation('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'])

        """
        ## Load both plaintext and rich-text output into dp.md_string, dp.ansi_string, and dp.html_string:
        if 'custom_item_formatter' in kwargs:
            print(f'WARNING: you must provide either `custom_plain_text_formatter` or `custom_rich_text_formatter` when calling save_documentation(...), but you passed `custom_item_formatter`')
            raise NotImplementedError
            
        if custom_plain_text_formatter is None:
            custom_plain_text_formatter = self._custom_plain_text_formatter
        if custom_rich_text_formatter is None:
            custom_rich_text_formatter = self._custom_rich_text_formatter

        # Plaintext version:
        with io.StringIO() as buf, redirect_stdout(buf):
            print(self.doc_header_string)
            print_keys_if_possible(*args, **kwargs, custom_item_formatter=custom_plain_text_formatter)
            self.md_string = buf.getvalue()

        # Rich (ANSI-colored) text version:
        with io.StringIO() as buf, redirect_stdout(buf):
            print(self.doc_rich_header_string)
            print_keys_if_possible(*args, **kwargs, custom_item_formatter=custom_rich_text_formatter)
            self.ansi_string = buf.getvalue()

        # ansi_string to html:
        self.html_string = self._asci_to_html_converter.convert(self.ansi_string)
        
        if not skip_save_to_file:
            self.write_to_files()
        
        if (not skip_print) and self.enable_print:
            print(self.ansi_string)
        

    def write_to_files(self):
        """Write variables out to files"""
        # Write plaintext version to file:
        with open(self.output_md_file, 'w', encoding='utf-8') as f:
            f.write(self.md_string)
        
        # Write html version to file:
        with open(self.output_html_file, 'w') as f_html:
            f_html.write(self.html_string)

        print(f"wrote to '{str(self.output_md_file)}' & '{str(self.output_html_file)}'.")

    def reveal_output_files_in_system_file_manager(self):
        reveal_in_system_file_manager(self.output_html_file)

    # extra methods ______________________________________________________________________________________________________ #
    def display_widget(self):
        """ Display an interactive jupyter-widget that allows you to open/reveal the generated files in the fileystem or default system display program. 
        
        Usage:
        
            doc_printer = DocumentationFilePrinter(doc_output_parent_folder=Path('C:/Users/pho/repos/PhoPy3DPositionAnalysis2021/EXTERNAL/DEVELOPER_NOTES/DataStructureDocumentation'), doc_name='ComputationResult')
            doc_printer.save_documentation('ComputationResult', curr_active_pipeline.computation_results['maze1'], non_expanded_item_keys=['_reverse_cellID_index_map'])
            doc_printer.display_widget()
        
        """
        import ipywidgets as widgets
        from IPython.display import display
        from pyphocorehelpers.gui.Jupyter.JupyterButtonRowWidget import build_fn_bound_buttons, JupyterButtonRowWidget, JupyterButtonColumnWidget
        from pyphocorehelpers.Filesystem.open_in_system_file_manager import reveal_in_system_file_manager
        from pyphocorehelpers.Filesystem.path_helpers import open_file_with_system_default
        
        btn_layout = widgets.Layout(width='auto', height='40px') #set width and height
        default_kwargs = dict(display='flex', flex_flow='column', align_items='stretch', layout=btn_layout)

        #TODO 2023-12-12 16:43: - [ ] Can potentially replace these complicated definitions with the simplier `fullwidth_path_widget` implementation which contains the two buttons by default
        # from pyphocorehelpers.gui.Jupyter.simple_widgets import fullwidth_path_widget       
        _out_row = JupyterButtonRowWidget.init_from_button_defns(button_defns=[("Documentation Folder", lambda _: reveal_in_system_file_manager(self.doc_output_parent_folder), default_kwargs),
            ("Generated Documentation", lambda _: self.reveal_output_files_in_system_file_manager(), default_kwargs),
            ])

        _out_row_html = JupyterButtonRowWidget.init_from_button_defns(button_defns=[("Open generated .html Documentation", lambda _: open_file_with_system_default(str(self.output_html_file.resolve())), default_kwargs),
                ("Reveal Generated .html Documentation", lambda _: reveal_in_system_file_manager(self.output_html_file), default_kwargs),
            ])

        _out_row_md = JupyterButtonRowWidget.init_from_button_defns(button_defns=[("Open generated .md Documentation", lambda _: open_file_with_system_default(str(self.output_md_file.resolve())), default_kwargs),
                ("Reveal Generated .md Documentation", lambda _: reveal_in_system_file_manager(self.output_md_file), default_kwargs),
            ])

        return widgets.VBox([_out_row.root_widget,
            _out_row_html.root_widget,
            _out_row_md.root_widget,
        ])

    # private methods ____________________________________________________________________________________________________ #
    @classmethod
    def never_string_rep(cls, value_rep: str):
        """ always returns None indicating no string-rep of the value should be included """
        return None

    @classmethod
    def string_rep_if_short_enough(cls, value: Any, max_length:int=280, max_num_lines:int=1, allow_reformatting:bool=True, allow_ellipsis_fill_too_long_regions:bool=True, debug_print:bool=False):
        """ returns the formatted str-rep of the value if it meets the criteria, otherwise nothing. An example `value_formatting_fn` 
        
        allow_reformatting: if True, allows removing lines to meet max_num_lines requirements so long as max_length is short enough
        
        
        Usage:
            from functools import partial
            from neuropy.utils.print_helpers import DocumentationFilePrinter

            custom_value_formatting_fn = partial(DocumentationFilePrinter.string_rep_if_short_enough, max_length=280, max_num_lines=1)
            new_custom_item_formatter = partial(DocumentationFilePrinter._default_rich_text_formatter, value_formatting_fn=custom_value_formatting_fn)
            print_keys_if_possible('context', context, max_depth=4, custom_item_formatter=new_custom_item_formatter)


        """
        if not isinstance(value, str):
            value = str(value)
            
        reformatting_line_replacement_str: str = '<br>'
        # reformatting_line_replacement_str: str = '\t'
        
        
        ellipsis_join_chars: str = '...'
        

        does_repr_have_too_many_lines: bool = (len(value.splitlines()) > max_num_lines)
        
        if (does_repr_have_too_many_lines and allow_reformatting):
            _val_arr = value.splitlines()
            _original_num_lines: int = len(_val_arr)
            _num_lines_to_combine: int = _original_num_lines - max_num_lines
                
            if (max_num_lines == 1) and (_original_num_lines > 1):
                if allow_ellipsis_fill_too_long_regions:
                    _reformatted_value = reformatting_line_replacement_str.join(_val_arr) ## join all lines
                else:
                    _reformatted_value = reformatting_line_replacement_str.join(_val_arr)
    
            elif (max_num_lines >= 3) and (_original_num_lines >= 3):
                _first_line = _val_arr.pop(0)
                _last_line = _val_arr.pop(-1) # remove the last line
                _middle_lines = _val_arr # remaining lines
                if allow_ellipsis_fill_too_long_regions:
                    _reformatted_value = '\n'.join((_first_line, ellipsis_join_chars, _last_line))  ## join extra lines after that                    
                else:
                    _reformatted_value = '\n'.join((_first_line, reformatting_line_replacement_str.join(_middle_lines), _last_line))  ## join extra lines after that
            else:
                raise NotImplementedError(f"_original_num_lines: {_original_num_lines}, max_num_lines: {max_num_lines}, _val_arr: {_val_arr}")

            does_repr_have_too_many_lines: bool = (len(_reformatted_value.splitlines()) > max_num_lines)
            assert (not does_repr_have_too_many_lines), f"string_rep_if_short_enough(...): ERROR:\n even after reformatting the _reformatted_value has too many lines! max_num_lines: {max_num_lines}\n len(_reformatted_value.splitlines()): {len(_reformatted_value.splitlines())}\n _reformatted_value: {_reformatted_value}\n value: {value}"
            ## update value if needed
            if value != _reformatted_value:
                if debug_print:
                    print(f'string_rep_if_short_enough(...): valueChanged:\n value: {value}\n\n _reformatted_value: {_reformatted_value}')
            value = _reformatted_value

        is_repr_too_long: bool = (len(value) > max_length)
        if is_repr_too_long and allow_ellipsis_fill_too_long_regions:
            # replaces all characters following the allowed max_length with ellipses
            characters_to_ellipses: int = (max_length-len(ellipsis_join_chars)) - len(value) # characters needed to be replaced by elipses
            _value_start = value[:(max_length-len(ellipsis_join_chars))]
            _reformatted_value = f"{_value_start}{ellipsis_join_chars}"
            is_repr_too_long: bool = (len(_reformatted_value) > max_length)
            
            assert (not is_repr_too_long), f"string_rep_if_short_enough(...): ERROR:\n even after replacing with ellipses the _reformatted_value has too many chars! max_length: {max_length}\n len(_reformatted_value): {len(_reformatted_value)}\n _reformatted_value: {_reformatted_value}\n value: {value}"
            ## update value if needed
            if (value != _reformatted_value):
                if debug_print:
                    print(f'string_rep_if_short_enough(...): valueChanged:\n value: {value}\n\n _reformatted_value: {_reformatted_value}')
            value = _reformatted_value


        is_repr_good_as_is: bool = (not does_repr_have_too_many_lines) and (not is_repr_too_long)
        if is_repr_good_as_is:
            # valid rep, include the value
            return f' = {value}'
        else:
            return None

    @classmethod
    def value_memory_usage_MB(cls, value: Any):
        """ returns the formatted memory size in MB. An example `value_formatting_fn` """
        if value is not None:
            memory_size_str_value: str = f"{print_object_memory_usage(value, enable_print=False):0.3f} MB"
            return memory_size_str_value        
        else:
            return None


    # Default formatters _________________________________________________________________________________________________ #
    @classmethod
    def _default_plain_text_formatter(cls, depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False, value_formatting_fn=None):
        """       """
        if value_formatting_fn is None:
            value_formatting_fn = cls.string_rep_if_short_enough

        return CustomTreeFormatters.basic_custom_tree_formatter(depth_string=depth_string, curr_key=curr_key, curr_value=curr_value, type_string=type_string, type_name=type_name, is_omitted_from_expansion=is_omitted_from_expansion, value_formatting_fn=value_formatting_fn)
    
    @classmethod
    def _default_rich_text_formatter(cls, depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False, value_formatting_fn=None):
        """ formats using ANSI_Coloring for rich colored output """
        if value_formatting_fn is None:
            # value_string_rep_fn = cls.never_string_rep
            value_formatting_fn = cls.string_rep_if_short_enough
            
        key_color = ANSI_COLOR_STRINGS.OKBLUE
        variable_type_color = ANSI_COLOR_STRINGS.LIGHTGREEN # looks better on screen
        # variable_type_color = ANSI_COLOR_STRINGS.LIGHTMAGENTA # converts to greyscale for printing better
        if is_omitted_from_expansion:
            value_str = f"{(ANSI_COLOR_STRINGS.WARNING + ' (children omitted)' + ANSI_COLOR_STRINGS.ENDC)}"
        else:
            ## try to get the value:
            value_str = value_formatting_fn(curr_value)
            if (value_str is not None) and (len(value_str) > 0):
                value_str = f"{(ANSI_COLOR_STRINGS.WARNING + value_str + ANSI_COLOR_STRINGS.ENDC)}"
            else:
                value_str = ""

        return f"{depth_string}- {key_color}{curr_key}{ANSI_COLOR_STRINGS.ENDC}: {variable_type_color}{ANSI_Coloring.ansi_highlight_only_suffix(type_name, suffix_color=ANSI_COLOR_STRINGS.BOLD)}{ANSI_COLOR_STRINGS.ENDC}{value_str}"

    @classmethod
    def _default_rich_text_greyscale_print_formatter(cls, depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False, value_formatting_fn=None):
        if value_formatting_fn is None:
            # value_string_rep_fn = cls.never_string_rep
            value_formatting_fn = cls.string_rep_if_short_enough
    
        """ formats using ANSI_Coloring for rich colored output """
        key_color = ANSI_COLOR_STRINGS.OKBLUE
        variable_type_color = ANSI_COLOR_STRINGS.LIGHTMAGENTA # converts to greyscale for printing better
        value_str = f"{(ANSI_COLOR_STRINGS.WARNING + ' (children omitted)' + ANSI_COLOR_STRINGS.ENDC) if is_omitted_from_expansion else (value_formatting_fn(curr_value) or '')}"
        return f"{depth_string}- {key_color}{curr_key}{ANSI_COLOR_STRINGS.ENDC}: {variable_type_color}{ANSI_Coloring.ansi_highlight_only_suffix(type_name, suffix_color=ANSI_COLOR_STRINGS.BOLD)}{ANSI_COLOR_STRINGS.ENDC}{value_str}"



# ==================================================================================================================== #
# HTML Formatted Text                                                                                                  #
# ==================================================================================================================== #

# function_attributes causes circular import issue :[
# @function_attributes(short_name='generate_html_string', tags=['html','format','text','labels','title','pyqtgraph'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-04-18 14:50')
def generate_html_string(input_str, color=None, font_size=None, bold=False, italic=False):
    """Generate an HTML string for use in a pyqtgraph label or title from an input string with optional formatting options.
    
    Args:
        input_str (str): The input string.
        color (str, optional): The color of the text. Defaults to None.
        font_size (str, optional): The font size of the text. Defaults to None.
        bold (bool, optional): Whether the text should be bold. Defaults to False.
        italic (bool, optional): Whether the text should be italic. Defaults to False.
    
    Returns:
        str: The HTML string.

    Usage:
        from neuropy.utils.print_helpers import generate_html_string
        i_str = generate_html_string('i', color='white', bold=True)
        j_str = generate_html_string('j', color='red', bold=True)
        title_str = generate_html_string(f'JSD(p_x_given_n, pf[{i_str}]) - JSD(p_x_given_n, pf[{j_str}]) where {j_str} non-firing')
        win.setTitle(title_str)

        >> 'JSD(p_x_given_n, pf[<b><span style="color:white;">i</span></b>]) - JSD(p_x_given_n, pf[<b><span style="color:red;">j</span></b>]) where <b><span style="color:red;">j</span></b> non-firing'
    """
    html_str = input_str
    if color:
        html_str = f'<span style="color:{color};">{html_str}</span>'
    if font_size:
        html_str = f'<font size="{font_size}">{html_str}</font>'
    if bold:
        html_str = f'<b>{html_str}</b>'
    if italic:
        html_str = f'<i>{html_str}</i>'
    return html_str



# ==================================================================================================================== #
# Category: Getting Current Date/Time as String for File Names/Logging/etc:                                            #
# ==================================================================================================================== #
# Helpers to get the current date and time. Written as functions so they stay current:
# from neuropy.utils.print_helpers import get_now_day_str, get_now_time_str, get_now_time_precise_str, get_now_rounded_time_str
def get_now_day_str() -> str:
    return datetime.today().strftime('%Y-%m-%d')
def get_now_time_str(time_separator='-') -> str:
    return str(time.strftime(f"%Y-%m-%d_%H{time_separator}%m", time.localtime(time.time())))
def get_now_time_precise_str(time_separator='-') -> str:
    return str(time.strftime(f"%Y-%m-%d_%H{time_separator}%m{time_separator}%S", time.localtime(time.time())))
def get_now_rounded_time_str(rounded_minutes:float=2.5, time_separator='') -> str:
    """ rounded_minutes:float=2.5 - nearest previous minute mark to round to
    """
    current_time = datetime.now()
    rounded_time = (current_time - timedelta(minutes=current_time.minute % rounded_minutes)).replace(second=0, microsecond=0)
    formatted_time = rounded_time.strftime(f"%Y-%m-%d_%I{time_separator}%M%p")
    return formatted_time

# TODO: enable simple backup-filename output
# current_time = datetime.now()
# rounded_time = (current_time + timedelta(minutes=2.5)).replace(second=0, microsecond=0)
# formatted_time = rounded_time.strftime("%I%M%p").lower() # gets time like "530pm"



# ==================================================================================================================== #
# Category: Formatting Seconds as Human Readable:                                                                      #
# ==================================================================================================================== #
def split_seconds_human_readable(seconds):
    """ splits the seconds argument into hour, minute, seconds, and fractional_seconds components.
        Does no formatting itself, but used by format_seconds_human_readable(...) for formatting seconds as a human-redable HH::MM:SS.FRACTIONAL time. 
    """
    if isinstance(seconds, int):
        whole_seconds = seconds
        fractional_seconds = None
    else:    
        whole_seconds = int(seconds)
        fractional_seconds = seconds - whole_seconds
    
    m, s = divmod(whole_seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s, fractional_seconds

def format_seconds_human_readable(seconds, h_m_s_format_array = ['{0:02}','{0:02}','{0:02}'], fixed_width=False):
    """ returns the formatted string built from the seconds argument as a human-redable HH::MM:SS.FRACTIONAL time. 
    
    fixed_width: bool - if True, always returns HH:MM:SS.sss components even if the hours, minutes, etc are zero. Otherwise it returns starting with the MSB non-zero component
 
    Usage:
        test_seconds_array = [0, 10, 95, 1503, 543812]

        for test_seconds in test_seconds_array:
            print_seconds_human_readable(test_seconds, fixed_width=True)
            print_seconds_human_readable(test_seconds)

        >> Output >>
            00:00:00
            00
            00:00:10
            10
            00:01:35
            01:35
            00:25:03
            25:03
            151:03:32
            151:03:32

     """
    included_h_m_s_formatted_output_strs_array = []
    h, m, s, fractional_seconds = split_seconds_human_readable(seconds)
    if fixed_width or (h > 0): 
        included_h_m_s_formatted_output_strs_array.append(h_m_s_format_array[0].format(h))
        # if we include hours, we must also include minutes (even if the minute components themselves are zero)
        included_h_m_s_formatted_output_strs_array.append(h_m_s_format_array[1].format(m))
        # if we include minutes, we must also include seconds (even if the seconds components themselves are zero)
        included_h_m_s_formatted_output_strs_array.append(h_m_s_format_array[2].format(s))
    elif (m > 0):
        included_h_m_s_formatted_output_strs_array.append(h_m_s_format_array[1].format(m))
        # if we include minutes, we must also include seconds (even if the seconds components themselves are zero)
        included_h_m_s_formatted_output_strs_array.append(h_m_s_format_array[2].format(s))
    else:
        # Otherwise we have both hours and minutes as zero, but we'll display seconds no matter what (even if they are zero):
        included_h_m_s_formatted_output_strs_array.append(h_m_s_format_array[2].format(s))

    formatted_timestamp_str = ':'.join(included_h_m_s_formatted_output_strs_array)
    if fractional_seconds is not None:
        frac_seconds_string = ('%f' % fractional_seconds).rstrip('0').rstrip('.').lstrip('0').lstrip('.') # strips any insignficant zeros from the right, and then '0.' string from the left.        
        formatted_timestamp_str = '{}:{}'.format(formatted_timestamp_str, frac_seconds_string) # append the fracitonal seconds string to the timestamp string
    return h, m, s, fractional_seconds, formatted_timestamp_str

def print_seconds_human_readable(seconds, h_m_s_format_array = ['{0:02}','{0:02}','{0:02}'], fixed_width=False):
    """ prints the seconds arguments as a human-redable HH::MM:SS.FRACTIONAL time. """
    h, m, s, fractional_seconds, formatted_timestamp_str = format_seconds_human_readable(seconds, h_m_s_format_array = h_m_s_format_array, fixed_width=fixed_width)
    print(formatted_timestamp_str) # print the timestamp
    return h, m, s, fractional_seconds, formatted_timestamp_str


# ==================================================================================================================== #
# Category: Memory Usage:                                                                                              #
# ==================================================================================================================== #
def print_dataframe_memory_usage(df, enable_print=True):
    """ df: a Pandas.DataFrame such as curr_active_pipeline.sess.spikes_df
    
    Usage:
        from neuropy.utils.print_helpers import print_dataframe_memory_usage
        print_dataframe_memory_usage(curr_active_pipeline.sess.spikes_df)

    >> prints >>:        
        ======== print_dataframe_memory_usage(df): ========
        Index                 0.00 MB
        t                     7.12 MB
        t_seconds             7.12 MB
        t_rel_seconds         7.12 MB
        shank                 3.56 MB
        cluster               3.56 MB
        aclu                  3.56 MB
        qclu                  3.56 MB
        x                     7.12 MB
        y                     7.12 MB
        speed                 7.12 MB
        traj                  3.56 MB
        lap                   3.56 MB
        maze_relative_lap     3.56 MB
        maze_id               3.56 MB
        neuron_type            35.58 MB
        flat_spike_idx        3.56 MB
        x_loaded              7.12 MB
        y_loaded              7.12 MB
        lin_pos               7.12 MB
        fragile_linear_neuron_IDX               3.56 MB
        PBE_id                7.12 MB
        dtype: object
        ============================
        Dataframe Total: 142.303 MB
    """
    print(f'======== print_dataframe_memory_usage(df): ========')
    curr_datatypes = df.dtypes
    each_columns_usage_bytes = df.memory_usage(deep=True)  # memory usage in bytes. Returns a Pandas.Series with the dataframe's column name as the index and a value in bytes.
    # each_columns_usage.index
    curr_column_names = each_columns_usage_bytes.index
    each_columns_usage_MB = each_columns_usage_bytes.apply(lambda x: x/(1024*1024))
    # each_columns_usage_MB
    if enable_print:
        each_columns_usage_MB_string = each_columns_usage_MB.apply(lambda x: f'{x:.2f} MB') # Round to 2 decimal places (the nearest 0.01 MB)
        print(f'{each_columns_usage_MB_string}')
    
    # Index                 0.00 MB
    # t                     7.12 MB
    # t_seconds             7.12 MB
    # t_rel_seconds         7.12 MB
    # shank                 3.56 MB
    # cluster               3.56 MB
    # aclu                  3.56 MB
    # qclu                  3.56 MB
    # x                     7.12 MB
    # y                     7.12 MB
    # speed                 7.12 MB
    # traj                  3.56 MB
    # lap                   3.56 MB
    # maze_relative_lap     3.56 MB
    # maze_id               3.56 MB
    # neuron_type            35.58 MB
    # flat_spike_idx        3.56 MB
    # x_loaded              7.12 MB
    # y_loaded              7.12 MB
    # lin_pos               7.12 MB
    # fragile_linear_neuron_IDX               3.56 MB
    # PBE_id                7.12 MB
    total_df_usage_MB = each_columns_usage_MB.sum()
    total_df_usage_MB_string = f'Dataframe Total: {total_df_usage_MB:.3f} MB' # round the total to 3 decimal places.
    
    print(f'============================\n{total_df_usage_MB_string}')
    return total_df_usage_MB # return the numeric number of megabytes that this df uses.
    
def print_object_memory_usage(obj, enable_print=True):
    """ prints the size of the passed in object in MB (Megabytes)
    Usage:
        from neuropy.utils.print_helpers import print_object_memory_usage, print_filesystem_file_size
        print_object_memory_usage(curr_bapun_pipeline.sess)
    """
    # size_bytes = obj.__sizeof__() # 1753723032
    size_bytes = objsize.get_deep_size(obj)
    size_MB = size_bytes/(1024*1024)
    if enable_print:
        object_size_string_MB = f'{size_MB:0.6f} MB'
        print(f'object size: {object_size_string_MB}')
    return size_MB

def print_filesystem_file_size(file_path, enable_print=True):
    """ prints the size of the file represented by the passed in path (if it exists) in MB (Megabytes)
    Usage:
        from neuropy.utils.print_helpers import print_filesystem_file_size
        print_filesystem_file_size(global_computation_results_pickle_path)
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    size_bytes = file_path.stat().st_size # Output is in bytes.
    size_MB = size_bytes/(1024*1024)
    if enable_print:
        file_size_string_MB = f'{size_MB:0.6f} MB'
        print(f'file size of {str(file_path)}: {file_size_string_MB}')
    return size_MB


# ==================================================================================================================== #
# Category: Debug Print                                                                                                #
# ==================================================================================================================== #
def debug_print(*args, **kwargs):
    # print(f'xbin_edges: {xbin_edges}\nxbin_centers: {xbin_centers}\nybin_edges: {ybin_edges}\nybin_centers: {ybin_centers}')
    out_strings = []
    for i, an_ordered_arg in enumerate(args):
        out_strings.append(f'args[{i}]: {args[i]}')
        
    for key, val in kwargs.items():
        out_strings.append(f'{key}: {val}')

    out_string = '\n'.join(out_strings)
    print(out_string)
    
def print_callexp(*args, **kwargs):
    """ DOES NOT WORK FROM Jupyter-lab notebook, untested in general.
    https://stackoverflow.com/questions/28244921/how-can-i-get-the-calling-expression-of-a-function-in-python?noredirect=1&lq=1
    
    """
    def _find_caller_node(root_node, func_name, last_lineno):
        # init search state
        found_node = None
        lineno = 0

        def _luke_astwalker(parent):
            nonlocal found_node
            nonlocal lineno
            for child in ast.iter_child_nodes(parent):
                # break if we passed the last line
                if hasattr(child, "lineno"):
                    lineno = child.lineno
                if lineno > last_lineno:
                    break

                # is it our candidate?
                if (isinstance(child, ast.Name)
                        and isinstance(parent, ast.Call)
                        and child.id == func_name):
                    # we have a candidate, but continue to walk the tree
                    # in case there's another one following. we can safely
                    # break here because the current node is a Name
                    found_node = parent
                    break

                # walk through children nodes, if any
                _luke_astwalker(child)

        # dig recursively to find caller's node
        _luke_astwalker(root_node)
        return found_node

    # get some info from 'inspect'
    frame = inspect.currentframe()
    backf = frame.f_back
    this_func_name = frame.f_code.co_name

    # get the source code of caller's module
    # note that we have to reload the entire module file since the
    # inspect.getsource() function doesn't work in some cases (i.e.: returned
    # source content was incomplete... Why?!).
    # --> is inspect.getsource broken???
    #     source = inspect.getsource(backf.f_code)
    #source = inspect.getsource(backf.f_code)
    with open(backf.f_code.co_filename, "r") as f:
        source = f.read()

    # get the ast node of caller's module
    # we don't need to use ast.increment_lineno() since we've loaded the whole
    # module
    ast_root = ast.parse(source, backf.f_code.co_filename)
    #ast.increment_lineno(ast_root, backf.f_code.co_firstlineno - 1)

    # find caller's ast node
    caller_node = _find_caller_node(ast_root, this_func_name, backf.f_lineno)

    # now, if caller's node has been found, we have the first line and the last
    # line of the caller's source
    if caller_node:
        #start_index = caller_node.lineno - backf.f_code.co_firstlineno
        #end_index = backf.f_lineno - backf.f_code.co_firstlineno + 1
        print("Hoooray! Found it!")
        start_index = caller_node.lineno - 1
        end_index = backf.f_lineno
        lineno = caller_node.lineno
        for ln in source.splitlines()[start_index:end_index]:
            print("  {:04d} {}".format(lineno, ln))
            lineno += 1

def dbg_dump(*args, dumpopt_stream=sys.stderr, dumpopt_forcename=True, dumpopt_pformat={'indent': 2}, dumpopt_srcinfo=1, **kwargs):
    """ DOES NOT WORK FROM Jupyter-lab notebook, untested in general.
    # pydump
    # A Python3 pretty-printer that also does introspection to detect the original
    # name of the passed variables
    #
    # Jean-Charles Lefebvre <polyvertex@gmail.com>
    # Latest version at: http://gist.github.com/polyvertex (pydump)

    Pretty-format every passed positional and named parameters, in that order,
    prefixed by their **original** name (i.e.: the one used by the caller), or
    by their type name for literals.
    Depends on the ``pprint``, ``inspect`` and ``ast`` standard modules.
    Note that the names of the keyword arguments you want to dump must not begin
    with ``dumpopt_`` since this prefix is used internally to differentiate
    options over values to dump.
    Also, the introspection code won't behave as expected if you make recursive
    calls to this function.
    Options can be passed as keyword arguments to tweak behavior and output
    format:
    * ``dumpopt_stream``:
      May you wish to print() the result directly, you can pass a stream object
      (e.g.: ``sys.stdout``) through this option, that will be given to
      ``print()``'s ``file`` keyword argument.
      You can also specify None in case you just want the output string to be
      returned without further ado.
    * ``dumpopt_forcename``:
      A boolean value to indicate wether you want every dumped value to be
      prepended by its name (i.e.: its name or its type).
      If ``False``, only non-literal values will be named.
    * ``dumpopt_pformat``:
      The dictionary of keyword arguments to pass to ``pprint.pformat()``
    * ``dumpopt_srcinfo``:
      Specify a false value (``None``, ``False``, zero) to skip caller's info.
      Specify ``1`` to output caller's line number only.
      Specify ``2`` to output caller's file name and line number.
      Specify ``3`` or greater to output caller's file path and line number.
    Example:
        ``dbg_dump(my_var, None, True, 123, "Bar", (4, 5, 6), fcall(), hello="world")``
    Result:
    ::
    DUMP(202):
        my_var: 'Foo'
        None: None
        Bool: True
        Num: 123
        Str: 'Bar'
        Tuple: (4, 5, 6)
        fcall(): "Function's Result"
        hello: 'world'
    """
    try:
        def _find_caller_node(root_node, func_name, last_lineno):
            # find caller's node by walking down the ast, searching for an
            # ast.Call object named func_name of which the last source line is
            # last_lineno
            found_node = None
            lineno = 0
            def _luke_astwalker(parent):
                nonlocal found_node
                nonlocal lineno
                for child in ast.iter_child_nodes(parent):
                    # break if we passed the last line
                    if hasattr(child, "lineno") and child.lineno:
                        lineno = child.lineno
                    if lineno > last_lineno:
                        break
                    # is it our candidate?
                    if (isinstance(child, ast.Name)
                            and isinstance(parent, ast.Call)
                            and child.id == func_name):
                        found_node = parent
                        break
                    _luke_astwalker(child)
            _luke_astwalker(root_node)
            return found_node

        frame = inspect.currentframe()
        backf = frame.f_back
        this_func_name = frame.f_code.co_name
        #this_func = backf.f_locals.get(
        #    this_func_name, backf.f_globals.get(this_func_name))

        # get the source code of caller's module
        # note that we have to reload the entire module file since the
        # inspect.getsource() function doesn't work in some cases (i.e.:
        # returned source content was incomplete... Why?!).
        # --> is inspect.getsource broken???
        #     source = inspect.getsource(backf.f_code)
        #source = inspect.getsource(backf.f_code)
        with open(backf.f_code.co_filename, "r") as f:
            source = f.read()

        # get the ast node of caller's module
        # we don't need to use ast.increment_lineno() since we've loaded the
        # whole module
        ast_root = ast.parse(source, backf.f_code.co_filename)
        #ast.increment_lineno(ast_root, backf.f_code.co_firstlineno - 1)

        # find caller's ast node
        caller_node = _find_caller_node(ast_root, this_func_name, backf.f_lineno)
        if not caller_node:
            raise Exception("caller's AST node not found")

        # keep some useful info for later
        src_info = {
            'file': backf.f_code.co_filename,
            'name': (
                backf.f_code.co_filename.replace("\\", "/").rpartition("/")[2]),
            'lineno': caller_node.lineno}

        # if caller's node has been found, we now have the AST of our parameters
        args_names = []
        for arg_node in caller_node.args:
            if isinstance(arg_node, ast.Name):
                args_names.append(arg_node.id)
            elif isinstance(arg_node, ast.Attribute):
                if hasattr(arg_node, "value") and hasattr(arg_node.value, "id"):
                    args_names.append(arg_node.value.id + "." + arg_node.attr)
                else:
                    args_names.append(arg_node.attr)
            elif isinstance(arg_node, ast.Subscript):
                args_names.append(arg_node.value.id + "[]")
            elif (isinstance(arg_node, ast.Call)
                    and hasattr(arg_node, "func")
                    and hasattr(arg_node.func, "id")):
                args_names.append(arg_node.func.id + "()")
            elif dumpopt_forcename:
                if (isinstance(arg_node, ast.NameConstant)
                        and arg_node.value is None):
                    args_names.append("None")
                elif (isinstance(arg_node, ast.NameConstant)
                        and arg_node.value in (False, True)):
                    args_names.append("Bool")
                else:
                    args_names.append(arg_node.__class__.__name__)
            else:
                args_names.append(None)
    except:
        #import traceback
        #traceback.print_exc()
        src_info = None
        args_names = [None] * len(args)

    args_count = len(args) + len(kwargs)

    output = ""
    if dumpopt_srcinfo:
        if not src_info:
            output += "DUMP(<unknown>):"
        else:
            if dumpopt_srcinfo <= 1:
                fmt = "DUMP({2}):"
            elif dumpopt_srcinfo == 2:
                fmt = "{1}({2}):"
            else:
                fmt = "{0}({2}):"
            output += fmt.format(
                        src_info['file'], src_info['name'], src_info['lineno'])
        output += "\n" if args_count > 1 else " "
    else:
        src_info = None

    for name, obj in zip(
            args_names + list(kwargs.keys()),
            list(args) + list(kwargs.values())):
        if name and name.startswith("dumpopt_"):
            continue
        if dumpopt_srcinfo and args_count > 1:
            output += "  "
        if name:
            output += name + ": "
        output += pprint.pformat(obj, **dumpopt_pformat) + "\n"

    if dumpopt_stream:
        print(output, end="", file=dumpopt_stream)
        return None # explicit is better than implicit
    else:
        return output.rstrip()
    

# ==================================================================================================================== #
# Category: Structural Overview/Outline:                                                                               #
# ==================================================================================================================== #

from pyphocorehelpers.DataStructure.enum_helpers import ExtendedEnum

class TypePrintMode(ExtendedEnum):
    """Describes the various ways of formatting an objects  type identity (`type(obj)`)
    Used by `print_file_progress_message(...)`
    """
    FULL_TYPE_STRING = "FullTypeString" # the complete output of calling type(obj) on the object. -- e.g. "<class 'pandas.core.frame.DataFrame'>"
    FULL_TYPE_FQDN = "FQDN" # the fully qualified path to the type. -- e.g. "pandas.core.frame.DataFrame"
    TYPE_NAME_ONLY = "NameOnly" # just the type name itself. -- e.g. "DataFrame"

    # Public methods _____________________________________________________________________________________________________ #
    def convert(self, curr_str:str, new_type) -> str:
        """ Converts from a more complete TypePrintMode down to a less complete one 

        Testing:
            TypePrintMode.FULL_TYPE_STRING.convert("<class 'pandas.core.frame.DataFrame'>", new_type=TypePrintMode.FULL_TYPE_FQDN) == 'pandas.core.frame.DataFrame'
            TypePrintMode.FULL_TYPE_STRING.convert("<class 'pandas.core.frame.DataFrame'>", new_type=TypePrintMode.FULL_TYPE_STRING) == "<class 'pandas.core.frame.DataFrame'>" # unaltered
            TypePrintMode.FULL_TYPE_STRING.convert("<class 'pandas.core.frame.DataFrame'>", new_type=TypePrintMode.TYPE_NAME_ONLY) == 'DataFrame'

        """
        _action_dict = {TypePrintMode.FULL_TYPE_STRING:{TypePrintMode.FULL_TYPE_FQDN:TypePrintMode._convert_FULL_TYPE_STR_to_FQDN, TypePrintMode.TYPE_NAME_ONLY:(lambda x: TypePrintMode._convert_FQDN_to_NAME_ONLY(TypePrintMode._convert_FULL_TYPE_STR_to_FQDN(x)))},
                        TypePrintMode.FULL_TYPE_FQDN:{TypePrintMode.TYPE_NAME_ONLY:TypePrintMode._convert_FQDN_to_NAME_ONLY}
                        }
        _curr_conversion_dict = _action_dict.get(self, {}) # if not found for this type, return empty dict
        _conversion_fcn = _curr_conversion_dict.get(new_type, (lambda x: x)) # if not found for this type, return current string
        return _conversion_fcn(curr_str) # call the conversion function on the curr_str

    # Private Conversion Functions _______________________________________________________________________________________ #
    @classmethod
    def _convert_FULL_TYPE_STR_to_FQDN(cls, curr_str: str) -> str:
        """ Extracts the class string out of the string returned by type(an_obj) 
        a_type_str: a string returned by type(an_obj) in the form of ["<class 'tuple'>", "<class 'int'>", "<class 'float'>", "<class 'numpy.ndarray'>", "<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>", "<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>", "<class 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters'>"]
        return: str
        
        Example:
            test_input_class_strings = ["<class 'tuple'>", "<class 'int'>", "<class 'float'>", "<class 'numpy.ndarray'>", "<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>", "<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>", "<class 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters'>"]
            m = [strip_type_str_to_classname(a_test_str) for a_test_str in test_input_class_strings]
            print(m)
            
            >> ['tuple', 'int', 'float', 'numpy.ndarray', 'pandas.core.series.Series', 'pandas.core.frame.DataFrame', 'pyphocorehelpers.indexing_helpers.BinningInfo', 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters']

        TESTING: TODO:
            test_input_class_strings = ["<class 'tuple'>", "<class 'int'>", "<class 'float'>", "<class 'numpy.ndarray'>", "<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>", "<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>", "<class 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters'>"]
            desired_output_class_strings = ['tuple','int','float','numpy.ndarray', 'pandas.core.series.Series', 'pandas.core.frame.DataFrame', 'pyphocorehelpers.indexing_helpers.BinningInfo', 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters']
            m = [strip_type_str_to_classname(a_test_str) for a_test_str in test_input_class_strings]
            ## TODO: compare m element-wise to desired_output_class_strings

        """
        if isinstance(curr_str, type):
            curr_str = str(curr_str) # convert to a string

        _search_results = re.search(r"<class '([^']+)'>", curr_str)
        if _search_results is not None:        
            return _search_results.group(1)

        _search_results = re.search(r"typing\.Optional\[\s*([\w\.\[\],\s]+)\s*\]", curr_str) # 'typing.Optional[float]'
        return f"Optional[{_search_results.group(1)}]"


    @classmethod
    def _convert_FQDN_to_NAME_ONLY(cls, curr_str: str) -> str:
        """ returns only the last portion of the dotted name. e.g. 'pandas.core.frame.DataFrame' -> 'DataFrame'
        TESTING: TODO:
            TypePrintMode._convert_FQDN_to_NAME_ONLY('pandas.core.frame.DataFrame') == 'DataFrame'
            TypePrintMode._convert_FQDN_to_NAME_ONLY('numpy.ndarray') == 'ndarray'
            TypePrintMode._convert_FQDN_to_NAME_ONLY('float') == 'float'
        """
        return curr_str.rsplit('.', 1)[-1] # 

def strip_type_str_to_classname(a_type_str: str) -> str:
    """ Extracts the class string out of the string returned by type(an_obj) 
    a_type_str: a string returned by type(an_obj) in the form of ["<class 'tuple'>", "<class 'int'>", "<class 'float'>", "<class 'numpy.ndarray'>", "<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>", "<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>", "<class 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters'>"]
    return: str
    
    Example:
        test_input_class_strings = ["<class 'tuple'>", "<class 'int'>", "<class 'float'>", "<class 'numpy.ndarray'>", "<class 'pandas.core.series.Series'>", "<class 'pandas.core.frame.DataFrame'>", "<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>", "<class 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters'>"]
        m = [strip_type_str_to_classname(a_test_str) for a_test_str in test_input_class_strings]
        print(m)        
        >> ['tuple', 'int', 'float', 'numpy.ndarray', 'pandas.core.series.Series', 'pandas.core.frame.DataFrame', 'pyphocorehelpers.indexing_helpers.BinningInfo', 'pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters']

    """
    return TypePrintMode._convert_FULL_TYPE_STR_to_FQDN(a_type_str)

def safe_get_variable_shape(a_value):
    """ generally and safely tries several methods of determining a_value's shape 
    
    assert safe_get_variable_shape(active_one_step_decoder.time_bin_size) is None
    assert isinstance(safe_get_variable_shape(active_one_step_decoder.spikes_df), tuple)
    assert isinstance(safe_get_variable_shape(active_one_step_decoder.F), tuple)
    """
    try:
        value_shape = np.shape(a_value)
    except ValueError:
        # 'ipdb>  np.array(a_value) >>> *** ValueError: could not broadcast input array from shape (2,12) into shape (2,)' occurs when a_value is a list of differently shaped np.arrays
        value_shape = () # set value_shape to () to continue trying other size tests
    except Exception as e:
        raise e
    
    if value_shape != ():
        # np.shape(...) worked
        return value_shape
    else:
        # empty shape:
        if hasattr(a_value, 'shape'):
            ## get the shape property
            value_shape = a_value.shape
            return value_shape
        else:
            # didn't work, try len(a_value):
            try:
                value_shape = len(a_value)
            except TypeError as e:
                # no length, no way to get shape
                value_shape = None
                return value_shape # value_shape = 'scalar'
            except Exception as e:
                raise e

    return value_shape


_GLOBAL_DO_NOT_EXPAND_CLASS_TYPES = [pd.DataFrame, pd.TimedeltaIndex, TimedeltaIndexResampler, logging.Logger, logging.Manager]
_GLOBAL_DO_NOT_EXPAND_CLASSNAMES = ["<class 'pyvista.core.pointset.StructuredGrid'>", "<class 'pyvista.core.pointset.UnstructuredGrid'>", "<class 'pandas.core.series.Series'>", "<class 'logging.Logger'>", "<class 'pyphoplacecellanalysis.General.Pipeline.Stages.Display.Plot'>"]
_GLOBAL_MAX_DEPTH = 20
def print_keys_if_possible(curr_key, curr_value, max_depth=20, depth=0, omit_curr_item_print=False, additional_excluded_item_classes=None, non_expanded_item_keys=None, custom_item_formatter=None):
    """Prints the keys of an object if possible, in a recurrsive manner.

    Args:
        curr_key (str): the current key
        curr_value (_type_): the current value
        depth (int, optional): _description_. Defaults to 0.
        additional_excluded_item_classes (list, optional): A list of class types to exclude
        non_expanded_item_keys (list, optional): a list of keys which will not be expanded, no matter their type, only themselves printed.
        custom_item_formater (((depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False) -> str), optional): e.g. , custom_item_formatter=(lambda depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False: f"{depth_string}- {curr_key}: {type_name}")

            custom_item_formater Examples:
                from neuropy.utils.print_helpers import TypePrintMode
                print_keys_if_possible('computation_config', curr_active_pipeline.computation_results['maze1'].computation_config, custom_item_formatter=(lambda depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion=False: f"{depth_string}- {curr_key}: <{TypePrintMode.FULL_TYPE_STRING.convert(type_string, new_type=TypePrintMode.TYPE_NAME_ONLY)}>{' (children omitted)' if is_omitted_from_expansion else ''}))
                ! See `DocumentationFilePrinter._plain_text_format_curr_value` and `DocumentationFilePrinter._rich_text_format_curr_value` for further examples 

    Returns:
        None
        
    Usage:
        print_keys_if_possible('computed_data', curr_computations_results.computed_data, depth=0)
        
        - computed_data: <class 'dict'>
            - pf1D: <class 'neuropy.analyses.placefields.PfND'>
            - pf2D: <class 'neuropy.analyses.placefields.PfND'>
            - pf2D_Decoder: <class 'pyphoplacecellanalysis.Analysis.Decoder.reconstruction.BayesianPlacemapPositionDecoder'>
            - pf2D_TwoStepDecoder: <class 'dict'>
                - xbin: <class 'numpy.ndarray'> - (59,)
                - ybin: <class 'numpy.ndarray'> - (21,)
                - avg_speed_per_pos: <class 'numpy.ndarray'> - (59, 21)
                - K: <class 'numpy.float64'>
                - V: <class 'float'>
                - sigma_t_all: <class 'numpy.ndarray'> - (59, 21)
                - all_x: <class 'numpy.ndarray'> - (59, 21, 2)
                - flat_all_x: <class 'list'>
                - original_all_x_shape: <class 'tuple'>
                - flat_p_x_given_n_and_x_prev: <class 'numpy.ndarray'> - (1239, 1717)
                - p_x_given_n_and_x_prev: <class 'numpy.ndarray'> - (59, 21, 1717)
                - most_likely_positions: <class 'numpy.ndarray'> - (2, 1717)
                - all_scaling_factors_k: <class 'numpy.ndarray'> - (1717,)
            - extended_stats: <class 'dict'>
                - time_binned_positioned_resampler: <class 'pandas.core.resample.TimedeltaIndexResampler'>
                - time_binned_position_df: <class 'pandas.core.frame.DataFrame'> - (1717, 18)
                - time_binned_position_mean: <class 'pandas.core.frame.DataFrame'> - (29, 16)
                - time_binned_position_covariance: <class 'pandas.core.frame.DataFrame'> - (16, 16)
            - firing_rate_trends: <class 'dict'>
                - active_rolling_window_times: <class 'pandas.core.indexes.timedeltas.TimedeltaIndex'>
                - mean_firing_rates: <class 'numpy.ndarray'> - (39,)
                - desired_window_length_seconds: <class 'float'>
                - desired_window_length_bins: <class 'int'>
                - active_firing_rates_df: <class 'pandas.core.frame.DataFrame'> - (1239, 39)
                - moving_mean_firing_rates_df: <class 'pandas.core.frame.DataFrame'> - (1239, 39)
            - placefield_overlap: <class 'dict'>
                - all_pairwise_neuron_IDs_combinations: <class 'numpy.ndarray'> - (741, 2)
                - total_pairwise_overlaps: <class 'numpy.ndarray'> - (741,)
                - all_pairwise_overlaps: <class 'numpy.ndarray'> - (741, 59, 21)
        
        ## Defining custom formatting functions:
            def _format_curr_value(depth_string, curr_key, curr_value, type_string, type_name):
                return f"{depth_string}['{curr_key}']: {type_name}"                
        
            print_keys_if_possible('active_firing_rate_trends', active_firing_rate_trends, custom_item_formatter=_format_curr_value)
        
            ['active_firing_rate_trends']: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
                ['time_bin_size_seconds']: float
                ['all_session_spikes']: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
                    ['time_window_edges']: numpy.ndarray - (5784,)
                    ['time_window_edges_binning_info']: pyphocorehelpers.indexing_helpers.BinningInfo
                        ['variable_extents']: tuple - (2,)
                        ['step']: float
                        ['num_bins']: int
                        ['bin_indicies']: numpy.ndarray - (5784,)
                    ['time_binned_unit_specific_binned_spike_rate']: pandas.core.frame.DataFrame - (5783, 52)
                    ['min_spike_rates']: pandas.core.series.Series - (52,)
                    ['median_spike_rates']: pandas.core.series.Series - (52,)
                    ['max_spike_rates']: pandas.core.series.Series - (52,)
                ['pf_included_spikes_only']: pyphocorehelpers.DataStructure.dynamic_parameters.DynamicParameters
                    ['time_window_edges']: numpy.ndarray - (5779,)
                    ['time_window_edges_binning_info']: pyphocorehelpers.indexing_helpers.BinningInfo
                        ['variable_extents']: tuple - (2,)
                        ['step']: float
                        ['num_bins']: int
                        ['bin_indicies']: numpy.ndarray - (5779,)
                    ['time_binned_unit_specific_binned_spike_rate']: pandas.core.frame.DataFrame - (5778, 52)
                    ['min_spike_rates']: pandas.core.series.Series - (52,)
                    ['median_spike_rates']: pandas.core.series.Series - (52,)
                    ['max_spike_rates']: pandas.core.series.Series - (52,)

    
    """
    if (depth >= _GLOBAL_MAX_DEPTH):
        print(f'OVERFLOW AT DEPTH {_GLOBAL_MAX_DEPTH}!')
        raise OverflowError
    elif (depth > max_depth):
        # print(f'finished at DEPTH {depth} with max_depth: {max_depth}!')
        return None
        
    else:
        depth_string = '\t' * depth
        curr_value_type = type(curr_value)
        curr_value_type_string = str(curr_value_type) # string like "<class 'numpy.ndarray'>"
        curr_value_type_name: str = strip_type_str_to_classname(curr_value_type_string) # string like "numpy.ndarray"
        
        is_non_expanded_item: bool = curr_key in (non_expanded_item_keys or [])

        if custom_item_formatter is None:
            # Define default print format function if no custom one is provided:
            # see DocumentationFilePrinter._plain_text_format_curr_value and DocumentationFilePrinter._rich_text_format_curr_value for examples
            # custom_item_formatter = DocumentationFilePrinter._default_plain_text_formatter
            ## Plaintext:
            custom_value_formatting_fn = partial(DocumentationFilePrinter.string_rep_if_short_enough, max_length=280, max_num_lines=1)
            custom_item_formatter = partial(DocumentationFilePrinter._default_plain_text_formatter, value_formatting_fn=custom_value_formatting_fn)
            
            # ## Rich:
            # custom_value_formatting_fn = partial(DocumentationFilePrinter.string_rep_if_short_enough, max_length=280, max_num_lines=1)
            # custom_item_formatter = partial(DocumentationFilePrinter._default_rich_text_formatter, value_formatting_fn=custom_value_formatting_fn)

        # e.g. lambda depth_string, curr_key, curr_value, type_string, type_name, is_omitted_from_expansion: f"{depth_string}- {curr_key}: {type_name}"

        if isinstance(curr_value, tuple(_GLOBAL_DO_NOT_EXPAND_CLASS_TYPES)) or (curr_value_type_string in _GLOBAL_DO_NOT_EXPAND_CLASSNAMES) or (curr_value_type_string in (additional_excluded_item_classes or [])) or (is_non_expanded_item):
            # Non-expanded items (won't recurrsively call `print_keys_if_possible` but will print unless omit_curr_item_print is True:
            if not omit_curr_item_print:
                curr_item_str = custom_item_formatter(depth_string=depth_string, curr_key=curr_key, curr_value=curr_value, type_string=curr_value_type_string, type_name=curr_value_type_name, is_omitted_from_expansion=True)
                # Recommendationa against using hasattr suggested here: https://hynek.me/articles/hasattr/
                try:
                    print(f"{curr_item_str} - {curr_value.shape}")
                except (AttributeError, AssertionError) as e:
                    print(f"{curr_item_str} - OMITTED TYPE WITH NO SHAPE")
        elif isinstance(curr_value, (np.ndarray, list, tuple)): 
            # Objects that are considered list-like are for example Python lists, tuples, sets, NumPy arrays, and Pandas Series.
            if not omit_curr_item_print:
                curr_item_str = custom_item_formatter(depth_string=depth_string, curr_key=curr_key, curr_value=curr_value, type_string=curr_value_type_string, type_name=curr_value_type_name, is_omitted_from_expansion=False)
                print(f"{curr_item_str} - {safe_get_variable_shape(curr_value)}") ## Shape only
                
        else:
            # Print the current item first:
            # See if the curr_value has .items() or not.
            # Check if all values are scalar types
            if isinstance(curr_value, dict) and all(isinstance(v, (int, float, str, bool)) for v in curr_value.values()):
                if not omit_curr_item_print:
                    # Get value type since they're all the same
                    # dict_values_type_string:str = str(type(list(curr_value.values())[0]))
                    curr_item_str = custom_item_formatter(depth_string=depth_string, curr_key=curr_key, curr_value=curr_value, type_string=curr_value_type_string, type_name=curr_value_type_name, is_omitted_from_expansion=True) + f"(all scalar values) - size: {len(curr_value)}"
                    print(curr_item_str)
                return  # Return early, don't print individual items
            else:
                # Typical case where we can't proclude expansion.
                if not omit_curr_item_print:
                    curr_item_str = custom_item_formatter(depth_string=depth_string, curr_key=curr_key, curr_value=curr_value, type_string=curr_value_type_string, type_name=curr_value_type_name, is_omitted_from_expansion=False)
                    print(curr_item_str)


            # Then recurrsively try to expand the item if possible:
            try:
                for (curr_child_key, curr_child_value) in curr_value.items():
                    # print children keys
                    print_keys_if_possible(curr_child_key, curr_child_value, max_depth=max_depth, depth=(depth+1), omit_curr_item_print=False, additional_excluded_item_classes=additional_excluded_item_classes, non_expanded_item_keys=non_expanded_item_keys, custom_item_formatter=custom_item_formatter)
            except AttributeError as e:
                # AttributeError: 'PfND' object has no attribute 'items'
                
                # Try to get __dict__ from the item:
                try:
                    curr_value_dict_rep = vars(curr_value) # gets the .__dict__ property if curr_value has one, otherwise throws a TypeError
                    print_keys_if_possible(f'{curr_key}.__dict__', curr_value_dict_rep, max_depth=max_depth, depth=depth, omit_curr_item_print=True, additional_excluded_item_classes=additional_excluded_item_classes, non_expanded_item_keys=non_expanded_item_keys, custom_item_formatter=custom_item_formatter) # do not increase depth in this regard so it prints at the same level. Also tell it not to print again.
                    
                except TypeError:
                    # print(f"{depth_string}- {curr_value_type}")
                    return None # terminal item
                
                except Exception as e:
                    print(f'Unhandled exception for innser block: {e}')
                    raise
            
            except Exception as e:
                print(f'Unhandled exception for outer block: {e}')
                raise



def debug_dump_object_member_shapes(obj):
    """ prints the name, type, and shape of all member variables. 
    Usage:
        debug_dump_object_member_shapes(active_one_step_decoder)
        >>>
            time_bin_size:	||	SCALAR	||	<class 'float'>
            pf:	||	SCALAR	||	<class 'neuropy.analyses.placefields.PfND'>
            spikes_df:	||	np.shape: (819170, 21)	||	<class 'pandas.core.frame.DataFrame'>
            debug_print:	||	SCALAR	||	<class 'bool'>
            neuron_IDXs:	||	np.shape: (64,)	||	<class 'numpy.ndarray'>
            neuron_IDs:	||	np.shape: (64,)	||	<class 'list'>
            F:	||	np.shape: (1856, 64)	||	<class 'numpy.ndarray'>
            P_x:	||	np.shape: (1856, 1)	||	<class 'numpy.ndarray'>
            unit_specific_time_binned_spike_counts:	||	np.shape: (64, 1717)	||	<class 'numpy.ndarray'>
            time_window_edges:	||	np.shape: (1718,)	||	<class 'numpy.ndarray'>
            time_window_edges_binning_info:	||	SCALAR	||	<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>
            total_spike_counts_per_window:	||	np.shape: (1717,)	||	<class 'numpy.ndarray'>
            time_window_centers:	||	np.shape: (1717,)	||	<class 'numpy.ndarray'>
            time_window_center_binning_info:	||	SCALAR	||	<class 'pyphocorehelpers.indexing_helpers.BinningInfo'>
            flat_p_x_given_n:	||	np.shape: (1856, 1717)	||	<class 'numpy.ndarray'>
            p_x_given_n:	||	np.shape: (64, 29, 1717)	||	<class 'numpy.ndarray'>
            most_likely_position_flat_indicies:	||	np.shape: (1717,)	||	<class 'numpy.ndarray'>
            most_likely_position_indicies:	||	np.shape: (2, 1717)	||	<class 'numpy.ndarray'>
        <<< (end output example)
    """
    for a_property_name, a_value in obj.__dict__.items():
        out_strings_arr = [f'{a_property_name}:']
        # np.isscalar(a_value)
        a_shape = np.shape(a_value)
        if a_shape != ():
            out_strings_arr.append(f'shape: {a_shape}')
        else:
            out_strings_arr.append(f'SCALAR')
            
        out_strings_arr.append(f'{str(type(a_value))}')
        out_string = '\t||\t'.join(out_strings_arr)
        print(out_string)

def print_value_overview_only(a_value, should_return_string=False):
    """ prints only basic information about a value, such as its type and shape if it has one. 
    
    Usage:
    
        test_value_1 = np.arange(15)
        print_value_overview_only(test_value_1)

        test_value_1 = list(range(15))
        print_value_overview_only(test_value_1)

        test_value_1 = 15
        print_value_overview_only(test_value_1)
            
        test_value_1 = 'test_string'
        print_value_overview_only(test_value_1)

        test_value_1 = {'key1': 0.34, 'key2': 'a'}
        print_value_overview_only(test_value_1)


    Note:
        str(value_type) => "<class 'numpy.ndarray'>"
        value_type.__name__ => 'ndarray'
        str(test_value_1.__class__).split("'") => ['<class ', 'numpy.ndarray', '>']
    """
    value_type = type(a_value)
    formatted_value_type_string = str(value_type).split("'")[1] # 'numpy.ndarray'
    value_shape = np.shape(a_value)
    if value_shape == ():
        # empty shape:
        # print(f'WARNING: value_shape is ().')
        try:
            value_shape = len(a_value)
        except TypeError as e:
            value_shape = 'scalar'
        except Exception as e:
            raise e

    output_string = f'<{formatted_value_type_string}; shape: {value_shape}>'
    if should_return_string:
        return output_string
    else:
        print(output_string)
        return None
    
def min_mean_max_sum(M, print_result=True):
    """Computes the min, mean, max, and sum of a matrix M (ignoring NaN values) and returns a tuple containing the results. Optionally can print the values.
    Useful for getting a simple summary/overview of a matrix.

    Args:
        M (_type_): _description_
        print_result (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    out = (np.nanmin(M), np.nanmean(M), np.nanmax(M), np.nansum(M))
    if print_result:
        print(f'min: {out[0]}, mean: {out[1]}, max: {out[2]}, sum: {out[3]}')
    return out
    
def document_active_variables(params, include_explicit_values=False, enable_print=True):
    """ Builds a skeleton for documenting variables and parameters by using the values set for a passed in instance.
    
    TODO: UNFINISHED!! UNTESTED.
    
    Usage:
        document_active_variables(active_curve_plotter_3d.params, enable_print=True)
    """
    keys = [str(a_key) for a_key in params.keys()]
    output_entries = dict()
    # for a_key, a_value in params.items():
    for a_key in params.keys():
        try:
            a_value = params.__dict__[a_key]
        except KeyError as e:
            # Fixes for DynamicParameters type objects
            a_value = params[a_key]
            
        curr_key_type = type(a_key)
        curr_key_str_rep = str(a_key)
        if curr_key_type == str:
            curr_key_type_string = ''
        else:
            # non-string keys included
            curr_key_type_string = f'<{str(curr_key_type)}>'
            
        curr_value_type = type(a_value)
        if curr_value_type == str:
            curr_value_type_string = ''
        else:
            # non-string values included
            curr_value_type_string = f'<{str(curr_value_type)}>'
            
        if include_explicit_values:
            curr_value_str_rep = str(a_value)
        else:
            # if include_explicit_values is false, don't include explicit default values
            curr_value_str_rep = ''
        # build output string:
        curr_output_string = f'{curr_key_str_rep}{curr_key_type_string}: ({curr_value_str_rep}{curr_value_type_string})'
        output_entries[curr_key_str_rep] = curr_output_string
        
    # print(f'keys: {keys}')
    if enable_print:
        print('\n'.join(list(output_entries.values())))
    return output_entries
    
    

    
# ==================================================================================================================== #
# LOGGING                                                                                                              #
# ==================================================================================================================== #

def get_system_hostname(enable_print:bool=False) -> str:
    hostname = socket.gethostname()
    if enable_print:
        print(f"Hostname: {hostname}") # Hostname: LNX00052
    return hostname



# logging.basicConfig()
# logging.root.setLevel(logging.DEBUG)

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("debug.log"),
#         logging.StreamHandler()
#     ]
# )

# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         fileHandler,
#         consoleHandler
#     ]
# )



def build_run_log_task_identifier(run_context: Union[str, List[str]], logging_root_FQDN: str = f'com.PhoHale.Spike3D', include_curr_time_str: bool=True, include_hostname:bool=True, additional_suffix:Optional[str]=None) -> str:
    """ Builds an identifier string for logging task progress like 'LNX00052.kdiba.gor01.two.2006-6-07_16-40-19'
    

    Usage:    
        from neuropy.utils.print_helpers import build_run_log_task_identifier

        build_run_log_task_identifier('test')
        
        >>> '2024-05-01_14-05-31.Apogee.com.PhoHale.Spike3D.test'

        build_run_log_task_identifier('test', logging_root_FQDN='Spike3D') # '2024-05-01_14-05-26.Apogee.Spike3D.test'
    
    """
    _out_parts = []

    if include_curr_time_str:
        # runtime_start_str: str = f'{datetime.now().strftime("%Y%m%d%H%M%S")}'
        runtime_start_str: str = get_now_time_precise_str()
        _out_parts.append(runtime_start_str)
    
    if include_hostname:
        hostname: str = get_system_hostname() # get the system's hostname
        # print(f"Hostname: {hostname}") # Hostname: LNX00052
        _out_parts.append(hostname)


    if (logging_root_FQDN is not None) and (len(logging_root_FQDN) > 0):
        _out_parts.append(logging_root_FQDN)

    ## add the main content name
    if isinstance(run_context, (list, tuple)):
        _out_parts.extend(run_context)
    elif hasattr(run_context, 'get_description'):
        # like an IdentifyingContext object
        _out_parts.append(str(run_context.get_description(separator='.'))) # 'kdiba.gor01.two.2006-6-07_16-40-19'
    else:
        _out_parts.append(str(run_context))

    ## build the output string
    out_str: str = '.'.join(_out_parts)

    if additional_suffix is not None:
        out_str = f"{out_str}.{additional_suffix.lstrip('.')}"

    return out_str


def build_logger(full_logger_string: str, file_logging_dir=None,
                logFormatter: Optional[logging.Formatter]=None, debug_print=True):
    """ builds a logger
    
    from neuropy.utils.print_helpers import build_run_log_task_identifier, build_logger

    Default used to be:
        file_logging_dir=Path('EXTERNAL/TESTING/Logging')
    """
    if logFormatter is None:
        # logFormatter = logging.Formatter("%(relativeCreated)d %(name)s]  [%(levelname)-5.5s]  %(message)s")
        logFormatter = logging.Formatter("%(asctime)s %(name)s]  [%(levelname)-5.5s]  %(message)s")
    
    task_logger: logging.Logger = logging.getLogger(full_logger_string) # create logger
    print(f'build_logger(full_logger_string="{full_logger_string}", file_logging_dir: {file_logging_dir}):')
    if debug_print:
        print(f'\t task_logger.handlers: {task_logger.handlers}')
    task_logger.handlers = []
    # task_logger.removeHandler()

    if file_logging_dir is not None:
        # file logging enabled:
        if file_logging_dir.is_file():
            # file_logging_dir is an entire logging file:
            module_logging_path = file_logging_dir.resolve()
            file_logging_dir = module_logging_path.parent.resolve() # get the parent of the log file provided as the logging directory
        else:
            # file_logging_dir = Path('EXTERNAL/TESTING/Logging') # 'C:\Users\pho\repos\PhoPy3DPositionAnalysis2021\EXTERNAL\TESTING\Logging'
            module_logging_path = file_logging_dir.joinpath(f'debug_{task_logger.name}.log') # task_logger.name # 'com.PhoHale.Spike3D.notebook'

        # Create logging directory if it doesn't exist
        file_logging_dir.mkdir(parents=True, exist_ok=True)

        # File Logging:    
        print(f'\t Task logger "{task_logger.name}" has file logging enabled and will log to "{str(module_logging_path)}"')
        fileHandler: logging.FileHandler = logging.FileHandler(module_logging_path)
        fileHandler.setFormatter(logFormatter)
        
        # fileHandler.baseFilename
        task_logger.addHandler(fileHandler)

    # consoleHandler = logging.StreamHandler(sys.stdout)
    # consoleHandler.setFormatter(logFormatter)
    # # task_logger.addHandler(consoleHandler)

    # General Logger Setup:
    task_logger.setLevel(logging.DEBUG)
    task_logger.info(f'==========================================================================================\n========== Logger INIT "{task_logger.name}" ==============================')
    return task_logger


# ==================================================================================================================== #
# Tree/Hierarchy Renderers and Previewers                                                                              #
# ==================================================================================================================== #





# ==================================================================================================================== #
# PPRINTING                                                                                                            #
# ==================================================================================================================== #



# ==================================================================================================================== #
# 2024-05-30 - Custom Formatters                                                                                       #
# ==================================================================================================================== #
import numpy as np
import pandas as pd
import math
from IPython.display import HTML
import matplotlib.pyplot as plt

# @function_attributes(short_name=None, tags=['elide', 'truncation', 'display'], input_requires=[], output_provides=[], uses=[], used_by=['render_scrollable_colored_table_from_dataframe'], creation_date='2025-01-01 14:44', related_items=[])
def ellided_dataframe(df: pd.DataFrame, max_rows_to_include: int = 200, num_truncated_rows_per_ellipsis_rows: int = 1000) -> pd.DataFrame:
    """ returns a truncated/elided dataframe if the number of rows exceeds `max_rows_to_include`. Returned unchanged if n_rows is already less than `max_rows_to_include`.
    
    from neuropy.utils.print_helpers import ellided_dataframe
    
    truncated_df = ellided_dataframe(df=df, max_rows_to_include=100)
    """
    # Truncate DataFrame if too many rows ________________________________________________________________________________ #
    n_rows: int = len(df)
    n_rows_truncated: int = 0
    if n_rows <= max_rows_to_include:
        return df
    else:
        n_rows_truncated: int = (n_rows - max_rows_to_include)
        n_truncation_symbols: int = math.ceil(n_rows_truncated / num_truncated_rows_per_ellipsis_rows)
        half = max_rows_to_include // 2
        top_df = df.head(half)
        bottom_df = df.tail(half)
        
        # Generate multiple ellipsis rows (One per 1,000 truncated rows)
        # starts = (np.arange(n_truncation_symbols) * num_truncated_rows_per_ellipsis_rows)
        # ends = ((np.arange(n_truncation_symbols)+1) * num_truncated_rows_per_ellipsis_rows) # [1:] could use modulo `n_rows_truncated`
        # ellipsis_index = [f"... trunc<{starts[i]}-{ends[i]}>)" for i in range(n_truncation_symbols)]
        ellipsis_index = [f"... trunc[{i}<{num_truncated_rows_per_ellipsis_rows} rows>)" for i in range(n_truncation_symbols)]
        ellipsis_rows = pd.DataFrame(
            [['…'] * df.shape[1]] * n_truncation_symbols,
            columns=df.columns,
            index=ellipsis_index
        )
        return pd.concat([top_df, ellipsis_rows, bottom_df], axis=0)


# @function_attributes(short_name=None, tags=['HTML', 'render', 'formatting', 'layout'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2025-01-01 15:30', related_items=[])  
def estimate_rendered_df_table_height(df: pd.DataFrame, debug_print=False) -> float:
    """ estimates the required height in px for rendered df
    
    Usage:
        estimated_table_height = estimate_rendered_df_table_height(df=normalized_df)
        will_scroll_vertically: bool = (estimated_table_height >= max_height)
    
    """
    table_row_height: int = 24
    const_table_parts_height = 72

    # Get the number of levels in the MultiIndex
    # Get the number of levels in the index and columns
    index_levels = df.index.nlevels
    column_levels = df.columns.nlevels

    # Determine the maximum number of levels
    max_header_levels = max(index_levels, column_levels) + 1
    if debug_print:
        print(f"Index levels: {index_levels}, Column levels: {column_levels}")
        print(f"Maximum number of levels: {max_header_levels}")

    n_rows: int = len(df)

    total_table_height = (n_rows * table_row_height) + (max_header_levels * table_row_height)
    return total_table_height


# @function_attributes(short_name=None, tags=['table', 'dataframe', 'formatter', 'display', 'render'], input_requires=[], output_provides=[], uses=['ellided_dataframe], used_by=[], creation_date='2025-01-01 13:39', related_items=[])
def render_scrollable_colored_table_from_dataframe(df: pd.DataFrame, cmap_name: str = 'viridis', max_height: Optional[int] = 400, width: str = '100%', is_dark_mode: bool=True, max_rows_to_render_for_performance: int = 100, output_fn=HTML, **kwargs) -> Union[HTML, str]:
    """ Takes a numpy array of values and returns a scrollable and color-coded table rendition of it

    Usage:    
        from neuropy.utils.print_helpers import render_scrollable_colored_table_from_dataframe

        # Example usage:

        # Example 2D NumPy array
        array = np.random.rand(100, 10)
        # Draw it
        render_scrollable_colored_table(array)
        
        # Example 2:
            render_scrollable_colored_table(np.random.rand(100, 10), cmap_name='plasma', max_height=500, width='80%')
            render_scrollable_colored_table_from_dataframe(df=normalized_df, cmap_name=cmap_name, max_height=max_height, width=width, **kwargs)
            
    """
    # white_color: str = 'white'
    white_color: str = '#cacaca'
    black_color: str = 'black'

    # Validate input array
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a DataFrame array.")
        
    # Validate colormap name
    if cmap_name is not None:
        if cmap_name not in plt.colormaps():
            raise ValueError(f"Invalid colormap name '{cmap_name}'. Use one of: {', '.join(plt.colormaps())}.")
    

    # Truncate DataFrame (for PERFORMANCE) if too many rows ________________________________________________________________________________ #
    n_rows: int = len(df)
    n_rows_truncated: int = 0
    if n_rows > max_rows_to_render_for_performance:
        n_rows_truncated: int = (n_rows - max_rows_to_render_for_performance)
        truncated_df = ellided_dataframe(df=df, max_rows_to_include=max_rows_to_render_for_performance)

    else:
        truncated_df = df
        
    ## Normalize each column to the 0, 1 range or ignore formatting
    # Normalize the data to [0, 1] range
    # normalized_df = (df - df.min().min()) / (df.max().max() - df.min().min())
    
    normalized_df = deepcopy(truncated_df)
    
    will_scroll_vertically: bool = False
    estimated_table_height = estimate_rendered_df_table_height(df=normalized_df)
    if max_height is None:
        max_height_str: str = 'auto'
    else:
        will_scroll_vertically = (estimated_table_height >= max_height)
        max_height_str: str = f"{max_height}px"

    # Function to calculate luminance and return appropriate text color
    def text_contrast(rgba):
        r, g, b, a = rgba[:4]
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        return black_color if luminance > 0.5 else white_color

    # Define a function to apply a colormap and text color based on luminance
    def color_map(val):
        use_default_formatting = True
        cmap = None
        if cmap_name is not None:
            cmap = plt.get_cmap(cmap_name)  # Use the provided colormap
        
        try:
            color = cmap(val)
            text_color = text_contrast(color)
            use_default_formatting = False
        except TypeError as e:
            ## Not formattable, return default text color (white)
            use_default_formatting = True
        except Exception as e:
            raise e
        
        if use_default_formatting:
            if is_dark_mode:
                color = black_color
                text_color = white_color
            else:
                color = white_color
                text_color = black_color

        return f'background-color: rgba({color[0]*255}, {color[1]*255}, {color[2]*255}, {color[3]}); color: {text_color}'

    # Apply the color map with contrast adjustment
    styled_df = normalized_df.style.applymap(color_map)
    

    # Updated shadow and gradient for scroll indication
    box_shadow = ""
    if will_scroll_vertically:
        ## only add the shadow if the table's estimated height exceeds the available height
        box_shadow = 'box-shadow: inset 0 -32px 15px -10px rgba(20,255,20,0.5);' ## this one is good

    formatted_table = styled_df.set_table_attributes(f'style="display:block;overflow-x:auto;max-height:{max_height_str};width:{width};border-collapse:collapse;position:relative;{box_shadow};"').render()
    
    table_shape_footer = f"""
        <div style="text-align: left; margin-top: 10px; font-size: 12px; color: {white_color if is_dark_mode else black_color};">
            {df.shape[0]} rows × {df.shape[1]} columns
    """    

    if n_rows_truncated > 0:
        # table_shape_footer += f"    <{n_rows_truncated} rows truncated>\n"
        table_shape_footer += f"    <{n_rows - n_rows_truncated}/{n_rows} rows displayed>\n"

    table_shape_footer += """
        </div>
    """
    
    # Updated gradient overlay placement
    scrollable_container = f"""
        <div style="position:relative;max-height:{max_height_str};overflow:auto;">
            {formatted_table}
        </div>
    """
    
    # scrollable_container = f"""
    #     <div style="position:relative;max-height:{max_height_str};overflow:auto;padding-bottom:20px;">
    #         {formatted_table}
    #         <div style="position:sticky;bottom:0;left:0;width:100%;height:20px;
    #         background:linear-gradient(to top, rgba(255,0,0,0.5), rgba(255,0,0,0));pointer-events:none;">
    #         </div>
    #     </div>
    # """

    # # Gradient overlay now positioned outside the scrollable container
    # gradient_overlay = f"""
    #     <div style="position:absolute;bottom:0;left:0;width:100%;height:40px;
    #     background:linear-gradient(to top, rgba(255,0,0,0.5), rgba(255,0,0,0));z-index:1;pointer-events:none;">
    #     </div>
    # """

    # Updated full HTML structure
    full_html = f"""
    """
    full_html += f""" <div style="position:relative;">
            {scrollable_container}
            {table_shape_footer}
        </div>
    """
    # {gradient_overlay}  <!-- Ensures gradient is outside the scrollable area -->

    # Render the DataFrame as a scrollable table with color-coded values
    scrollable_table = output_fn(full_html)

    return scrollable_table
        

def render_scrollable_colored_table(array: NDArray, cmap_name: str = 'viridis', max_height: Optional[int] = 400, width: str = '100%', **kwargs) -> Union[HTML, str]:
    """ Takes a numpy array of values and returns a scrollable and color-coded table rendition of it

    Usage:    
        from neuropy.utils.print_helpers import render_scrollable_colored_table

        # Example 2D NumPy array
        array = np.random.rand(100, 10)
        # Draw it
        render_scrollable_colored_table(array)

    """
    # Validate input array
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
    if array.ndim != 2:
        raise ValueError("Input array must be 2D.")
    
    # Validate colormap name
    if cmap_name not in plt.colormaps():
        raise ValueError(f"Invalid colormap name '{cmap_name}'. Use one of: {', '.join(plt.colormaps())}.")
    
    # Convert the array to a Pandas DataFrame
    df = pd.DataFrame(array)

    # Normalize the data to [0, 1] range
    normalized_df = (df - df.min().min()) / (df.max().max() - df.min().min())

    return render_scrollable_colored_table_from_dataframe(df=normalized_df, cmap_name=cmap_name, max_height=max_height, width=width, **kwargs)

    
# Example usage:
# render_scrollable_colored_table(np.random.rand(100, 10), cmap_name='plasma', max_height=500, width='80%')

    
    

    
    
def array_preview_with_shape(arr):
    """ Text-only Represntation that prints np.shape(arr) 
    
        from neuropy.utils.print_helpers import array_preview_with_shape

        # Register the custom display function for numpy arrays
        import IPython
        ip = IPython.get_ipython()
        ip.display_formatter.formatters['text/html'].for_type(np.ndarray, array_preview_with_shape) # only registers for NDArray

        # Example usage
        arr = np.random.rand(3, 4)
        display(arr)

    """
    if isinstance(arr, np.ndarray):
        display(HTML(f"<pre>array{arr.shape} of dtype {arr.dtype}</pre>"))
    elif isinstance(arr, (list, tuple)):
        display(HTML(f"<pre>native-python list {len(arr)}</pre>"))
    elif isinstance(arr, pd.DataFrame):
        display(HTML(f"<pre>DataFrame with {len(arr)} rows and {len(arr.columns)} columns</pre>"))
    else:
        raise ValueError("The input is not a NumPy array.")


def array_preview_with_graphical_shape_repr_html(arr):
    """Generate an HTML representation for a NumPy array, similar to Dask.
        
    from neuropy.utils.print_helpers import array_preview_with_graphical_shape_repr_html
    
    # Register the custom display function for NumPy arrays
    import IPython
    ip = IPython.get_ipython()
    ip.display_formatter.formatters['text/html'].for_type(np.ndarray, lambda arr: array_preview_with_graphical_shape_repr_html(arr))

    # Example usage
    arr = np.random.rand(3, 4)
    display(arr)


    arr = np.random.rand(9, 64)
    display(arr)

    arr = np.random.rand(9, 64, 4)
    display(arr)

    """
    if isinstance(arr, np.ndarray):
        arr = da.array(arr)
        return display(arr)
        # shape_str = ' &times; '.join(map(str, arr.shape))
        # dtype_str = arr.dtype
        # return f"<pre>array[{shape_str}] dtype={dtype_str}</pre>"
    else:
        raise ValueError("The input is not a NumPy array.")



# Generate heatmap

    
def _subfn_create_heatmap(data: NDArray, brokenaxes_kwargs=None) -> Optional[BytesIO]: # , omission_indices: list = None
    """ 
    
    #TODO 2024-08-16 04:05: - [ ] Make non-interactive and open in the background

    from neuropy.utils.matplotlib_helpers import matplotlib_configuration
    with matplotlib_configuration(is_interactive=False, backend='AGG'):
        # Perform non-interactive Matplotlib operations with 'AGG' backend
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Non-interactive Mode with AGG Backend')
        plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

            
    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    _bak_rcParams = mpl.rcParams.copy()

    matplotlib.use('Qt5Agg')
    # %matplotlib inline
    # %matplotlib auto


    # _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')
    _restore_previous_matplotlib_settings_callback = matplotlib_configuration_update(is_interactive=True, backend='Qt5Agg')

        
    """
    if (data.ndim < 2):
        data = np.atleast_2d(data)
        # fix issues with 1D data like `TypeError: Invalid shape (58,) for image data`
    
    import matplotlib.pyplot as plt
    
    try:
        imshow_shared_kwargs = {
            'origin': 'lower',
        }

        active_cmap = 'viridis'
        fig = plt.figure(figsize=(3, 3), num='_jup_backend')
        ax = fig.add_subplot(111)
        ax.imshow(data, cmap=active_cmap, **imshow_shared_kwargs)
        ax.axis('off')
            
        buf = BytesIO()
        
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)

        buf.seek(0)
        
    except SystemError as err:
        # SystemError: tile cannot extend outside image
        print(f'ERROR: Encountered error while plotting heatmap:\n\terr: {err}')
        print(f'\tnp.shape(data): {np.shape(data)}\n\tdata: {data}')
        buf = None

    finally:
        plt.close()        
    
    return buf

# Convert to ipywidgets Image
def _subfn_display_heatmap(data: NDArray, brokenaxes_kwargs=None, **img_kwargs) -> Optional[Image]:
    """ Renders a small thumbnail Image of a heatmap array
    
    """
    img_kwargs = dict(width=None, height=img_kwargs.get('height', 100), format='png') | img_kwargs
    buf = _subfn_create_heatmap(data, brokenaxes_kwargs=brokenaxes_kwargs)
    if buf is not None:
        # Create an IPython Image object
        img = Image(data=buf.getvalue(), **img_kwargs)
        return img
    else:
        return None

def array_preview_with_heatmap_repr_html(arr, include_shape: bool=True, horizontal_layout=True, include_plaintext_repr:bool=False, **kwargs):
    """ Generate an HTML representation for a NumPy array with a Dask shape preview and a thumbnail heatmap
    
        from neuropy.utils.print_helpers import array_preview_with_heatmap_repr_html

        # Register the custom display function for numpy arrays
        import IPython
        ip = IPython.get_ipython()
        ip.display_formatter.formatters['text/html'].for_type(np.ndarray, array_preview_with_heatmap) # only registers for NDArray

        # Example usage
        arr = np.random.rand(3, 4)
        display(arr)

    """
    max_allowed_arr_elements: int = 10000

    if isinstance(arr, np.ndarray):
        
        n_dim: int = np.ndim(arr)
        if n_dim > 2:
            print(f'WARN: n_dim: {n_dim} greater than 2 is unsupported!')
            from pyphocorehelpers.plotting.media_output_helpers import get_array_as_image_stack
            # #TODO 2024-08-13 05:05: - [ ] use get_array_as_image_stack to render the 3D array
            message = f"Heatmap Err: n_dim: {n_dim} greater than 2 is unsupported!"
            heatmap_html = f"""
            <div style="text-align: center; padding: 20px; border: 1px solid #ccc;">
                <p style="font-size: 16px; color: red;">{message}</p>
            </div>
            """

        else:
            ## n_dim == 2
            if np.shape(arr)[0] > max_allowed_arr_elements: 
                # truncate 
                arr = arr[max_allowed_arr_elements:]
            
            heatmap_image = _subfn_display_heatmap(arr, **kwargs)
            if (heatmap_image is not None):
                orientation = "row" if horizontal_layout else "column"
                ## Lays out side-by-side:
                # Convert the IPython Image object to a base64-encoded string
                heatmap_image_data = heatmap_image.data
                b64_image = base64.b64encode(heatmap_image_data).decode('utf-8')
                # Create an HTML widget for the heatmap
                heatmap_size_format_str: str = ''
                width = kwargs.get('width', None)
                if (width is not None) and (width > 0):
                    heatmap_size_format_str = heatmap_size_format_str + f'width="{width}" '
                height = kwargs.get('height', None)
                if (height is not None) and (height > 0):
                    heatmap_size_format_str = heatmap_size_format_str + f'height="{height}" '
                
                heatmap_html = f'<img src="data:image/png;base64,{b64_image}" {heatmap_size_format_str}style="background:transparent;"/>' #  width="{ndarray_preview_config.heatmap_thumbnail_width}"

            else:
                # getting image failed:
                # Create an HTML widget for the heatmap
                message = "Heatmap Err"
                heatmap_html = f"""
                <div style="text-align: center; padding: 20px; border: 1px solid #ccc;">
                    <p style="font-size: 16px; color: red;">{message}</p>
                </div>
                """

        # height="{height}"
        dask_array_widget_html = ""
        plaintext_html = ""
        
        if include_shape:
            dask_array_widget: widgets.HTML = widgets.HTML(value=da.array(arr)._repr_html_())
            dask_array_widget_html: str = dask_array_widget.value
            dask_array_widget_html = f"""
                <div style="margin-left: 10px;">
                    {dask_array_widget_html}
                </div>
            """

        if include_plaintext_repr:                
            # plaintext_repr = np.array2string(arr, edgeitems=3, threshold=5)  # Adjust these parameters as needed
            plaintext_repr = np.array2string(arr)
            plaintext_html = f"<pre>{plaintext_repr}</pre>"
            plaintext_html = f"""
                <div style="margin-left: 10px;">
                    {plaintext_html}
                </div>
            """
            
        # Combine both HTML representations
        if horizontal_layout:
            combined_html = f"""
            <div style="display: flex; flex-direction: row; align-items: flex-start;">
                <div>{heatmap_html}</div>
                {dask_array_widget_html}
                {plaintext_html}
            </div>
            """
        else:
            combined_html = f"""
            <div style="display: flex; flex-direction: column; align-items: center;">
                <div>{heatmap_html}</div>
                <div style="margin-top: 10px;">
                    {dask_array_widget_html}
                    {plaintext_html}
                </div>
            </div>
            """
        return combined_html

    else:
        raise ValueError("The input is not a NumPy array.")



# # omitted_indicies_list = []
# def _leading_trailing(a: NDArray, edgeitems: int, index=()): # , omitted_indicies_list=[]
#     """
#     Keep only the N-D corners (leading and trailing edges) of an array.

#     Should be passed a base-class ndarray, since it makes no guarantees about
#     preserving subclasses.
#     """
#     global omitted_indicies_list
#     axis = len(index)
#     if axis == a.ndim:
#         return a[index]

#     if a.shape[axis] > 2*edgeitems:
#         left_edge_idxs = (index + np.index_exp[:edgeitems])
#         right_edge_idxs = (index + np.index_exp[-edgeitems:])
#         print(f'index: {index}, axis: {axis}, a.shape[axis]: {a.shape[axis]}')
#         # omitted_idxs = (left_edge_idxs[-1], right_edge_idxs[0])
#         omitted_idxs = (int(edgeitems), int((a.shape[axis]-edgeitems)))
#         # omitted_idxs = (index + np.index_exp[edgeitems:-edgeitems])
#         omitted_indicies_list.append(omitted_idxs)
        
#         return np.concatenate((
#             _leading_trailing(a, edgeitems, left_edge_idxs),
#             _leading_trailing(a, edgeitems, right_edge_idxs)
#         ), axis=axis)#, omitted_idxs)
#     else:
#         # can include all items
#         # return (_leading_trailing(a, edgeitems, index + np.index_exp[:]), None)
#         return _leading_trailing(a, edgeitems, (index + np.index_exp[:]))

# omitted_indicies_list = []
# def _dropping_middle_leading_trailing(a: NDArray, edgeitems: int, index=(), n_fill_middle_items: int = 4): # , omitted_indicies_list=[]
#     """
#     Keep only the N-D corners (leading and trailing edges) of an array.

#     Should be passed a base-class ndarray, since it makes no guarantees about
#     preserving subclasses.
#     """
#     global omitted_indicies_list
#     axis = len(index)
#     if axis == a.ndim:
#         return a[index]
    
#     n_left_fill_middle_items = int(n_fill_middle_items / 2)
#     n_right_fill_middle_items = int(n_fill_middle_items / 2)
#     n_total_edgeitems: int = (2*edgeitems)
#     curr_axis_size: int = a.shape[axis]
    
#     if (curr_axis_size > n_total_edgeitems):
#         left_range = (0, (edgeitems-n_left_fill_middle_items))
#         right_range = ((-edgeitems + n_right_fill_middle_items), n_total_edgeitems)
#         middle_fill_range = ((left_range[-1]+1), (right_range[0]-1))
#         middle_fill_n_xbins = middle_fill_range[-1] - middle_fill_range[0]
#         print(f'left_range: {left_range}, right_range: {right_range}, middle_fill_range: {middle_fill_range}, middle_fill_n_xbins: {middle_fill_n_xbins}')
        
#         left_edge_idxs = (index + np.index_exp[:(edgeitems-n_left_fill_middle_items)])
#         right_edge_idxs = (index + np.index_exp[(-edgeitems + n_right_fill_middle_items):])
#         print(f'axis: {axis}, a.shape[axis]: {a.shape[axis]}') # index: {index}, 
#         # omitted_idxs = (left_edge_idxs[-1], right_edge_idxs[0])
#         omitted_idxs = (int(edgeitems), int((a.shape[axis]-edgeitems)))
#         # omitted_idxs = (index + np.index_exp[edgeitems:-edgeitems])
#         omitted_indicies_list.append(omitted_idxs)
        
#         middle_fill_array = np.full((middle_fill_n_xbins, *np.shape(a)[1:]), fill_value=np.nan)
        
        
#         return np.concatenate((
#             _leading_trailing(a, edgeitems, left_edge_idxs),
#             middle_fill_array,
#             _leading_trailing(a, edgeitems, right_edge_idxs)
#         ), axis=axis)#, omitted_idxs)
#     else:
#         # can include all items
#         # return (_leading_trailing(a, edgeitems, index + np.index_exp[:]), None)
#         return _leading_trailing(a, edgeitems, (index + np.index_exp[:]))
    

    

# max_allowed_arr_elements: int = 200

# arr = deepcopy(long_LR_pf1D_Decoder.p_x_given_n)
# arr_shape = np.array(np.shape(arr))
# arr_shape
# arr_too_large_dim = (arr_shape > max_allowed_arr_elements)
# arr_too_large_dim

# any_dim_arr_has_too_large_dimension: bool = np.any(arr_too_large_dim)

# edgeitems: int = int(max_allowed_arr_elements / 2)
# print(f'edgeitems: {edgeitems}')

# removed_items = [(a_len - (edgeitems * 2)) if is_too_large else 0 for a_len, is_too_large in zip(arr_shape, arr_too_large_dim)]
# # removed_items = arr_shape[arr_too_large_dim] - (edgeitems * 2)
# removed_items

# omitted_indicies_list = []
# # _corners = _leading_trailing(arr, edgeitems=edgeitems)
# _corners = _dropping_middle_leading_trailing(arr, edgeitems=edgeitems)

# _left_edge = _corners[:, :edgeitems]
# _right_edge = _corners[:, edgeitems:]
# # [_corners[:, :edgeitems], _corners[:, edgeitems:]]
# _corners

# omitted_indicies_list = np.array(omitted_indicies_list)
# omitted_indicies_list = np.squeeze(omitted_indicies_list.flatten()).tolist()
# omitted_indicies_list

# omission_indices = deepcopy(omitted_indicies_list)
# omission_indices = sorted(set(omission_indices))
# omission_indices

# ylims = [(0, arr_shape[0],)]
# ylims

# xlims = [(0, omission_indices[0],), (omission_indices[-1], arr_shape[1],)]
# xlims

