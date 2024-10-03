"""
This type stub file was generated by pyright.
"""

import numpy as np
from typing import Any, Dict, List
from neuropy.utils.result_context import IdentifyingContext
from neuropy.utils.mixins.AttrsClassHelpers import AttrsBasedClassHelperMixin, custom_define
from neuropy.utils.mixins.HDF5_representable import HDFMixin
from pyphocorehelpers.function_helpers import function_attributes

@custom_define(slots=False)
class SessionCellExclusivityRecord:
    """ 2023-10-04 - Holds hardcoded specifiers indicating whether a cell is LxC/SxC/etc """
    LxC: np.ndarray = ...
    LpC: np.ndarray = ...
    Others: np.ndarray = ...
    SpC: np.ndarray = ...
    SxC: np.ndarray = ...


@custom_define(slots=False)
class UserAnnotationsManager(HDFMixin, AttrsBasedClassHelperMixin):
    """ class for holding User Annotations of the data. Performed interactive by the user, and then saved to disk for later use. An example are the selected replays to be used as examples. 
    
    Usage:
        from neuropy.core.user_annotations import UserAnnotationsManager
        
    """
    annotations: Dict[IdentifyingContext, Any] = ...
    def __attrs_post_init__(self): # -> None:
        """ builds complete self.annotations from all the separate hardcoded functions. """
        ...
    
    @function_attributes(short_name=None, tags=['XxC', 'LxC', 'SxC'], input_requires=[], output_provides=[], uses=[], used_by=[], creation_date='2023-10-05 16:18', related_items=[])
    def add_neuron_exclusivity_column(self, neuron_indexed_df, included_session_contexts, neuron_uid_column_name=...):
        """ adds 'XxC_status' column to the `neuron_indexed_df`: the user-labeled cell exclusivity (LxC/SxC/Shared) status {'LxC', 'SxC', 'Shared'}
        
            annotation_man = UserAnnotationsManager()
            long_short_fr_indicies_analysis_table = annotation_man.add_neuron_exclusivity_column(long_short_fr_indicies_analysis_table, included_session_contexts, aclu_column_name='neuron_id')
            long_short_fr_indicies_analysis_table
    
        """
        ...
    
    @staticmethod
    def get_user_annotations(): # -> dict[Any, Any]:
        """ hardcoded user annotations
        

        New Entries can be generated like:
            from pyphoplacecellanalysis.GUI.Qt.Mixins.PaginationMixins import SelectionsObject
            from neuropy.core.user_annotations import UserAnnotationsManager
            from pyphoplacecellanalysis.Pho2D.stacked_epoch_slices import DecodedEpochSlicesPaginatedFigureController

            ## Stacked Epoch Plot
            example_stacked_epoch_graphics = curr_active_pipeline.display('_display_long_and_short_stacked_epoch_slices', defer_render=False, save_figure=True)
            pagination_controller_L, pagination_controller_S = example_stacked_epoch_graphics.plot_data['controllers']
            ax_L, ax_S = example_stacked_epoch_graphics.axes
            final_figure_context_L, final_context_S = example_stacked_epoch_graphics.context

            user_annotations = UserAnnotationsManager.get_user_annotations()

            ## Capture current user selection
            saved_selection_L: SelectionsObject = pagination_controller_L.save_selection()
            saved_selection_S: SelectionsObject = pagination_controller_S.save_selection()
            final_L_context = saved_selection_L.figure_ctx.adding_context_if_missing(user_annotation='selections')
            final_S_context = saved_selection_S.figure_ctx.adding_context_if_missing(user_annotation='selections')
            user_annotations[final_L_context] = saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected]
            user_annotations[final_S_context] = saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected]
            # Updates the context. Needs to generate the code.

            ## Generate code to insert int user_annotations:
            print('Add the following code to UserAnnotationsManager.get_user_annotations() function body:')
            print(f"user_annotations[{final_L_context.get_initialization_code_string()}] = np.array({list(saved_selection_L.flat_all_data_indicies[saved_selection_L.is_selected])})")
            print(f"user_annotations[{final_S_context.get_initialization_code_string()}] = np.array({list(saved_selection_S.flat_all_data_indicies[saved_selection_S.is_selected])})")


        Usage:
            user_anootations = get_user_annotations()
            user_anootations

        """
        ...
    
    @classmethod
    def has_user_annotation(cls, test_context): # -> bool:
        ...
    
    @classmethod
    def get_hardcoded_specific_session_cell_exclusivity_annotations_dict(cls) -> dict:
        """ hand-labeled by pho on 2023-10-04 """
        ...
    
    @classmethod
    def get_hardcoded_specific_session_override_dict(cls) -> dict:
        """ Extracted from `neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat` 
            ## Create a dictionary of overrides that have been specified manually for a given session:
            # Used in `build_lap_only_short_long_bin_aligned_computation_configs`

            Usage:
                ## Get specific grid_bin_bounds overrides from the `UserAnnotationsManager._specific_session_override_dict`
                override_dict = UserAnnotationsManager.get_hardcoded_specific_session_override_dict().get(sess.get_context(), {})
                if override_dict.get('grid_bin_bounds', None) is not None:
                    grid_bin_bounds = override_dict['grid_bin_bounds']
                else:
                    # no overrides present
                    pos_df = sess.position.to_dataframe().copy()
                    if not 'lap' in pos_df.columns:
                        pos_df = sess.compute_laps_position_df() # compute the lap column as needed.
                    laps_pos_df = pos_df[pos_df.lap.notnull()] # get only the positions that belong to a lap
                    laps_only_grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(laps_pos_df.x.to_numpy(), laps_pos_df.y.to_numpy()) # compute the grid_bin_bounds for these positions only during the laps. This means any positions outside of this will be excluded!
                    print(f'\tlaps_only_grid_bin_bounds: {laps_only_grid_bin_bounds}')
                    grid_bin_bounds = laps_only_grid_bin_bounds
                    # ## Determine the grid_bin_bounds from the long session:
                    # grid_bin_bounds = PlacefieldComputationParameters.compute_grid_bin_bounds(sess.position.x, sess.position.y) # ((22.736279243974774, 261.696733348342), (125.5644705153173, 151.21507349463707))
                    # # refined_grid_bin_bounds = ((24.12, 259.80), (130.00, 150.09))
                    # DO INTERACTIVE MODE:
                    # grid_bin_bounds = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze', should_block_for_input=True)
                    # print(f'grid_bin_bounds: {grid_bin_bounds}')
                    # print(f"Add this to `specific_session_override_dict`:\n\n{curr_active_pipeline.get_session_context().get_initialization_code_string()}:dict(grid_bin_bounds=({(grid_bin_bounds[0], grid_bin_bounds[1]), (grid_bin_bounds[2], grid_bin_bounds[3])})),\n")


        """
        ...
    
    @classmethod
    def get_hardcoded_good_sessions(cls) -> List[IdentifyingContext]:
        """Hardcoded included_session_contexts:
                
        Usage:
            from neuropy.core.user_annotations import UserAnnotationsManager
        
            included_session_contexts: List[IdentifyingContext] = UserAnnotationsManager.get_hardcoded_good_sessions()
            
            
        """
        ...
    
    @classmethod
    def get_hardcoded_bad_sessions(cls) -> List[IdentifyingContext]:
        """ Hardcoded excluded_session_contexts:
        
        Usage:
            from neuropy.core.user_annotations import UserAnnotationsManager
        
            excluded_session_contexts: List[IdentifyingContext] = UserAnnotationsManager.get_hardcoded_bad_sessions()
            bad_session_df: pd.DataFrame = pd.DataFrame.from_records([v.to_dict() for v in excluded_session_contexts], columns=['format_name', 'animal', 'exper_name', 'session_name'])
            bad_session_df

        Built Via:
            from neuropy.core.session.Formats.Specific.KDibaOldDataSessionFormat import KDibaOldDataSessionFormatRegisteredClass
            bad_sessions_csv_path = Path(r'~/repos/matlab-to-neuropy-exporter/output/2024-09-23_bad_sessions_table.csv').resolve() ## exported from `IIDataMat_Export_ToPython_2022_08_01.m`
            bad_session_df, bad_session_contexts = KDibaOldDataSessionFormatRegisteredClass.load_bad_sessions_csv(bad_sessions_csv_path=bad_sessions_csv_path)        
            
        
        """
        ...
    


