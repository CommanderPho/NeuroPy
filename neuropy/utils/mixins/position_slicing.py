from typing import Dict, List, Tuple, Optional
import pandas as pd

class PositionSlicedMixin:
    """ Used in Position-based classes to enable spatial filtering/slicing operations 
    
    from neuropy.utils.mixins.position_slicing import PositionSlicedMixin
    
    
    """
    
    @property
    def position_variable_names(self) -> Tuple[str, ...]:
        """ Returns tuple of position dimension column names ('x', 'y', 'z') """
        if 'y' in self._obj.columns:
            return ('x', 'y')
        return ('x',)

    @property
    def binned_position_variable_names(self) -> Tuple[str, ...]:
        """ Returns tuple of binned position column names ('binned_x', 'binned_y') """
        if 'y' in self._obj.columns:
            return ('binned_x', 'binned_y')
        return ('binned_x',)

    def position_sliced(self, xmin: Optional[float]=None, xmax: Optional[float]=None, ymin: Optional[float]=None, ymax: Optional[float]=None, xmin_xmax_tuple: Optional[Tuple[float, float]]=None, ymin_ymax_tuple: Optional[Tuple[float, float]]=None, grid_bin_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]=None) -> pd.DataFrame:
        """ Returns a copy of the dataframe filtered to only include positions within the specified bounds """
        # Build filter conditions
        filter_conditions = []
        if grid_bin_bounds is not None:
            (xmin, xmax), (ymin, ymax) = grid_bin_bounds
            assert ((xmin_xmax_tuple is None) and (ymin_ymax_tuple is None)), "only one mutually exclusive argument allowed at a time!"
        elif (xmin_xmax_tuple is not None) or (ymin_ymax_tuple is not None):
            if xmin_xmax_tuple is not None:
                xmin, xmax = xmin_xmax_tuple
            if ymin_ymax_tuple is not None:
                ymin, ymax = ymin_ymax_tuple
    
        if xmin is not None and xmax is not None:
            filter_conditions.append((self._obj['x'] >= xmin) & (self._obj['x'] <= xmax))
    
        if 'y' in self._obj.columns and ymin is not None and ymax is not None:
            filter_conditions.append((self._obj['y'] >= ymin) & (self._obj['y'] <= ymax))
    
        # Combine all conditions
        if filter_conditions:
            is_pos_sample_included = filter_conditions[0]
            for condition in filter_conditions[1:]:
                is_pos_sample_included = is_pos_sample_included & condition
                
            filtered_df = self._obj[is_pos_sample_included]
        else:
            filtered_df = self._obj
            
        return filtered_df.copy()
