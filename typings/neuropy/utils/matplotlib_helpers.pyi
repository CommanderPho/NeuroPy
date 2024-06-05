"""
This type stub file was generated by pyright.
"""

import contextlib
from enum import unique
from attrs import define
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea
from matplotlib.widgets import RectangleSelector, SpanSelector
from neuropy.utils.misc import AutoNameEnum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Tuple
from matplotlib.figure import Figure
from flexitext.text import Text as StyledText

if TYPE_CHECKING:
    ...
Width_Height_Tuple = ...
def compute_data_extent(xpoints, *other_1d_series):
    """Computes the outer bounds, or "extent" of one or more 1D data series.

    Args:
        xpoints ([type]): [description]
        other_1d_series: any number of other 1d data series

    Returns:
        xmin, xmax, ymin, ymax, imin, imax, ...: a flat list of paired min, max values for each data series provided.
        
    Usage:
        # arbitrary number of data sequences:        
        xmin, xmax, ymin, ymax, x_center_min, x_center_max, y_center_min, y_center_max = compute_data_extent(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin, active_epoch_placefields2D.ratemap.xbin_centers, active_epoch_placefields2D.ratemap.ybin_centers)
        print(xmin, xmax, ymin, ymax, x_center_min, x_center_max, y_center_min, y_center_max)

        # simple 2D extent:
        extent = compute_data_extent(active_epoch_placefields2D.ratemap.xbin, active_epoch_placefields2D.ratemap.ybin)
        print(extent)
    """
    ...

def compute_data_aspect_ratio(xbin, ybin, sorted_inputs=...): # -> tuple[Any, Width_Height_Tuple]:
    """Computes the aspect ratio of the provided data

    Args:
        xbin ([type]): [description]
        ybin ([type]): [description]
        sorted_inputs (bool, optional): whether the input arrays are pre-sorted in ascending order or not. Defaults to True.

    Returns:
        float: The aspect ratio of the data such that multiplying any height by the returned float would result in a width in the same aspect ratio as the data.
    """
    ...

@unique
class enumTuningMap2DPlotMode(AutoNameEnum):
    PCOLORFAST = ...
    PCOLORMESH = ...
    PCOLOR = ...
    IMSHOW = ...


@unique
class enumTuningMap2DPlotVariables(AutoNameEnum):
    TUNING_MAPS = ...
    SPIKES_MAPS = ...
    OCCUPANCY = ...


def set_margins(fig, left=..., right=..., top=..., bottom=..., is_in_inches: bool = ...): # -> None:
    """Set figure margins as [left, right, top, bottom] in inches
    from the edges of the figure.
    

    You can set the rectangle that the layout engine operates within. See the rect parameter for each engine at https://matplotlib.org/stable/api/layout_engine_api.html.
    It's unfortunately not a very friendly part of the API, especially because TightLayoutEngine and ConstrainedLayoutEngine have different semantics for rect: TightLayoutEngine uses rect = (left, bottom, right, top) and ConstrainedLayoutEngine uses rect = (left, bottom, width, height).

    Usage:
        
        from neuropy.utils.matplotlib_helpers import set_margins
        
        #your margins were [0.2, 0.8, 0.2, 0.8] in figure coordinates
        #which are 0.2*11 and 0.2*8.5 in inches from the edge
        set_margins(a_fig, top=0.2) # [0.2*11, 0.2*11, 0.2*8.5, 0.2*8.5]
            
    
    """
    ...

def add_inner_title(ax, title, loc, strokewidth=..., stroke_foreground=..., stroke_alpha=..., text_foreground=..., font_size=..., text_alpha=..., use_AnchoredCustomText: bool = ..., **kwargs): # -> AnchoredCustomText:
    """
    Add a figure title inside the border of the figure (instead of outside).

    Args:
        ax (matplotlib.axes.Axes): The axes object where the title should be added.
        title (str): The title text.
        loc (str or int): The location code for the title placement.
        strokewidth (int, optional): The line width for the stroke around the text. Default is 3.
        stroke_foreground (str, optional): The color for the stroke around the text. Default is 'w' (white).
        stroke_alpha (float, optional): The alpha value for the stroke. Default is 0.9.
        text_foreground (str, optional): The color for the text. Default is 'k' (black).
        font_size (int, optional): The font size for the title text. If not provided, it will use the value from plt.rcParams['legend.title_fontsize'].
        text_alpha (float, optional): The alpha value for the text itself. Default is 1.0 (opaque).
        **kwargs: Additional keyword arguments to be passed to AnchoredText.

    Returns:
        matplotlib.offsetbox.AnchoredText: The AnchoredText object containing the title.
    """
    ...

def draw_sizebar(ax): # -> None:
    """
    Draw a horizontal bar with length of 0.1 in data coordinates,
    with a fixed label underneath.
    """
    ...

def set_ax_emphasis_color(ax, emphasis_color=..., defer_draw: bool = ...): # -> None:
    """ for the provided axis: changes the spine color, the x/y tick/labels color to the emphasis color. 
    """
    ...

def add_selection_patch(ax, selection_color=..., alpha=..., zorder=..., action_button_configs=..., debug_print=..., defer_draw: bool = ...): # -> tuple[Any, Any | None]:
    """ adds a rectangle behind the ax, sticking out to the right side by default.
    
    Can be toggled on/off via 
    `rectangle.set_visible(not rectangle.get_visible())`

    """
    ...

def build_or_reuse_figure(fignum=..., fig=..., fig_idx: int = ..., **kwargs):
    """ Reuses a Matplotlib figure if it exists, or creates a new one if needed
    Inputs:
        fignum - an int or str that identifies a figure
        fig - an existing Matplotlib figure
        fig_idx:int - an index to identify this figure as part of a series of related figures, e.g. plot_pf_1D[0], plot_pf_1D[1], ... 
        **kwargs - are passed as kwargs to the plt.figure(...) command when creating a new figure
    Outputs:
        fig: a Matplotlib figure object

    History: factored out of `plot_ratemap_2D`

    Usage:
        from neuropy.utils.matplotlib_helpers import build_or_reuse_figure
        
    Example 1:
        ## Figure Setup:
        fig = build_or_reuse_figure(fignum=kwargs.pop('fignum', None), fig=kwargs.pop('fig', None), fig_idx=kwargs.pop('fig_idx', 0), figsize=kwargs.pop('figsize', (10, 4)), dpi=kwargs.pop('dpi', None), constrained_layout=True) # , clear=True
        subfigs = fig.subfigures(actual_num_subfigures, 1, wspace=0.07)
        ##########################

    Example 2:
        
        if fignum is None:
            if f := plt.get_fignums():
                fignum = f[-1] + 1
            else:
                fignum = 1

        ## Figure Setup:
        if ax is None:
            fig = build_or_reuse_figure(fignum=fignum, fig=fig, fig_idx=0, figsize=(12, 4.2), dpi=None, clear=True, tight_layout=False)
            gs = GridSpec(1, 1, figure=fig)

            if use_brokenaxes_method:
                # `brokenaxes` method: DOES NOT YET WORK!
                from brokenaxes import brokenaxes ## Main brokenaxes import 
                pad_size: float = 0.1
                # [(a_tuple.start, a_tuple.stop) for a_tuple in a_test_epoch_df.itertuples(index=False, name="EpochTuple")]
                lap_start_stop_tuples_list = [((a_tuple.start - pad_size), (a_tuple.stop + pad_size)) for a_tuple in ensure_dataframe(laps_Epoch_obj).itertuples(index=False, name="EpochTuple")]
                # ax = brokenaxes(xlims=((0, .1), (.4, .7)), ylims=((-1, .7), (.79, 1)), hspace=.05, subplot_spec=gs[0])
                ax = brokenaxes(xlims=lap_start_stop_tuples_list, hspace=.05, subplot_spec=gs[0])
            else:
                ax = plt.subplot(gs[0])

        else:
            # otherwise get the figure from the passed axis
            fig = ax.get_figure()
                    
            
    """
    ...

def scale_title_label(ax, curr_title_obj, curr_im, debug_print=...): # -> None:
    """ Scales some matplotlib-based figures titles to be reasonable. I remember that this was important and hard to make, but don't actually remember what it does as of 2022-10-24. It needs to be moved in to somewhere else.
    

    History: From PendingNotebookCode's 2022-11-09 section


    Usage:

        from neuropy.utils.matplotlib_helpers import scale_title_label

        ## Scale all:
        _display_outputs = widget.last_added_display_output
        curr_graphics_objs = _display_outputs.graphics[0]

        ''' curr_graphics_objs is:
        {2: {'axs': [<Axes:label='2'>],
        'image': <matplotlib.image.AxesImage at 0x1630c4556d0>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c4559a0>},
        4: {'axs': [<Axes:label='4'>],
        'image': <matplotlib.image.AxesImage at 0x1630c455f70>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c463280>},
        5: {'axs': [<Axes:label='5'>],
        'image': <matplotlib.image.AxesImage at 0x1630c463850>,
        'title_obj': <matplotlib.offsetbox.AnchoredText at 0x1630c463b20>},
        ...
        '''
        for aclu, curr_neuron_graphics_dict in curr_graphics_objs.items():
            curr_title_obj = curr_neuron_graphics_dict['title_obj'] # matplotlib.offsetbox.AnchoredText
            curr_title_text_obj = curr_title_obj.txt.get_children()[0] # Text object
            curr_im = curr_neuron_graphics_dict['image'] # matplotlib.image.AxesImage
            curr_ax = curr_neuron_graphics_dict['axs'][0]
            scale_title_label(curr_ax, curr_title_obj, curr_im)

    
    """
    ...

def add_value_labels(ax, spacing=..., labels=...): # -> None:
    """Add labels to the end (top) of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes of the plot to annotate.
        spacing (int): The distance between the labels and the bars.

    History:
        Factored out of `plot_short_v_long_pf1D_scalar_overlap_comparison` on 2023-03-28

    Usage:
        from neuropy.utils.matplotlib_helpers import add_value_labels
        # Call the function above. All the magic happens there.
        add_value_labels(ax, labels=x_labels) # 

    """
    ...

def fit_both_axes(ax_lhs, ax_rhs): # -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    """ 2023-05-25 - Computes the x and y bounds needed to fit all data on both axes, and the actually applies these bounds to each. """
    ...

@define(slots=False)
class FigureMargins:
    top_margin: float = ...
    left_margin: float = ...
    right_margin: float = ...
    bottom_margin: float = ...


@define(slots=False)
class FormattedFigureText:
    """ builds flexitext matplotlib figure title and footers 

    Consistent color scheme:
        Long: Red
        Short: Blue

        Context footer is along the bottom of the figure in gray.


    Usage:
        
        from neuropy.utils.matplotlib_helpers import FormattedFigureText

        # `flexitext` version:
        text_formatter = FormattedFigureText()
        plt.title('')
        plt.suptitle('')
        text_formatter.setup_margins(fig)

        ## Need to extract the track name ('maze1') for the title in this plot. 
        track_name = active_context.get_description(subset_includelist=['filter_name'], separator=' | ') # 'maze1'
        # TODO: do we want to convert this into "long" or "short"?
        header_text_obj = flexitext(text_formatter.left_margin, text_formatter.top_margin, f'<size:22><weight:bold>{track_name}</> replay|laps <weight:bold>firing rate</></>', va="bottom", xycoords="figure fraction")
        footer_text_obj = flexitext((text_formatter.left_margin*0.1), (text_formatter.bottom_margin*0.25), text_formatter._build_footer_string(active_context=active_context), va="top", xycoords="figure fraction")



    """
    margins: FigureMargins = ...
    @property
    def top_margin(self): # -> float:
        ...
    
    @top_margin.setter
    def top_margin(self, value): # -> None:
        ...
    
    @property
    def left_margin(self): # -> float:
        ...
    
    @left_margin.setter
    def left_margin(self, value): # -> None:
        ...
    
    @property
    def right_margin(self): # -> float:
        ...
    
    @right_margin.setter
    def right_margin(self, value): # -> None:
        ...
    
    @property
    def bottom_margin(self): # -> float:
        ...
    
    @bottom_margin.setter
    def bottom_margin(self, value): # -> None:
        ...
    
    def setup_margins(self, fig, **kwargs): # -> None:
        ...
    
    def add_flexitext_context_footer(self, active_context, override_left_margin_multipler: float = ..., override_bottom_margin_multiplier: float = ...):
        """ adds the default footer  """
        ...
    
    def add_flexitext(self, fig, active_context, **kwargs): # -> tuple[Any, Any]:
        ...
    
    @classmethod
    def clear_basic_titles(self, fig): # -> None:
        """ clears the basic title and suptitle in preparation for the flexitext version. """
        ...
    


def plot_position_curves_figure(position_obj, include_velocity=..., include_accel=..., figsize=...): # -> tuple[Any, list[Any]]:
    """ Renders a figure with a position curve and optionally its higher-order derivatives """
    ...

def draw_epoch_regions(epoch_obj, curr_ax, facecolor=..., edgecolors=..., alpha=..., labels_kwargs=..., defer_render=..., debug_print=..., **kwargs): # -> tuple[list[Any], list[Any] | None] | tuple[Any, list[Any] | None]:
    """ plots epoch rectangles with customizable color, edgecolor, and labels on an existing matplotlib axis
    2022-12-14

    Info:
    
    https://matplotlib.org/stable/tutorials/intermediate/autoscale.html
    
    Usage:
        from neuropy.utils.matplotlib_helpers import draw_epoch_regions
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, defer_render=False, debug_print=False)

    Full Usage Examples:

    ## Example 1:
        active_filter_epochs = curr_active_pipeline.sess.replay
        active_filter_epochs

        if not 'stop' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['stop'] = active_filter_epochs['end'].copy()
            
        if not 'label' in active_filter_epochs.columns:
            # Make sure it has the 'stop' column which is expected as opposed to the 'end' column
            active_filter_epochs['label'] = active_filter_epochs['flat_replay_idx'].copy()

        active_filter_epoch_obj = Epoch(active_filter_epochs)
        active_filter_epoch_obj


        fig, ax = plt.subplots()
        ax.plot(post_update_times, flat_surprise_across_all_positions)
        ax.set_ylabel('Relative Entropy across all positions')
        ax.set_xlabel('t (seconds)')
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, facecolor=('red','cyan'), alpha=0.1, edgecolors=None, labels_kwargs={'y_offset': -0.05, 'size': 14}, defer_render=True, debug_print=False)
        laps_epochs_collection, laps_epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.laps.as_epoch_obj(), ax, facecolor='red', edgecolors='black', labels_kwargs={'y_offset': -16.0, 'size':8}, defer_render=True, debug_print=False)
        replays_epochs_collection, replays_epoch_labels = draw_epoch_regions(active_filter_epoch_obj, ax, facecolor='orange', edgecolors=None, labels_kwargs=None, defer_render=False, debug_print=False)
        fig.show()


    ## Example 2:

        # Show basic relative entropy vs. time plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(post_update_times, flat_relative_entropy_results)
        ax.set_ylabel('Relative Entropy')
        ax.set_xlabel('t (seconds)')
        epochs_collection, epoch_labels = draw_epoch_regions(curr_active_pipeline.sess.epochs, ax, defer_render=False, debug_print=False)
        fig.show()

    """
    ...

def plot_overlapping_epoch_analysis_diagnoser(position_obj, epoch_obj): # -> tuple[Any, list[Any]]:
    """ builds a MATPLOTLIB figure showing the position and velocity overlayed by the epoch intervals in epoch_obj. Useful for diagnosing overlapping epochs.
    Usage:
        from neuropy.utils.matplotlib_helpers import plot_overlapping_epoch_analysis_diagnoser
        fig, out_axes_list = plot_overlapping_epoch_analysis_diagnoser(sess.position, curr_active_pipeline.sess.laps.as_epoch_obj())
    """
    ...

class MatplotlibFigureExtractors:
    """ 2023-06-26 - Unfinished class that aims to extract matplotlib.figure properties and settings.
    Usage:
        from neuropy.utils.matplotlib_helpers import MatplotlibFigureExtractors
    
    """
    @staticmethod
    def extract_figure_properties(fig): # -> dict[Any, Any]:
        """ UNTESTED, UNFINISHED
        Extracts styles, formatting, and set options from a matplotlib Figure object.
        Returns a dictionary with the following keys:
            - 'title': the Figure title (if any)
            - 'xlabel': the label for the x-axis (if any)
            - 'ylabel': the label for the y-axis (if any)
            - 'xlim': the limits for the x-axis (if any)
            - 'ylim': the limits for the y-axis (if any)
            - 'xscale': the scale for the x-axis (if any)
            - 'yscale': the scale for the y-axis (if any)
            - 'legend': the properties of the legend (if any)
            - 'grid': the properties of the grid (if any)
            
        TO ADD:
            -   fig.get_figwidth()
                fig.get_figheight()
                # fig.set_figheight()

                print(f'fig.get_figwidth(): {fig.get_figwidth()}\nfig.get_figheight(): {fig.get_figheight()}')


            
            Usage:        
                curr_fig = plt.gcf()
                curr_fig = out.figures[0]
                curr_fig_properties = extract_figure_properties(curr_fig)
                curr_fig_properties

        """
        ...
    
    @classmethod
    def extract_fig_suptitle(cls, fig: Figure): # -> tuple[Any, Any | str]:
        """To get the figure's suptitle Text object: https://stackoverflow.com/questions/48917631/matplotlib-how-to-return-figure-suptitle

        Usage:
            from matplotlib.figure import Figure
            from neuropy.utils.matplotlib_helpers import MatplotlibFigureExtractors

            sup, suptitle_string = MatplotlibFigureExtractors.extract_fig_suptitle(fig)
            suptitle_string

        """
        ...
    
    @classmethod
    def extract_titles(cls, fig: Optional[Figure] = ...): # -> dict[Any, Any]:
        """ 
        # Call the function to extract titles
            captured_titles = extract_titles()
            print(captured_titles)
        """
        ...
    


def add_range_selector(fig, ax, initial_selection=..., orientation=..., on_selection_changed=...) -> SpanSelector:
    """ 2023-06-06 - a 1D version of `add_rectangular_selector` which adds a selection band to an existing axis

    from neuropy.utils.matplotlib_helpers import add_range_selector
    curr_pos = deepcopy(curr_active_pipeline.sess.position)
    curr_pos_df = curr_pos.to_dataframe()

    curr_pos_df.plot(x='t', y=['lin_pos'])
    fig, ax = plt.gcf(), plt.gca()
    range_selector, set_extents = add_range_selector(fig, ax, orientation="vertical", initial_selection=None) # (-86.91, 141.02)

    """
    ...

def add_rectangular_selector(fig, ax, initial_selection=..., on_selection_changed=..., selection_rect_props=..., **kwargs) -> RectangleSelector:
    """ 2023-05-16 - adds an interactive rectangular selector to an existing matplotlib figure/ax.
    
    Usage:
    
        from neuropy.utils.matplotlib_helpers import add_rectangular_selector

        fig, ax = curr_active_pipeline.computation_results['maze'].computed_data.pf2D.plot_occupancy()
        rect_selector, set_extents = add_rectangular_selector(fig, ax, initial_selection=grid_bin_bounds) # (24.82, 257.88), (125.52, 149.19)

    
    The returned RectangleSelector object can have its selection accessed via:
        rect_selector.extents # (25.508610487986658, 258.5627661142404, 128.10121504465053, 150.48449186696848)
    
    Or updated via:
        rect_selector.extents = (25, 258, 128, 150)

    """
    ...

def interactive_select_grid_bin_bounds_1D(curr_active_pipeline, epoch_name=...): # -> tuple[Any, Any, Any, Any]:
    """ allows the user to interactively select the grid_bin_bounds for the pf1D
    
    Usage:
        from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_1D
        fig, ax, range_selector, set_extents = interactive_select_grid_bin_bounds_1D(curr_active_pipeline, epoch_name='maze')
    """
    ...

def interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name=..., should_block_for_input: bool = ..., should_apply_updates_to_pipeline=..., selection_rect_props=..., **kwargs): # -> tuple[Any, Any, Any, Any, Any] | None:
    """ allows the user to interactively select the grid_bin_bounds for the pf2D
    Uses:
        plot_occupancy, add_rectangular_selector


    Usage:
        from neuropy.utils.matplotlib_helpers import interactive_select_grid_bin_bounds_2D
        fig, ax, rect_selector, set_extents, reset_extents = interactive_select_grid_bin_bounds_2D(curr_active_pipeline, epoch_name='maze')
    """
    ...

def perform_update_title_subtitle(fig, ax, title_string: Optional[str], subtitle_string: Optional[str], active_context=..., use_flexitext_titles=...): # -> None:
    """ Only updates the title/subtitle if the value is not None
    
    Usage:
    
    from neuropy.utils.matplotlib_helpers import perform_update_title_subtitle
    perform_update_title_subtitle(fig=fig_long_pf_1D, ax=ax_long_pf_1D, title_string="TEST - 1D Placemaps", subtitle_string="TEST - SUBTITLE")
    
    """
    ...

def matplotlib_configuration_update(is_interactive: bool, backend: Optional[str] = ...): # -> Callable[[], None]:
    """Non-Context manager version for configuring Matplotlib interactivity, backend, and toolbar.
    
    The context-manager version notabily doesn't work for making the figures visible, I think because when it leaves the context handler the variables assigned within go away and thus the references to the Figures are lost.
    
    # Example usage:

        from neuropy.utils.matplotlib_helpers import matplotlib_configuration
        with matplotlib_configuration(is_interactive=False, backend='AGG'):
            # Perform non-interactive Matplotlib operations with 'AGG' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode with AGG Backend')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

        with matplotlib_configuration(is_interactive=True, backend='Qt5Agg'):
            # Perform interactive Matplotlib operations with 'Qt5Agg' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Interactive Mode with Qt5Agg Backend')
            plt.show()  # Display the plot interactively (interactive mode)
    """
    ...

@contextlib.contextmanager
def matplotlib_backend(backend: str): # -> Generator[None, Any, None]:
    """Context manager for switching Matplotlib backend and safely restoring it to its previous value when done.
        # Example usage:
        with matplotlib_backend('AGG'):
            # Perform non-interactive Matplotlib operations
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

        with matplotlib_backend('Qt5Agg'):
            # Perform interactive Matplotlib operations
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Interactive Mode')
            plt.show()  # Display the plot interactively (interactive mode)
    """
    ...

@contextlib.contextmanager
def matplotlib_interactivity(is_interactive: bool): # -> Generator[None, Any, None]:
    """Context manager for switching Matplotlib interactivity mode and safely restoring it to its previous value when done.

    # Example usage:
    with matplotlib_interactivity(is_interactive=False):
        # Perform non-interactive Matplotlib operations
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Non-interactive Mode')
        plt.show()  # Display the plot (if desired)


    with matplotlib_interactivity(is_interactive=True):
        # Perform interactive Matplotlib operations
        plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('Interactive Mode')
        plt.show()  # Display the plot immediately (if desired)
    """
    ...

@contextlib.contextmanager
def disable_function_context(obj, fn_name: str): # -> Generator[None, Any, None]:
    """ Disables a function within a context manager

    https://stackoverflow.com/questions/10388411/possible-to-globally-replace-a-function-with-a-context-manager-in-python

    Could be used for plt.show().
    ```python
    
    from neuropy.utils.matplotlib_helpers import disable_function_context
    import matplotlib.pyplot as plt
    with disable_function_context(plt, "show"):
        run_me(x)
    
    """
    ...

@contextlib.contextmanager
def matplotlib_configuration(is_interactive: bool, backend: Optional[str] = ...): # -> Generator[None, Any, None]:
    """Context manager for configuring Matplotlib interactivity, backend, and toolbar.
    # Example usage:

        from neuropy.utils.matplotlib_helpers import matplotlib_configuration
        with matplotlib_configuration(is_interactive=False, backend='AGG'):
            # Perform non-interactive Matplotlib operations with 'AGG' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode with AGG Backend')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)

        with matplotlib_configuration(is_interactive=True, backend='Qt5Agg'):
            # Perform interactive Matplotlib operations with 'Qt5Agg' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Interactive Mode with Qt5Agg Backend')
            plt.show()  # Display the plot interactively (interactive mode)
    """
    ...

@contextlib.contextmanager
def matplotlib_file_only(): # -> Generator[None, Any, None]:
    """Context manager for configuring Matplotlib to only render to file, using the 'AGG' backend, no interactivity, and no plt.show()
    # Example usage:
        from neuropy.utils.matplotlib_helpers import matplotlib_file_only
        with matplotlib_file_only():
            # Perform non-interactive Matplotlib operations with 'AGG' backend
            plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.title('Non-interactive Mode with AGG Backend')
            plt.savefig('plot.png')  # Save the plot to a file (non-interactive mode)
    """
    ...

def resize_window_to_inches(window, width_inches, height_inches, dpi=...): # -> None:
    """ takes a matplotlib figure size (specified in inches) and the figure dpi to compute the matching pixel size. # If you render a matplotlib figure in a pyqt5 backend window, you can appropriately set the size of this window using this function.

    # Example usage:
        # Assuming you have a QMainWindow instance named 'main_window'
        size=(5,12)
        resize_window_to_inches(mw.window(), *size)

    """
    ...

@define(slots=False)
class ValueFormatter:
    """ builds text formatting (for example larger values being rendered larger or more red) 

    Usage:
        a_val_formatter = ValueFormatter()
        a_val_formatter(0.934)
    """
    NONE_fallback_color: str = ...
    nan_fallback_color: str = ...
    out_of_range_fallback_color: str = ...
    cmap: mpl.colors.Colormap = ...
    norm: mpl.colors.Normalize = ...
    coloring_function: Callable = ...
    def __attrs_post_init__(self): # -> None:
        ...
    
    def value_to_color(self, value, debug_print=...) -> str:
        """ Maps a value between -1.0 and 1.0 to an RGB color code. Returns a hex-formatted color string
        """
        ...
    
    def value_to_format_dict(self, value, debug_print=...) -> Dict[str, Any]:
        """ Returns a formatting dict for rendering the value text suitable for use with flexitext_value_textprops

        Returns a formatting dict for rendering the value text suitable for use with flexitext_value_textprops

        """
        ...
    
    def matplotlib_colormap_value_to_color_fn(self, value, debug_print=...) -> str:
        """ uses self.cmap and self.norm to format the value. """
        ...
    
    def blue_grey_red_custom_value_to_color_fn(self, value, debug_print=...) -> str:
        """
        Maps a value between -1.0 and 1.0 to an RGB color code.
        -1.0 maps to bright blue, 0.0 maps to dark gray, and 1.0 maps to bright red.

        Returns a hex-formatted color string
        Does not use the matplotlib colormap properties.
        """
        ...
    


def parse_and_format_unformatted_values_text(unformatted_text_block: str, key_value_split: str = ..., desired_label_value_sep: str = ..., a_val_formatter: Optional[ValueFormatter] = ...) -> Tuple[List[StyledText], ValueFormatter]:
    """ takes a potentially multi-line string containing keys and values like:
        unformatted_text_block: str = "wcorr: -0.754\n$P_i$: 0.052\npearsonr: -0.76"
    to produce a list of flexitext.Text objects that contain styled text that can be rendered.

    
    desired_label_value_sep: str - the desired label/value separator to be rendered in the final string:

    
    Usage:
        ## FLEXITEXT-version
        from flexitext import FlexiText, Style
        from flexitext.textgrid import make_text_grid, make_grid
        from flexitext.text import Text as StyledText

        unformatted_text_block: str = "wcorr: -0.754\n$P_i$: 0.052\npearsonr: -0.76"
        texts: List[StyledText] = parse_and_format_unformatted_values_text(test_test)
        text_grid: VPacker = make_text_grid(texts, ha="right")
        text_grid


    """
    ...

class AnchoredCustomText(AnchoredOffsetbox):
    """
    AnchoredOffsetbox with Text.

    
    Usage:
        from typing import Tuple
        import matplotlib.pyplot as plt
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
        from neuropy.utils.matplotlib_helpers import AnchoredCustomText, build_formatted_label_values_stack, build_formatted_label_values_stack, value_to_color
                                
        # Create a figure and axis
        fig, ax = plt.subplots()
        formated_text_list = [("wcorr: ", -0.754),
                                ("$P_i$: ", 0.052), 
                                ("pearsonr: ", -0.76),
                            ]

        text_kwargs = _helper_build_text_kwargs_flat_top(a_curr_ax=ax)

        anchored_custom_text = AnchoredCustomText(formated_text_list=formated_text_list, pad=0., frameon=False,**text_kwargs, borderpad=0.)
        # anchored_box = AnchoredOffsetbox(child=stack_box, pad=0., frameon=False,**text_kwargs, borderpad=0.)

        # Add the offset box to the axes
        ax.add_artist(anchored_custom_text)

        # Display the plot
        plt.show()
            
            
    """
    def __init__(self, unformatted_text_block: str, custom_value_formatter: Optional[ValueFormatter] = ..., **kwargs) -> None:
        """
        Parameters
        ----------
        **kwargs
            All other parameters are passed to `AnchoredOffsetbox`.
        """
        ...
    
    @property
    def text_areas(self) -> List[TextArea]:
        """The text_areas property."""
        ...
    
    @property
    def text_objs(self) -> List[Text]:
        """The matplotlib.Text objects. """
        ...
    
    def update_text_alpha(self, value: float): # -> None:
        ...
    
    def update_text(self, unformatted_text_block: str) -> bool:
        """ not yet working """
        ...
    


