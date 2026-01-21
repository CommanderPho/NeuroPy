---
name: Implement trajectory simplification function
overview: Implement a function to simplify position trajectories using the Ramer-Douglas-Peucker (RDP) algorithm, returning a boolean mask for filtering the original dataframe.
todos:
  - id: add_import
    content: Add 'from rdp import rdp' import at top of position_util.py
    status: completed
  - id: implement_function
    content: Implement simplify_position_trajectory() function with RDP algorithm, input validation, and boolean mask return
    status: completed
  - id: replace_todo
    content: Replace TODO comment (lines 397-403) with the new function implementation
    status: completed
isProject: false
---

# Implement Trajectory Simplification Function

## Overview

Implement `simplify_position_trajectory()` function in `[neuropy/utils/position_util.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\position_util.py)` to downsample position trajectories while preserving path features. The function will use the RDP algorithm (already in dependencies) and return a boolean mask for filtering.

## Implementation Details

### Function Signature

```python
def simplify_position_trajectory(position_df: pd.DataFrame, epsilon: float = 0.5, algorithm: str = "rdp", algo: str = "iter") -> pd.Series:
```

**Parameters:**

- `position_df`: DataFrame with ['x', 'y'] columns
- `epsilon`: Simplification tolerance (default: 0.5). Higher values = more aggressive simplification
- `algorithm`: Algorithm choice - "rdp" (default) or "visvalingam" for future support
- `algo`: For RDP, use "iter" (default) to enable return_mask support

**Returns:**

- `pd.Series`: Boolean mask (same length as position_df) where True indicates point is kept

### Implementation Steps

1. **Add function at line 397** (replace the TODO comment)
  - Import `rdp` from `rdp` library at the top of the file
  - Validate input: check for ['x', 'y'] columns
  - Extract x, y coordinates as numpy array
  - Handle edge cases (empty df, single point, two points)
  - Call `rdp()` with `return_mask=True` and `algo="iter"`
  - Return boolean mask as pd.Series with same index as input df
2. **Error Handling**
  - Validate position_df has ['x', 'y'] columns
  - Handle cases with < 3 points (RDP requires at least 3 points)
  - Handle NaN values in x, y columns
3. **Code Style**
  - Function signature on single line (per user rules)
  - Follow existing code patterns in the file
  - Add docstring following existing function style in the file

### Example Usage

```python
position_df = ...  # DataFrame with ['x', 'y'] columns
mask = simplify_position_trajectory(position_df, epsilon=0.5)
simplified_df = position_df.loc[mask]
```

## Files to Modify

- `[neuropy/utils/position_util.py](h:\TEMP\Spike3DEnv_ExploreUpgrade\Spike3DWorkEnv\NeuroPy\neuropy\utils\position_util.py)`
  - Add import: `from rdp import rdp` (near top with other imports)
  - Replace TODO comment (lines 397-403) with function implementation

## Notes

- The `rdp` library is already in dependencies (pyproject.toml)
- RDP with `algo="iter"` and `return_mask=True` directly returns a boolean mask
- For future extensibility, the function signature includes `algorithm` parameter (currently only "rdp" implemented)
- The mask preserves the original dataframe index for proper filtering with `.loc[mask]`

