# CLAM 224-Grid Coordinate Extension

This branch is based on CLAM commit `9482cbc`. The main patching change is that
each contour's bounding box is extended back onto the global 224-step grid before
candidate patch coordinates are generated.

## Core Change

Original CLAM started the coordinate grid at each contour's own bounding box:

```python
start_x, start_y, w, h = cv2.boundingRect(cont)

step_size_x = step_size * patch_downsample[0]
step_size_y = step_size * patch_downsample[1]

x_range = np.arange(start_x, stop_x, step=step_size_x)
y_range = np.arange(start_y, stop_y, step=step_size_y)
```

This branch expands the contour box to the nearest global grid origin first:

```python
step_size_x = read_step_size * patch_downsample[0]
step_size_y = read_step_size * patch_downsample[1]

start_x, start_y, w, h = cv2.boundingRect(cont)

w += start_x % step_size_x
h += start_y % step_size_y
start_x -= start_x % step_size_x
start_y -= start_y % step_size_y
```

With the default `patch_size=224` and `step_size=224`, this makes patch
coordinates fall on the same 224-grid instead of using a different local grid
for each contour.

The coordinate writer then stores level-0 coordinates:

```python
coords = []

for patch in patch_gen:
    if patch.get('coord_level') == 0:
        coords.append([patch['x'], patch['y']])
    else:
        patch_downsample = (
            int(self.level_downsamples[patch_level][0]),
            int(self.level_downsamples[patch_level][1]),
        )
        x_level_0 = patch['x'] * patch_downsample[0] * custom_downsample
        y_level_0 = patch['y'] * patch_downsample[1] * custom_downsample
        coords.append([x_level_0, y_level_0])

asset_dict = {'coords': np.array(coords, dtype=np.int32)}
```

## Usage

```bash
python create_patches_fp.py \
  --source DATA_DIRECTORY \
  --save_dir RESULTS_DIRECTORY \
  --patch_size 224 \
  --step_size 224 \
  --seg \
  --patch
```
