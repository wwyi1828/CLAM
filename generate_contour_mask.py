#!/usr/bin/env python3

import os
import argparse
from wsi_core.WholeSlideImage import WholeSlideImage

def generate_contour_mask(wsi_path, output_path, vis_level=-1, line_thickness=50, color=(0,255,0), hole_color=(0,0,255)):
    """
    Generate contour mask for a single WSI file

    Args:
        wsi_path: Path to the WSI file (.svs)
        output_path: Path to save the output mask (.png)
        vis_level: Visualization level (-1 for auto)
        line_thickness: Thickness of contour lines
        color: RGB tuple for tissue contours
        hole_color: RGB tuple for hole contours
    """
    print(f"Processing: {wsi_path}")

    # Initialize WSI
    WSI_object = WholeSlideImage(wsi_path)

    # Set segmentation parameters (exactly from create_patches_png.py bwh_biopsy preset)
    seg_params = {
        'seg_level': -1,
        'sthresh': 15,
        'mthresh': 11,
        'close': 2,
        'use_otsu': False,
        'keep_ids': [],
        'exclude_ids': []
    }

    filter_params = {
        'a_t': 1,
        'a_h': 1,
        'max_n_holes': 2
    }

    # Auto-determine levels if needed
    if vis_level < 0:
        if len(WSI_object.level_dim) == 1:
            vis_level = 0
        else:
            wsi = WSI_object.getOpenSlide()
            vis_level = wsi.get_best_level_for_downsample(64)

    if seg_params['seg_level'] < 0:
        if len(WSI_object.level_dim) == 1:
            seg_params['seg_level'] = 0
        else:
            wsi = WSI_object.getOpenSlide()
            best_level = wsi.get_best_level_for_downsample(64)
            seg_params['seg_level'] = best_level

    print(f"Using seg_level: {seg_params['seg_level']}, vis_level: {vis_level}")

    # Check if image is too large
    w, h = WSI_object.level_dim[seg_params['seg_level']]
    if w * h > 1e8:
        print(f'Warning: level_dim {w} x {h} is very large, processing may be slow')

    # Segment tissue (required for contour generation)
    print("Segmenting tissue...")
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    # Check if segmentation produced contours
    if not hasattr(WSI_object, 'contours_tissue') or len(WSI_object.contours_tissue) == 0:
        print("Warning: No tissue contours found after segmentation")
    else:
        print(f"Found {len(WSI_object.contours_tissue)} tissue contours")

    # Generate visualization with contours
    print("Generating contour mask...")
    vis_params = {
        'vis_level': vis_level,
        'line_thickness': line_thickness,
        'color': color,
        'hole_color': hole_color
    }
    contour_mask = WSI_object.visWSI(**vis_params)

    # Save the mask
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    contour_mask.save(output_path)
    print(f"Contour mask saved to: {output_path}")

    return contour_mask

def main():
    parser = argparse.ArgumentParser(description='Generate contour mask for WSI file')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input WSI file (.svs)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output mask file (.png)')
    parser.add_argument('--vis_level', type=int, default=-1,
                        help='Visualization level (-1 for auto)')
    parser.add_argument('--line_thickness', type=int, default=350,
                        help='Thickness of contour lines')
    parser.add_argument('--color', type=str, default='76,136,249',
                        help='Contour color as R,G,B (e.g., "255,0,0" for red)')
    parser.add_argument('--hole_color', type=str, default='15,157,88',
                        help='Hole contour color as R,G,B')

    args = parser.parse_args()

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return

    # Parse color
    color = tuple(map(int, args.color.split(',')))
    hole_color = tuple(map(int, args.hole_color.split(',')))

    # Generate contour mask
    generate_contour_mask(
        wsi_path=args.input,
        output_path=args.output,
        vis_level=args.vis_level,
        line_thickness=args.line_thickness,
        color=color,
        hole_color=hole_color
    )

if __name__ == '__main__':
    main()
