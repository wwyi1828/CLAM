# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from utils.slide_magnification import (
    build_magnification_record,
    choose_patch_sampling_plan,
    format_patch_plan,
    resolve_magnification_strategy,
)

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start

	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed


def apply_record(df, idx, values):
	for key, value in values.items():
		df.loc[idx, key] = np.nan if value is None else value

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
				  patch_size = 224, step_size = 224, custom_downsample=1,
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8},
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  magnification_strategy = None,
				  target_magnification = None,
				  target_mpp = None,
				  unknown_magnification_policy = 'skip',
				  process_list_output = 'process_list_autogen.csv',
				  use_default_params = False,
				  seg = False, save_mask = True,
				  stitch= False,
				  patch = False, auto_skip=True, process_list = None):

	magnification_strategy = resolve_magnification_strategy(
		strategy=magnification_strategy,
		target_magnification=target_magnification,
		target_mpp=target_mpp,
	)


	slides = sorted(os.listdir(source))
	slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.
	process_list_output_path = os.path.join(save_dir, process_list_output)

	for i in range(total):
		df.to_csv(process_list_output_path, index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))

		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		full_path = os.path.join(source, slide)
		WSI_object = WholeSlideImage(full_path)
		selected_patch_plan = None

		if patch:
			selected_patch_plan, magnification_info, planning_status = choose_patch_sampling_plan(
				slide=WSI_object.getOpenSlide(),
				level_downsamples=WSI_object.level_downsamples,
				patch_size=patch_size,
				step_size=step_size,
				magnification_strategy=magnification_strategy,
				target_magnification=target_magnification,
				target_mpp=target_mpp,
				patch_level=patch_level,
				custom_downsample=custom_downsample,
				unknown_magnification_policy=unknown_magnification_policy,
			)
			skip_note = None
			if planning_status == 'skip' or selected_patch_plan is None:
				skip_note = (
					f'no usable magnification metadata found for {magnification_strategy} '
					'strategy; skipping slide'
				)
			apply_record(df, idx, build_magnification_record(
				strategy=magnification_strategy,
				info=magnification_info,
				plan=selected_patch_plan,
				target_magnification=target_magnification,
				target_mpp=target_mpp,
				note_override=skip_note,
			))

			if skip_note is not None or selected_patch_plan is None:
				print(skip_note)
				df.loc[idx, 'status'] = 'skipped_unknown_magnification'
				continue

			if magnification_info.warning:
				print(f'magnification warning: {magnification_info.warning}')
			print(f'magnification plan: {format_patch_plan(selected_patch_plan)}')
			print(f'magnification note: {selected_patch_plan.note}')
		else:
			df.loc[idx, 'magnification_strategy'] = magnification_strategy

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()

		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0

			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0

			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']]
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.png')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level if selected_patch_plan is None else selected_patch_plan.patch_level,
										 'patch_size': patch_size, 'step_size': step_size,
										 'save_path': patch_save_dir,
										 'custom_downsample': custom_downsample if selected_patch_plan is None else selected_patch_plan.custom_downsample,
										 'rescale_factor': selected_patch_plan.rescale_factor if selected_patch_plan is not None else None})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params)

		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.png')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(process_list_output_path, index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))

	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=224,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=224,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0,
					help='manual mode only: downsample level at which to patch')
parser.add_argument('--custom_downsample', type=int, default=1,
					help='manual mode only: additional read-time resize factor')
parser.add_argument('--magnification_strategy', type=str, choices=['manual', 'bin', 'rescale'], default=None,
					help='patch magnification strategy; defaults to inferring bin/rescale from target args, otherwise manual')
parser.add_argument('--target_magnification', type=float, choices=[20.0, 40.0], default=None,
					help='bin mode target magnification')
parser.add_argument('--target_mpp', type=float, default=None,
					help='rescale mode target microns-per-pixel')
parser.add_argument('--unknown_magnification_policy', type=str, choices=['skip', 'error', 'assume-target'], default='skip',
					help='how to handle slides without usable magnification metadata in bin/rescale mode')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')
parser.add_argument('--process_list_output', type=str, default='process_list_autogen.csv',
					help='output csv path relative to save_dir for recording processing status')

if __name__ == '__main__':
	args = parser.parse_args()

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	print('source: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)

	directories = {'source': args.source,
				   'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir,
				   'mask_save_dir' : mask_save_dir,
				   'stitch_save_dir': stitch_save_dir}

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	#preset: bwh_biopsy
	seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':1, 'a_h': 1, 'max_n_holes':2 }
	vis_params = {'vis_level': -1, 'line_thickness': 50}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]

	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=args.step_size,
											seg = args.seg,  use_default_params=False, save_mask = True,
											stitch= args.stitch, custom_downsample=args.custom_downsample,
											patch_level=args.patch_level, patch = args.patch,
											magnification_strategy=args.magnification_strategy,
											target_magnification=args.target_magnification,
											target_mpp=args.target_mpp,
											unknown_magnification_policy=args.unknown_magnification_policy,
											process_list_output=args.process_list_output,
											process_list = process_list, auto_skip=args.no_auto_skip)
