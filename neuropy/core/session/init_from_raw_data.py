import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import shutil
import sys
from typing import List, Dict, Tuple, Union, Set, Optional

## Neuropy Imports:
from neuropy.io import OptitrackIO
from neuropy.core import Position
from neuropy.io.binarysignalio import BinarysignalIO
# from neuropy.core.session.data_session_loader import DataSessionLoader


## UNUSED:
# from neuropy.io.neuroscopeio import NeuroscopeIO
# from neuropy.io.miniscopeio import MiniscopeIO
# from neuropy.io.spykingcircusio import SpykingCircusIO

# from neuropy.core.epoch import Epoch, EpochHelpers, ensure_dataframe, ensure_Epoch



def find_first_extant_column(df: pd.DataFrame, columns_list: List[str], raise_error_on_all_missing: bool=True) -> Optional[str]:
    """ finds the first extant column in the dataframe and returns it, optionally raising an error if all are missing """
    for a_col in columns_list:
        if a_col in df.columns:
            return a_col
    ## all failed
    if raise_error_on_all_missing:
        raise ValueError(f'no columns in the provided list: {columns_list} were found in the dataframe.\n\tdf.columns: {list(df.columns)}')
    else:
        return None



class RawDataInitializationMixin:
	""" helps initially setup a fresh session from the raw recording files. 
	Written 2026-05-25 by Pho Hale based off of a procedure I manually had to do to load a new Bapun recording session.
	
	
	Steps for initializing a brand-new, non-spike-sorted session:
		- [ ] Tracked Animal Positions - Optitrack Takes need to exported to using the proprietary Motiv software to .csv's.
		- [ ] An experiment, referred to as a "session", usually contains multiple near-contiguous recording folders, which exist due to disconnects (planned or unplanned), experimental setup, etc.
			- [ ] The first step in processing is currently to concatenate all of these files together so that they can be processed as if they were one. 
		- [ ] Recordings need to spike-sorted using Spyking Circus or some comparable software, which often requires manual curation (see detailed writeup). This produces spikes from individual units.


	# USAGE:
		from neuropy.core.session.init_from_raw_data import RawDataInitializationMixin
	
	"""

	@classmethod
	def build_session_datetime_csv(cls, search_dir: Path, basename: str='RatS-Day1Openfield', excluded_data_datetimes: Optional[List[str]]=None, n_channels: int = 195, sampling_rate: int = 30000, minimum_recording_duration_hours: Optional[float]=(5.0/60.0)):
		"""
		Recursively finds all 'settings.xml' files in search_dir and extracts their <DATE>.
		Then tries to find the correct continuous.dat (raw recording) file, loads its nFrames and duration in hours, and then exports the dateframe to CSV in the format required by 'RatS_Day1Openfield.datetime.csv'

		Usage:

			n_channels: int = 195
			sampling_rate: int = 30000
			basename: str = 'RatS-Day1Openfield'
			excluded_data_datetimes = ['2020-11-25_10-24-24']

			datetime_df, datetime_csv_out_path, found_raw_data_paths = RawDataInitializationMixin.build_session_datetime_csv(raw_data_path, basename=basename, excluded_data_datetimes=excluded_data_datetimes, n_channels=n_channels, sampling_rate=sampling_rate)
			datetime_df
			found_raw_data_paths


		"""
		import xml.etree.ElementTree as ET
		
		extracted_df = []
		found_raw_data_paths = []

		# rglob recursively searches the directory and all subdirectories
		for file_path in Path(search_dir).rglob('settings.xml'):
			try:
				# Parse the XML and concisely grab the text inside the <DATE> tag
				date_str = ET.parse(file_path).findtext('.//INFO/DATE')
				
				if date_str:
					# Convert "25 Nov 2020 10:20:27" to a Python datetime object
					parsed_date = datetime.strptime(date_str.strip(), "%d %b %Y %H:%M:%S")
					# extracted_dates[file_path] = parsed_date
					
					## Got the datetime, great start! Get the .dat file
					a_parent_recording_folder: Path = file_path.parent
					# a_binary_file_path: Path = a_parent_recording_folder.joinpath('experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat') # '/home/halechr/FastData/Bapun/RatS/Day4Openfield/Raw_data/2020-12-02_14-46-16/experiment1/recording1/continuous/Intan_Rec._Controller-100.0/continuous.dat'
					
					found_binary_file_paths: List[Path] = [file_path for file_path in a_parent_recording_folder.rglob('continuous.dat') if (file_path.exists() and file_path.is_file)]
					assert (len(found_binary_file_paths) > 0), f'ERROR: found no valid continuous.dat files in the subdirectory: "{a_parent_recording_folder.as_posix()}"!!'
					if len(found_binary_file_paths) > 1:
						print(f'WARNING: found multiple continuous.dat files in the subdirectory: "{a_parent_recording_folder.as_posix()}"\n\tfound_binary_file_paths: {found_binary_file_paths}\n\tusing the FIRST.')

					a_binary_file_path: Path = found_binary_file_paths[0]
					assert (a_binary_file_path.exists() and a_binary_file_path.is_file()), f"a_binary_file_path: {a_binary_file_path} does not exist!"

					found_raw_data_paths.append(a_binary_file_path)

					## Use `BinarysignalIO` to extract the signal duration information from the raw files
					a_raw_file_obj: BinarysignalIO = BinarysignalIO(a_binary_file_path, dtype="int16", n_channels=n_channels, sampling_rate=sampling_rate)
					
					n_frames: int = a_raw_file_obj.n_frames
					duration_hours: float = float(a_raw_file_obj.duration) / 3600.0
					# self.duration/3600
					extracted_df.append((parsed_date, n_frames, duration_hours))

			except (ET.ParseError, ValueError) as e:
				# Skips files with malformed XML or unexpected date formats
				print(f"Skipping {file_path} due to error: {e}")

		columns = ["startTime", "nFrames", "duration (h)"]
		extracted_df = pd.DataFrame.from_records(extracted_df, columns=columns) 

		# Convert just the 'startTime' column to the desired string format
		extracted_df['startTime'] = extracted_df['startTime'].dt.strftime('%Y-%m-%d_%H-%M-%S')
		extracted_df['is_included'] = True ## all true by default
		is_included = extracted_df['is_included'].to_numpy()
		
		if excluded_data_datetimes is not None:
			is_included_considering_exclusions = np.logical_not(extracted_df['startTime'].isin(excluded_data_datetimes))
			is_included = np.logical_and(is_included, is_included_considering_exclusions)
			
		if minimum_recording_duration_hours is not None:
			is_included_considering_duration = (extracted_df['duration (h)'] >= minimum_recording_duration_hours)
			is_included = np.logical_and(is_included, is_included_considering_duration)
			duration_excluded_indicies = np.where(np.logical_not(is_included_considering_duration))[0]
			if len(duration_excluded_indicies) > 0:
				print(f'excluded duration_excluded_indicies: {duration_excluded_indicies} because they were shorter than minimum_recording_duration_hours: {minimum_recording_duration_hours}')
			
		## INPUTS: is_included
		extracted_df['is_included'] = is_included
		included_indicies = np.where(is_included)[0]

		## filter out the excluded ones
		extracted_df = extracted_df[extracted_df['is_included']].reset_index(drop=True)
		found_raw_data_paths = [v for i, v in enumerate(found_raw_data_paths) if i in included_indicies]
			
		csv_output_path: Path = search_dir.parent
		csv_out_path: Path = csv_output_path.joinpath(f'{basename}.datetime.csv')
		print(f'output csv path: "{csv_out_path.as_posix()}"')
		extracted_df.to_csv(csv_out_path)

		return extracted_df, csv_out_path, found_raw_data_paths


	@classmethod
	def build_initial_position_data(cls, sess, position_csvs_path: Path):
		""" builds the initial position data, creating the output '*.position.npy' file.
		Modifies:
			sess.position
			
		Usage:
		
			position_csvs_path: Path = sess.basepath.joinpath('Raw_data/position/CSVs')
			position: Position = RawDataInitializationMixin.build_initial_position_data(sess, position_csvs_path=position_csvs_path)
			position

		"""
		# opti_folder = Path('W:/Data/Bapun/RatS/Day1Openfield/Raw_data/position/CSVs')
		if position_csvs_path is None:
			position_csvs_path = sess.basepath.joinpath('Raw_data/position/CSVs')
		
		assert position_csvs_path.exists(), f"position_csvs_path: '{position_csvs_path.as_posix()}' does not exist!"
		opti_data: OptitrackIO = OptitrackIO(dirname=position_csvs_path)

		#---- startimes of concatenated .dat files
		tracking_sRate = opti_data.sampling_rate
		rec_datetime: pd.DataFrame = pd.read_csv(sess.filePrefix.with_suffix('.datetime.csv'))
		data_time = []
		start_col_name: str = find_first_extant_column(rec_datetime, columns_list=['StartTime', 'startTime'])

		for i, file_time in enumerate(rec_datetime[start_col_name]):
			tbegin = datetime.strptime(file_time, "%Y-%m-%d_%H-%M-%S")
			nframes = rec_datetime["nFrames"][i]
			duration = pd.Timedelta(nframes / sess.recinfo.dat_sampling_rate, unit="sec")
			tend = tbegin + duration
			trange = pd.date_range(
				start=tbegin,
				end=tend,
				periods=int(duration.total_seconds() * tracking_sRate),
			)
			data_time.extend(trange)
		data_time = pd.to_datetime(data_time)

		# ------- deleting intervals that were deleted from .dat file after concatenating
		if ("deletedStart (minutes)" in rec_datetime.columns) and ("deletedEnd (minutes)" in rec_datetime.columns):
			ndeletedintervals = rec_datetime.count()["deletedStart (minutes)"]
			for i in range(ndeletedintervals):
				tnoisy_begin = data_time[0] + pd.Timedelta(
					rec_datetime["deletedStart (minutes)"][i], unit="m"
				)
				tnoisy_end = data_time[0] + pd.Timedelta(
					rec_datetime["deletedEnd (minutes)"][i], unit="m"
				)

				del_index = np.where((data_time > tnoisy_begin) & (data_time < tnoisy_end))[
					0
				]

				data_time = np.delete(data_time, del_index)


		## resume main
		x,y,z = opti_data.get_position_at_datetimes(data_time)
		traces = np.vstack((z,x,y))

		# rx, ry, rz, rw = opti_data.get_rotations_at_datetimes(data_time)
		# traces_rot = np.vstack((rx, ry, rz, rw)) ## won't work because it only accepts 3-column rotation at most
		traces_rot = None
		position: Position = Position.init(traces=traces, t_start=0, sampling_rate=opti_data.sampling_rate, traces_rot=traces_rot)

		## use this update method instead to include rotations data
		position = opti_data.updating_position_obj(position)
		# pos_df = position.to_dataframe()

		## Save out to file:
		position.filename = sess.filePrefix.with_suffix('.position.npy')
		print(f'saving out to {position.filename.as_posix()}...')
		position.save()
		print(f'\tdone.')
		
		## update session's position
		sess.position = position

		return position
	

	@classmethod
	def concatenate_to_output_file(cls, input_paths: List[Path], output_path: Path):
		""" concatenates the binary files in input_paths to a single joined output_path. 
		"""
		with open(output_path, "wb") as outfile:
			for path in input_paths:
				with open(path, "rb") as infile:
					shutil.copyfileobj(infile, outfile)
				

	@classmethod
	def step_perform_concat(cls, found_raw_data_paths: List[Path], spyk_circ_output_dir: Path, basename: str='continuous_combined'):
		""" performs the concatenation, creates the output directory if needed
		

		Usage:		
			## Copy the concatenated files to the output directory
			concatenated_file_output_path: Path = RawDataInitializationMixin.step_perform_concat(found_raw_data_paths=found_raw_data_paths, spyk_circ_output_dir=spyk_circ_output_dir)
			print(f'have concatenated_file_output_path: "{concatenated_file_output_path.as_posix()}"')
			concatenated_file_output_path

		"""
		## make the "spyk-circ" output directory
		# spyk_circ_output_dir: Path = Path('W:/Data/Bapun/RatS/Day1Openfield/spyk-circ').resolve()
		spyk_circ_output_dir.mkdir(exist_ok=True, parents=True) ## dang I sure hope we're on Windows or I'll add some garbage paths :P
		
		## Copy the concatenated files to the output directory
		concatenated_file_output_path: Path = spyk_circ_output_dir.joinpath(f'{basename}.dat').resolve() ## do I need to do anything with the adjacent `timestamps.npy` or anything??
		if not concatenated_file_output_path.exists():
			cls.concatenate_to_output_file(input_paths=found_raw_data_paths, output_path=concatenated_file_output_path)
		return concatenated_file_output_path


	@classmethod
	def build_process_resample_command(cls, sess, n_channels: int = 195, dat_sampling_rate_Hz: int = 30000, desired_eeg_sampling_rate_Hz: int = 1250):
		""" builds the terminal command to run to downsample the .dat file to an eeg file

		Usage:
			output_process_resample_cmd: Optional[str] = RawDataInitializationMixin.build_process_resample_command(sess, n_channels=n_channels, dat_sampling_rate_Hz=sampling_rate, desired_eeg_sampling_rate_Hz=1250)
			output_process_resample_cmd
		"""
		datfile_path: Path = sess.filePrefix.with_suffix('.dat')
		eegfile_path: Path = sess.filePrefix.with_suffix('.eeg')

		if not eegfile_path.exists():
			assert datfile_path.exists(), f"EEG file does not exist but neither does the expected .dat file. (expected datfile_path: '{datfile_path.as_posix()}'"
			output_process_resample_cmd: str = f"process_resample -f {dat_sampling_rate_Hz},{desired_eeg_sampling_rate_Hz} -n {n_channels} '{datfile_path.as_posix()}' '{eegfile_path.as_posix()}'"
			print(f'Run the output_process_resample_cmd:\n{output_process_resample_cmd}\n\n to produce the downsampled .eeg file.')
			return output_process_resample_cmd
		else:
			return None ## not needed

	@classmethod
	def step_perform_downsample(cls, concatenated_file_output_path: Path, sampling_frequency=30000, num_chan=195, resample_rate=1250) -> Path:
		""" not working (even with proper libraries """
		raise NotImplementedError(f'Not finished even with proper dependencies.')
		# import spikeinterface.extractors as se
		# import spikeinterface.preprocessing as spre
		# import spikeinterface.full as si
		# from spikeinterface.core import write_binary_recording

		# # 1. Map the raw binary file (doesn't load into RAM yet)
		# recording = se.read_binary(
		# 	file_paths=concatenated_file_output_path.as_posix(), 
		# 	sampling_frequency=sampling_frequency, 
		# 	num_chan=num_chan, 
		# 	dtype='int16'
		# )

		# # 2. Set up the downsampling node (applies anti-aliasing filter automatically)
		# recording_lfp = spre.resample(recording, resample_rate=resample_rate)

		# output_lfp_path: Path = concatenated_file_output_path.with_suffix('.lfp').resolve()
		# print(f'trying to write to: "{output_lfp_path.as_posix()}"')
		# # 3. Execute and write to a flat binary file in chunks
		# # n_jobs=-1 uses all CPU cores for faster processing
		# write_binary_recording(
		# 	recording_lfp, 
		# 	file_paths=output_lfp_path.as_posix(), 
		# 	dtype='int16', 
		# 	n_jobs=-1, 
		# 	chunk_duration='1s'
		# )
		# print(f'\tdone.')
		# return output_lfp_path



	@classmethod
	def run_all(cls, basedir: Path,
			 basename: str = 'RatS-Day1Openfield', n_channels: int = 195, dat_file_sampling_rate: int = 30000, excluded_data_datetimes: List[str]=None,
		):
		""" runs all needed steps 

		## Bapun Format:
		# basedir = '/media/share/data/Bapun/Day5TwoNovel' # Linux
		basedir: Path = Path('W:/Data/Bapun/RatS/Day1Openfield') # Windows
		# basedir = '/Volumes/iNeo/Data/Bapun/Day5TwoNovel' # MacOS
				
		n_channels: int = 195
		dat_file_sampling_rate: int = 30000
		basename: str = 'RatS-Day1Openfield'
		excluded_data_datetimes = ['2020-11-25_10-24-24']

		
		sess = RawDataInitializationMixin.run_all(basedir=basedir,
			basename=basename, excluded_data_datetimes=excluded_data_datetimes, n_channels=n_channels, sampling_rate=dat_file_sampling_rate,
		)
		
		
		"""
		from neuropy.core.session.data_session_loader import DataSessionLoader

		## INPUTS: basedir
		sess = DataSessionLoader.bapun_data_session(basedir, enable_continue_on_required_path_failure=True)
		active_sess_config = sess.config
	
		if basename is None:
			basename = sess.name

		print(f'basename: {basename}')

		recinfo_dict = sess.recinfo.to_dict()

		if n_channels is None:
			n_channels = recinfo_dict.get('n_channels', n_channels)

		if dat_file_sampling_rate is None:
			dat_file_sampling_rate = recinfo_dict.get('dat_sampling_rate', dat_file_sampling_rate)

		eeg_sampling_rate: int = recinfo_dict.get('eeg_sampling_rate', 1250)

		print(f'n_channels: {n_channels}')
		print(f'dat_file_sampling_rate: {dat_file_sampling_rate}')
		print(f'eeg_sampling_rate: {eeg_sampling_rate}')

		raw_data_path: Path = basedir.joinpath('Raw_data')
		assert raw_data_path.exists(), f"raw_data_path: '{raw_data_path.as_posix()}' does not exist!"
		print(f'raw_data_path: "{raw_data_path.as_posix()}"')

		## make the "spyk-circ" output directory
		spyk_circ_output_dir: Path = basedir.joinpath('spyk-circ')
		spyk_circ_output_dir.mkdir(exist_ok=True, parents=True) ## dang I sure hope we're on Windows or I'll add some garbage paths :P

		# ## INPUTS: raw_data_path: Path # Path(r'W:/Data/Bapun/RatS/Day1Openfield/Raw_data').resolve()
		# print(f'raw_data_path: "{raw_data_path.as_posix()}"')

		# ## find all constitutent "continuous.dat" files recurrsively in all subdirectories: "W:\Data\Bapun\RatS\Day1Openfield\Raw_data\2020-11-25_10-24-24\experiment1\recording1\continuous\Rhythm_FPGA-100.0\continuous.dat"
		# found_raw_data_paths = ["W:/Data/Bapun/RatS/Day1Openfield/Raw_data/2020-11-25_10-20-27/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat",
		#                         # "W:/Data/Bapun/RatS/Day1Openfield/Raw_data/2020-11-25_10-24-24/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat", ## BAD ONE, only has 32 channels, skip
		#                         "W:/Data/Bapun/RatS/Day1Openfield/Raw_data/2020-11-25_13-02-47/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat",
		#                         "W:/Data/Bapun/RatS/Day1Openfield/Raw_data/2020-11-25_14-30-32/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat",
		#                         "W:/Data/Bapun/RatS/Day1Openfield/Raw_data/2020-11-25_15-06-02/experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat",
		# ]    
		# ## *-24-24 is a bad one with only 30 good channels!

		# ## Iterate through and make proper paths, check their existance
		# found_raw_data_paths: List[Path] = [Path(v).resolve() for v in found_raw_data_paths]
		## could assert that they all exist... but let's NOT!
		# excluded_data_datetimes = ['2020-11-25_10-24-24']
		

		datetime_df, datetime_csv_out_path, found_raw_data_paths = cls.build_session_datetime_csv(raw_data_path, basename=basename, excluded_data_datetimes=excluded_data_datetimes,
																												    	n_channels=n_channels, sampling_rate=dat_file_sampling_rate,
																														)
		datetime_df
		found_raw_data_paths
		


		## Copy the concatenated files to the output directory
		concatenated_file_output_path: Path = cls.step_perform_concat(found_raw_data_paths=found_raw_data_paths, spyk_circ_output_dir=spyk_circ_output_dir, basename=basename)
		print(f'have concatenated_file_output_path: "{concatenated_file_output_path.as_posix()}"')
		concatenated_file_output_path


		# sess.eegfile ## has an EEG file object
		output_process_resample_cmd: Optional[str] = cls.build_process_resample_command(sess, n_channels=n_channels, dat_sampling_rate_Hz=dat_file_sampling_rate, desired_eeg_sampling_rate_Hz=eeg_sampling_rate)
		output_process_resample_cmd

		## POSITIONS:
		from neuropy.core import Position

		position_csvs_path: Path = sess.basepath.joinpath('Raw_data/position/CSVs')
		position: Position = RawDataInitializationMixin.build_initial_position_data(sess, position_csvs_path=position_csvs_path)
		position

		return sess