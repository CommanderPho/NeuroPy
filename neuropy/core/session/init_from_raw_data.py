import pandas as pd
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
import os
import numpy as np
import shutil
import subprocess
import sys
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Union, Set, Optional, Callable, Literal

_PHY_REQUIRED_FILES = ("params.py", "spike_times.npy", "spike_clusters.npy", "cluster_info.tsv")
NeuronSourceType = Literal["auto", "spyk_circ", "sorting"]
NeuronSourceTypeResolved = Literal["spyk_circ", "sorting"]
PhyFolderKind = Literal["invalid", "si_phy_export", "spyk_circ_phy_export"]


@dataclass
class NeuronLoadConfig:
    source_type: NeuronSourceType = "auto"
    phy_folder: Optional[Path] = None
    curation_review_path: Optional[Path] = None
    run_name: Optional[str] = None
    include_groups: tuple[str, ...] = ("good", "mua")
    unit_filter: str = 'prediction == "sua"'
    save_neurons: bool = True
    estimate_neuron_type: bool = True

## Neuropy Imports:
from neuropy.io import OptitrackIO
from neuropy.core.epoch import Epoch, EpochHelpers, ensure_dataframe, ensure_Epoch
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

def windows_to_wsl_path_if_needed(path: Union[str, Path]) -> Path:
    """
    Converts a Windows path to a WSL-compatible path only when useful:
    - If the original path exists, returns it unchanged.
    - If running inside WSL and the converted path exists, returns the WSL path.
    - Otherwise returns the original path unchanged.

    Usage:

    from neuropy.core.session.init_from_raw_data import RawDataInitializationMixin, windows_to_wsl_path_if_needed

    windows_to_wsl_path_if_needed(path=path)

    """
    original = Path(path)

    # Original path already works.
    if original.exists():
        return original

    # Only attempt conversion when running under WSL.
    if "WSL_DISTRO_NAME" not in os.environ:
        return original

    s = str(path).strip().strip('"')

    # Already POSIX-like.
    if s.startswith("/") or s.startswith("~"):
        return original

    p = PureWindowsPath(s)

    # Skip UNC/network paths.
    if p.drive.startswith("\\\\"):
        return original

    # Convert drive-letter path.
    if p.drive:
        tail = PurePosixPath(*p.parts[1:]).as_posix()
        converted = Path(
            f"/mnt/{p.drive[:-1].lower()}" + (f"/{tail}" if tail else "")
        )

        if converted.exists():
            return converted

    return original


def subfn_copy_if_needed(ref_file: Path, target_file: Path, force_overwrite: bool = False) -> bool:
    """ copies the file only if the target_file does not exist, unless force_overwrite is True """
    ref_file, target_file = Path(ref_file), Path(target_file)
    if target_file.exists() and not force_overwrite:
        print(f'target_file: "{target_file.as_posix()}" already exists.')
        return False
    assert ref_file.exists(), f'ref_file: {ref_file} does not exist (and it is needed because target_file: "{target_file.as_posix()}" does not exist either)!'
    target_file.parent.mkdir(exist_ok=True, parents=True)
    print(f'copying: "{ref_file.as_posix()}" -> "{target_file.as_posix()}"...')
    shutil.copy2(ref_file, target_file)
    print(f'\tdone.')
    return True


def _subfn_add_epoch_buffer(epoch_df: pd.DataFrame, buffer_sec: float or int or tuple or list):
    """Extend each epoch by buffer_sec before/after start/stop of each epoch"""
    if type(buffer_sec) in [int, float]:
        buffer_sec = (buffer_sec, buffer_sec)
    else:
        assert len(buffer_sec) == 2

    epoch_df['start'] -= buffer_sec[0]
    epoch_df['stop'] += buffer_sec[1]


def find_first_file_rglob(search_root: Union[str, Path], filename: str, warn_on_multiple: bool = True, error_context_label: Optional[str] = None, recursive: bool = True, raise_on_none_found: bool = True) -> Optional[Path]:
    """Find the first existing file matching ``filename`` under ``search_root``.

    When ``recursive`` is True (default), searches all subdirectories via ``rglob``.
    When False, searches only the immediate directory via ``glob``.

    When no match is found, raises ``FileNotFoundError`` if ``raise_on_none_found`` is True (default),
    otherwise returns ``None``. When multiple matches exist, prints a warning and returns the first (glob order).
    """
    search_root = Path(search_root)
    context = error_context_label or search_root.as_posix()
    globber = search_root.rglob if recursive else search_root.glob
    found_files: List[Path] = [p for p in globber(filename) if p.is_file()]
    if len(found_files) == 0:
        if not raise_on_none_found:
            return None
        scope = 'subdirectory' if recursive else 'directory'
        raise FileNotFoundError(f'ERROR: found no valid {filename!r} files in the {scope}: "{context}"!!')
    if len(found_files) > 1 and warn_on_multiple:
        scope = 'subdirectory' if recursive else 'directory'
        print(f'WARNING: found multiple {filename!r} files in the {scope}: "{context}"\n\tfound_files: {found_files}\n\tusing the FIRST.')
    return found_files[0]



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
    def step_copy_dat_to_spyk_circ_output_dir(cls, sess, spyk_circ_dir: Path, prompt_on_replace: bool=True) -> Path:
        """ copies the session .dat from basedir into spyk_circ_dir if missing or if the user confirms replace when sizes differ """
        source_dat_path: Path = windows_to_wsl_path_if_needed(sess.filePrefix.with_suffix('.dat'))
        assert source_dat_path.exists(), f"Source .dat file does not exist: '{source_dat_path.as_posix()}'"
        spyk_circ_dir.mkdir(exist_ok=True, parents=True)
        dest_dat_path: Path = spyk_circ_dir.joinpath(source_dat_path.name).resolve()
        source_size: int = source_dat_path.stat().st_size
        if dest_dat_path.exists():
            dest_size: int = dest_dat_path.stat().st_size
            if dest_size == source_size:
                print(f'spyk-circ .dat already exists with matching size ({source_size} bytes): "{dest_dat_path.as_posix()}"')
                return dest_dat_path
            print(f'spyk-circ .dat exists but size differs (dest: {dest_size}, source: {source_size} bytes):\n\t"{dest_dat_path.as_posix()}"')
            if prompt_on_replace:
                replace_response: str = input(f'Replace destination .dat with source? [y/N]: ').strip().lower()
                if replace_response not in ('y', 'yes'):
                    print(f'keeping existing file: "{dest_dat_path.as_posix()}"')
                    return dest_dat_path
            else:
                print(f'skipping copy (prompt_on_replace=False)')
                return dest_dat_path
        print(f'copying .dat to spyk-circ: "{source_dat_path.as_posix()}" -> "{dest_dat_path.as_posix()}"...')
        shutil.copy2(source_dat_path, dest_dat_path)
        print(f'\tdone.')
        return dest_dat_path


    @classmethod
    def _find_7z_executable(cls) -> str:
        for candidate_name in ('7z', '7za', '7zr'):
            found_executable: Optional[str] = shutil.which(candidate_name)
            if found_executable is not None:
                return found_executable
        for windows_path in (Path(r'C:\Program Files\7-Zip\7z.exe'), Path(r'C:\Program Files (x86)\7-Zip\7z.exe')):
            if windows_path.is_file():
                return str(windows_path)
        raise FileNotFoundError("Could not find 7z executable. Install 7-Zip (Windows) or p7zip-full (Linux), or ensure '7z' is on PATH.")


    @classmethod
    def archive_directory_to_7z(cls, source_dir: Path, archive_path: Path) -> Path:
        """ archives source_dir into a .7z file at archive_path """
        source_dir = Path(source_dir).resolve()
        archive_path = Path(archive_path).resolve()
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Cannot archive missing directory: '{source_dir.as_posix()}'")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        seven_z_exe: str = cls._find_7z_executable()
        print(f'archiving "{source_dir.as_posix()}" to "{archive_path.as_posix()}"...')
        completed = subprocess.run([seven_z_exe, 'a', '-t7z', str(archive_path), str(source_dir)], capture_output=True, text=True)
        if completed.returncode != 0:
            error_output: str = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(f"7z archive failed (exit {completed.returncode}): {error_output}")
        if not archive_path.is_file() or archive_path.stat().st_size <= 0:
            raise RuntimeError(f"7z archive was not created or is empty: '{archive_path.as_posix()}'")
        print(f'\tdone.')
        return archive_path


    @classmethod
    def delete_previous_spyk_circ_run(cls, sess, archive_previous: bool=True):
        """ deletes a previous/incorrect spyking circus run directory, backing it up to a .7z archive first if desired"""
        ## make the "spyk-circ" output directory
        spyk_circ_dir: Path = sess.basepath.joinpath('spyk-circ')
        dest_dat_path: Path = cls.step_copy_dat_to_spyk_circ_output_dir(sess, spyk_circ_dir=spyk_circ_dir)
        dest_dat_path

        ## backup the previous extant spyk_circ outputs, which are located at 
        spyk_circ_outputs: Path = spyk_circ_dir.joinpath(sess.name)
        if not spyk_circ_outputs.exists():
            print(f'no previous spyk-circ outputs found at: "{spyk_circ_outputs.as_posix()}"')
            return
        if archive_previous:
            archive_path: Path = spyk_circ_dir.joinpath(f'{sess.name}_spyk-circ_{datetime.now().strftime("%Y%m%d_%H%M%S")}.7z')
            cls.archive_directory_to_7z(source_dir=spyk_circ_outputs, archive_path=archive_path)
        else:
            delete_response: str = input(f'Delete spyk-circ outputs without backup at "{spyk_circ_outputs.as_posix()}"? [y/N]: ').strip().lower()
            if delete_response not in ('y', 'yes'):
                print(f'keeping existing spyk-circ outputs: "{spyk_circ_outputs.as_posix()}"')
                return
        print(f'deleting spyk-circ outputs: "{spyk_circ_outputs.as_posix()}"...')
        shutil.rmtree(spyk_circ_outputs)
        print(f'\tdone.')



    @classmethod
    def build_session_datetime_csv(cls, search_dir: Path, basename: str='RatS-Day1Openfield', excluded_data_datetimes: Optional[List[str]]=None, minimum_recording_duration_hours: Optional[float]=(5.0/60.0),
                                    n_channels: int = 195, sampling_rate: int = 30000, debug_print: bool = True):
        """
        Recursively finds all 'settings.xml' files in search_dir and extracts their <DATE>.
        Then tries to find the correct continuous.dat (raw recording) file, loads its nFrames and duration in hours, and then exports the dateframe to CSV in the format required by 'RatS_Day1Openfield.datetime.csv'

        Usage:

            n_channels: int = 195
            sampling_rate: int = 30000
            basename: str = 'RatS-Day1Openfield'
            excluded_data_datetimes = ['2020-11-25_10-24-24']

            included_only_datetime_df, datetime_csv_out_path, found_raw_data_paths, all_datetime_df, all_found_files_dict = RawDataInitializationMixin.build_session_datetime_csv(raw_data_path, basename=basename, excluded_data_datetimes=excluded_data_datetimes, n_channels=n_channels, sampling_rate=sampling_rate)
            datetime_df
            found_raw_data_paths


        """
        import xml.etree.ElementTree as ET
        
        csv_output_path: Path = search_dir.parent
        csv_out_path: Path = csv_output_path.joinpath(f'{basename}.datetime.csv')

        all_datetime_df = []
        all_found_files_dict = {'settings_xml': [], 'continuous_xml': [], 'dat': [], 'recording_folder': []}
        found_raw_data_paths = []

        # rglob recursively searches the directory and all subdirectories
        for settings_xml_file_path in Path(search_dir).rglob('settings.xml'):
            try:
                # Parse the XML and concisely grab the text inside the <DATE> tag
                date_str = ET.parse(settings_xml_file_path).findtext('.//INFO/DATE')
                
                if date_str:
                    # Convert "25 Nov 2020 10:20:27" to a Python datetime object
                    parsed_date = datetime.strptime(date_str.strip(), "%d %b %Y %H:%M:%S")
                    # extracted_dates[file_path] = parsed_date
                    
                    ## Got the datetime, great start! Get the .dat file
                    a_parent_recording_folder: Path = settings_xml_file_path.parent
                    # a_binary_file_path: Path = a_parent_recording_folder.joinpath('experiment1/recording1/continuous/Rhythm_FPGA-100.0/continuous.dat') # '/home/halechr/FastData/Bapun/RatS/Day4Openfield/Raw_data/2020-12-02_14-46-16/experiment1/recording1/continuous/Intan_Rec._Controller-100.0/continuous.dat'
                    all_found_files_dict['recording_folder'].append(a_parent_recording_folder)

                    ## try to parse the number of channels from the continuous.xml file:
                    a_continuous_xml: Path = find_first_file_rglob(a_parent_recording_folder, 'continuous.xml')
                    parsed_n_channels: int = int(ET.parse(a_continuous_xml).findtext('.//acquisitionSystem/nChannels'))

                    ## try to parse the number of channels from the settings.xml file:
                    # all_channel_names = ET.parse(settings_xml_file_path).findall('.//PROCESSOR[@pluginName="Rhythm FPGA"]/CHANNEL')
                    # if len(all_channel_names) > 0:
                    #     parsed_n_channels: int = int(all_channel_names[-1].attrib['number'])
                    #     if debug_print:                            print(f'settings.xml: "{settings_xml_file_path.as_posix()}" -- last_channel_number: {parsed_n_channels}')
                    # else:
                    #     raise ValueError(f'could not extract the last channel number from the settings.xml: "{settings_xml_file_path.as_posix()}"')
                    # # -> '200'
                    

                    if (parsed_n_channels != n_channels):
                        raise ValueError(f'number of channels parsed from one of the .xml files is {parsed_n_channels}, differing from the specified n_channels: {n_channels}.') 


                    a_binary_file_path: Path = find_first_file_rglob(a_parent_recording_folder, 'continuous.dat')

                    found_raw_data_paths.append(a_binary_file_path)

                    ## Use `BinarysignalIO` to extract the signal duration information from the raw files
                    a_raw_file_obj: BinarysignalIO = BinarysignalIO(a_binary_file_path, dtype="int16", n_channels=n_channels, sampling_rate=sampling_rate)
                    
                    dat_file_size_bytes: int = a_binary_file_path.stat().st_size

                    ## Add collected files:
                    all_found_files_dict['settings_xml'].append(settings_xml_file_path)
                    all_found_files_dict['continuous_xml'].append(a_continuous_xml)
                    all_found_files_dict['dat'].append(a_binary_file_path)
                    all_found_files_dict['recording_folder'].append(a_parent_recording_folder)
                    
                    n_frames: int = a_raw_file_obj.n_frames
                    duration_hours: float = float(a_raw_file_obj.duration) / 3600.0
                    # self.duration/3600
                    
                    all_datetime_df.append((parsed_date, n_frames, duration_hours, parsed_n_channels, dat_file_size_bytes, a_parent_recording_folder.as_posix(), a_binary_file_path.as_posix(), a_continuous_xml.as_posix()))


            except (ET.ParseError, ValueError) as e:
                # Skips files with malformed XML or unexpected date formats
                print(f"Skipping {settings_xml_file_path} due to error: {e}")

        columns = ["startTime", "nFrames", "duration (h)", "n_channels", "dat_file_size_bytes", "recording_folder", "dat_file", "continuous_xml"]
        all_datetime_df = pd.DataFrame.from_records(all_datetime_df, columns=columns) 

        # Convert just the 'startTime' column to the desired string format
        if len(all_datetime_df) > 0:
            all_datetime_df['startTime'] = all_datetime_df['startTime'].dt.strftime('%Y-%m-%d_%H-%M-%S')
            
        all_datetime_df['is_included'] = True ## all true by default
        is_included = all_datetime_df['is_included'].to_numpy()
        
        if excluded_data_datetimes is not None:
            is_included_considering_exclusions = np.logical_not(all_datetime_df['startTime'].isin(excluded_data_datetimes))
            is_included = np.logical_and(is_included, is_included_considering_exclusions)
            
        if minimum_recording_duration_hours is not None:
            is_included_considering_duration = (all_datetime_df['duration (h)'] >= minimum_recording_duration_hours)
            is_included = np.logical_and(is_included, is_included_considering_duration)
            duration_excluded_indicies = np.where(np.logical_not(is_included_considering_duration))[0]
            if len(duration_excluded_indicies) > 0:
                print(f'excluded duration_excluded_indicies: {duration_excluded_indicies} because they were shorter than minimum_recording_duration_hours: {minimum_recording_duration_hours}')
            
        ## INPUTS: is_included
        all_datetime_df['is_included'] = is_included
        included_indicies = np.where(is_included)[0]

        ## filter out the excluded ones
        included_only_datetime_df = all_datetime_df[all_datetime_df['is_included']].reset_index(drop=True)
        found_raw_data_paths = [v for i, v in enumerate(found_raw_data_paths) if i in included_indicies]

        ## sanity check the number of channels
        raw_data_derived_n_channels_list: List[int] = included_only_datetime_df['n_channels'].tolist()
        if len(raw_data_derived_n_channels_list) == 0:
            print(f'WARNING: raw_data_derived_n_channels_list is empty!')
            raise ValueError(f'raw_data_derived_n_channels_list is empty!')
        raw_data_derived_n_channels: int = raw_data_derived_n_channels_list[0]
        assert np.all([(v == raw_data_derived_n_channels) for v in raw_data_derived_n_channels_list]), f"raw_data_derived_n_channels_list entires must all be the same but: {raw_data_derived_n_channels_list}, n_channels: {n_channels}"
        assert raw_data_derived_n_channels == n_channels, f"raw_data_derived_n_channels: {raw_data_derived_n_channels} != n_channels: {n_channels}"


        print(f'output csv path: "{csv_out_path.as_posix()}"')
        included_only_datetime_df.to_csv(csv_out_path)

        return included_only_datetime_df, csv_out_path, found_raw_data_paths, all_datetime_df, all_found_files_dict


    @classmethod
    def build_initial_position_data(cls, sess, position_csvs_path: Path):
        """ builds the initial position data, creating the output '*.position.npy' file.

        If ``sess.position`` is already loaded or ``{basename}.position.npy`` exists, skips
        Optitrack CSV processing and returns the existing position data.

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
        position_npy_path: Path = sess.filePrefix.with_suffix('.position.npy')

        if sess.position is not None:
            print(f'Skipping position build; session already has position loaded.')
            return sess.position
        if position_npy_path.exists():
            print(f'Skipping position CSV build; loading existing file: "{position_npy_path.as_posix()}"')
            position: Position = Position.from_file(position_npy_path)
            sess.position = position
            return position
        if not position_csvs_path.exists():
            raise FileNotFoundError(f"Neither position CSV directory ('{position_csvs_path.as_posix()}') nor position file ('{position_npy_path.as_posix()}') exists.")

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

        # position.compute_speed_info() ## make sure we have computed fields (I guess.. we don't have to store these actually)

        ## Save out to file:
        position.filename = sess.filePrefix.with_suffix('.position.npy')
        print(f'saving out to {position.filename.as_posix()}...')
        position.save()
        print(f'\tdone.')
        
        ## update session's position
        sess.position = position

        return position
    

    @classmethod
    def concatenate_to_output_file(cls, input_paths: List[Path], output_path: Path, disable_single_file_symlink: bool = False):
        """ concatenates the binary files in input_paths to a single joined output_path.
        When there is only one input file, creates a symlink at output_path instead of copying,
        unless ``disable_single_file_symlink`` is True.

        Raises ``RuntimeError`` if the output file size does not equal the sum of input file sizes.
        """
        expected_size_bytes: int = sum(input_path.stat().st_size for input_path in input_paths)
        if len(input_paths) == 1 and not disable_single_file_symlink:
            print(f'Linking single input file to output...')
            print(f'\tInput:  "{input_paths[0].as_posix()}"')
            print(f'\tOutput: "{output_path.as_posix()}"')
            output_path.symlink_to(input_paths[0].resolve())
        else:
            print(f'Concatenating {len(input_paths)} input files to output...')
            for input_path in input_paths:
                print(f'\t"{input_path.as_posix()}"')
            print(f'\tOutput: "{output_path.as_posix()}"')
            with open(output_path, "wb") as outfile:
                for path in input_paths:
                    with open(path, "rb") as infile:
                        shutil.copyfileobj(infile, outfile)
        print(f'\tdone.')
        output_size_bytes: int = output_path.stat().st_size
        if output_size_bytes != expected_size_bytes:
            size_difference_bytes: int = output_size_bytes - expected_size_bytes
            raise RuntimeError(f'Concatenated output file size ({output_size_bytes} bytes) does not match expected sum of input files ({expected_size_bytes} bytes). Difference: {size_difference_bytes:+d} bytes. Output path: "{output_path.as_posix()}"')
                

    @classmethod
    def step_perform_concat(cls, found_raw_data_paths: List[Path], basedir: Path, basename: str='continuous_combined', force_overwrite: bool=False):
        """ performs the concatenation, creates the output directory if needed

        When the output file already exists, its size is compared to the sum of the input file sizes.
        If they match, concatenation is skipped. If they do not match, a warning is printed and
        concatenation is skipped by default unless ``force_overwrite`` is True.

        When ``force_overwrite`` is True and the output file already exists with a size mismatch, the user is prompted to back up the
        existing file by renaming it with a ``.bak`` suffix before concatenation proceeds. If the ``.bak`` file
        already exists, an error is raised.

        Usage:		
            ## Copy the concatenated files to the output directory
            concatenated_file_output_path: Path = RawDataInitializationMixin.step_perform_concat(found_raw_data_paths=found_raw_data_paths, spyk_circ_output_dir=spyk_circ_output_dir)
            print(f'have concatenated_file_output_path: "{concatenated_file_output_path.as_posix()}"')
            concatenated_file_output_path

        """
        assert basedir.exists(), f"basedir: {basedir} does not exist."        

        ## Copy the concatenated files to the output directory
        concatenated_file_output_path: Path = basedir.joinpath(f'{basename}.dat').resolve() ## do I need to do anything with the adjacent `timestamps.npy` or anything??
        if concatenated_file_output_path.exists() and len(found_raw_data_paths) == 0:
            return concatenated_file_output_path
        expected_size_bytes: int = sum(input_path.stat().st_size for input_path in found_raw_data_paths)
        if concatenated_file_output_path.exists():
            existing_size_bytes: int = concatenated_file_output_path.stat().st_size
            if existing_size_bytes == expected_size_bytes:
                return concatenated_file_output_path
            size_difference_bytes: int = existing_size_bytes - expected_size_bytes
            print(f'WARNING: Existing output file size ({existing_size_bytes} bytes) does not match expected sum of input files ({expected_size_bytes} bytes). Difference: {size_difference_bytes:+d} bytes.')
            print(f'\tExisting file: "{concatenated_file_output_path.as_posix()}"')
            if not force_overwrite:
                print(f'\tSkipping concatenation. Pass force_overwrite=True to re-concatenate.')
                return concatenated_file_output_path
            backup_path: Path = concatenated_file_output_path.with_suffix(concatenated_file_output_path.suffix + '.bak')
            backup_response: str = input(f'Output file already exists: "{concatenated_file_output_path.as_posix()}". Back up existing file to "{backup_path.as_posix()}"? [y/N]: ').strip().lower()
            if backup_response in ('y', 'yes'):
                if backup_path.exists():
                    raise FileExistsError(f'Backup path already exists, refusing to overwrite: "{backup_path.as_posix()}"')
                print(f'backing up existing file to: "{backup_path.as_posix()}"...')
                concatenated_file_output_path.rename(backup_path)
                print(f'\tdone.')
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
    def step_deploy_session_root_xml_template(cls, valid_reference_session_basepath: Path, ref_basename: str,
                                    target_session_basepath: Path, target_basename: str = 'RatS-Day1Openfield'):
        """ copies the potentially missing probe file from an existing valid session's.

        Usage:

            neuroscope_obj = cls.step_deploy_session_root_xml_template(valid_reference_session_basepath='', ref_basename = 'RatS-Day5TwoNovel-2020-12-04_07-55-09',
                                                target_session_basepath=sess.basepath, target_basename = 'RatS-Day1Openfield')

        """
        # from neuropy.core.probe import Shank, Probe, ProbeGroup
        from neuropy.io.neuroscopeio import NeuroscopeIO

        xml_path = target_session_basepath.joinpath(f"{target_basename}.xml").resolve()

        ## Reference Object
        # nrs_path = Path(r"W:\Data\Bapun\RatK\Day4Openfield\RatK_Day4_2019-08-16_04-42-36.nrs").resolve()
        # ref_nrs_path = Path(r"W:\Data\Bapun\RatS\Day5TwoNovel\RatS-Day5TwoNovel-2020-12-04_07-55-09.nrs").resolve()
        # ref_xml_path = Path(r"W:\Data\Bapun\RatS\Day5TwoNovel\RatS-Day5TwoNovel-2020-12-04_07-55-09.xml").resolve()
        ref_xml_path = valid_reference_session_basepath.joinpath(f"{ref_basename}.xml")
        did_copy_new_session_xml_file: bool = subfn_copy_if_needed(ref_xml_path, xml_path)


        ref_neuroscope_obj: NeuroscopeIO = NeuroscopeIO(xml_filename=ref_xml_path)
        ref_good_channels = ref_neuroscope_obj.good_channels
        print(f'ref_good_channels: {ref_good_channels}')
        ref_channel_groups_dict: Dict[int, np.array] = {(grp_idx+1):channels_list for grp_idx, channels_list in enumerate(ref_neuroscope_obj.channel_groups.tolist())} ## 1-indexed dict of channel groups
        ref_channel_groups_dict

        ## Target to update
        
        
        # xml_path = Path(r"W:\Data\Bapun\RatS\Day1Openfield\RatS-Day1Openfield.xml").resolve()
        neuroscope_obj: NeuroscopeIO = NeuroscopeIO(xml_filename=xml_path)
        good_channels = neuroscope_obj.good_channels
        print(f'good_channels: {good_channels}')
        channel_groups_dict: Dict[int, np.array] = {(grp_idx+1):channels_list for grp_idx, channels_list in enumerate(neuroscope_obj.channel_groups.tolist())} ## 1-indexed dict of channel groups
        channel_groups_dict

        is_channel_missing_original = np.isin(ref_good_channels, good_channels)
        assert np.all(is_channel_missing_original), f"the later reference session should not contain any channels that don't exist in the unfiltered session.\n\tref_good_channels (ref only): {ref_good_channels[is_channel_missing_original]}"
        is_channel_marked_bad_ref = np.logical_not(np.isin(good_channels, ref_good_channels))
        channels_marked_bad_in_ref = good_channels[is_channel_marked_bad_ref]
        channels_marked_bad_in_ref

        ## Update curr from ref
        neuroscope_obj.skipped_channels = ref_neuroscope_obj.skipped_channels
        neuroscope_obj.discarded_channels = ref_neuroscope_obj.discarded_channels
        neuroscope_obj._good_channels() ## update good channels on the object
        neuroscope_obj.channel_groups = ref_neuroscope_obj.channel_groups
        _bak_xml_path = neuroscope_obj.backup_xml_file()
        neuroscope_obj.update_xml_file()


        # probe_file = sess.filePrefix.with_suffix('.probegroup.npy')
        # probe_file = Path(r"W:\Data\Bapun\RatS\Day5TwoNovel\RatS-Day5TwoNovel-2020-12-04_07-55-09.probegroup.npy").resolve()
        # assert target_probe_file.exists(), f"probe_file: '{target_probe_file.as_posix()}'"
        # probe_group: ProbeGroup = ProbeGroup.from_file(target_probe_file)
        # probe_group

        return neuroscope_obj


    # ==================================================================================================================================================================================================================================================================================== #
    # spyking_circus functions                                                                                                                                                                                                                                                             #
    # ==================================================================================================================================================================================================================================================================================== #
    @classmethod
    def step_deploy_spyking_circus_files_from_template(cls, valid_reference_session_basepath: Path, ref_basename: str,
                                    target_session_basepath: Path, target_basename: str = 'RatS-Day1Openfield'):
        """ copies the potentially missing probe file from an existing valid session's.

        Usage:

            cls.step_deploy_spyking_circus_files_from_template(valid_reference_session_basepath='', ref_basename = 'RatS-Day5TwoNovel-2020-12-04_07-55-09',
                                                target_session_basepath=sess.basepath, target_basename = 'RatS-Day1Openfield')

        """
        # from neuropy.core.probe import Shank, Probe, ProbeGroup
        from neuropy.io.neuroscopeio import NeuroscopeIO

        ## Reference Object
        ref_spykcirc_path = valid_reference_session_basepath.joinpath('spyk-circ')
        target_spykcirc_path = target_session_basepath.joinpath('spyk-circ')
        target_spykcirc_path.mkdir(exist_ok=True)

        ## Params File - "W:/Data/Bapun/RatS/Day1Openfield/spyk-circ/RatS-Day1Openfield.params":
        ref_params_file = ref_spykcirc_path.joinpath(f"{ref_basename}.params")
        target_params_file = target_spykcirc_path.joinpath(f"{target_basename}.params")
        did_copy_params_file: bool = subfn_copy_if_needed(ref_params_file, target_params_file)
        if did_copy_params_file and (ref_basename != target_basename) and target_params_file.exists():
            params_text = target_params_file.read_text()
            params_text_updated = params_text.replace(ref_basename, target_basename)
            if params_text_updated != params_text:
                target_params_file.write_text(params_text_updated)
                print(f'updated basenames in params file: "{target_params_file.as_posix()}"')


        ## Probe File - "W:/Data/Bapun/RatS/Day1Openfield/spyk-circ/RatS-Day1Openfield.prb"
        ref_probe_file = ref_spykcirc_path.joinpath(f"{ref_basename}.prb")
        target_probe_file = target_spykcirc_path.joinpath(f"{target_basename}.prb")
        subfn_copy_if_needed(ref_probe_file, target_probe_file)


    @classmethod
    def step_update_session_files_n_channels(cls, basedir: Path, n_channels: int, *, basename: Optional[str] = None, dry_run: bool = False, debug_print: bool = True) -> Dict[str, int]:
        """Set channel count in Spyking Circus / Neuroscope session config files.

        Parameters
        ----------
        basedir : Path
            Session folder (e.g. .../RatS/Day1Openfield).
        n_channels : int
            Value written to each target file that exists.
        session_stem : str, optional
            File prefix; default RatS-{basedir.name}.
        dry_run : bool
            If True, print would-be updates without writing.
        debug_print : bool
            If True, print when an expected target file is not found.

        Returns
        -------
        dict
            relative_path -> previous integer for files that existed and were changed.
        Prints each changed file; missing targets are reported when debug_print is True.


        Usage:
            # Example:
            prev = RawDataInitializationMixin.step_update_session_files_n_channels(Path('/mnt/w/Data/Bapun/RatS/Day1Openfield'), 195)
            print(prev)


        """
        # (relative_path_from_session_basedir, line replacer)
        _ChannelFileUpdate = Tuple[str, Callable[[str, int], str]]

        def _replace_prb_total_nb_channels(text: str, n_channels: int) -> str:
            new_text, n = re.subn(r'^total_nb_channels\s*=\s*\d+\s*$', f'total_nb_channels = {n_channels}', text, count=1, flags=re.MULTILINE)
            if n != 1:
                raise ValueError('expected exactly one total_nb_channels assignment')
            return new_text


        def _replace_params_nb_channels(text: str, n_channels: int) -> str:
            new_text, n = re.subn(r'^nb_channels\s*=\s*\d+\s*$', f'nb_channels = {n_channels}', text, count=1, flags=re.MULTILINE)
            if n != 1:
                raise ValueError('expected exactly one nb_channels assignment in [data]')
            return new_text


        def _replace_xml_n_channels(text: str, n_channels: int) -> str:
            new_text, n = re.subn(r'<nChannels>\d+</nChannels>', f'<nChannels>{n_channels}</nChannels>', text, count=1)
            if n != 1:
                raise ValueError('expected exactly one <nChannels> element')
            return new_text

        ########### BEGIN FUNCTION BODY
        if n_channels < 1:
            raise ValueError(f'n_channels must be >= 1, got {n_channels}')
        base = Path(basedir)
        if not base.is_dir():
            raise FileNotFoundError(f'session basedir not found: {base}')
        stem = basename or f'RatS-{base.name}'
        updates: List[_ChannelFileUpdate] = [
            (f'{stem}.prb', _replace_prb_total_nb_channels),
            (f'{stem}.params', _replace_params_nb_channels),
            (f'{stem}.xml', _replace_xml_n_channels),
            (f'spyk-circ/{stem}.prb', _replace_prb_total_nb_channels),
            (f'spyk-circ/{stem}.params', _replace_params_nb_channels),
            (f'spyk-circ/{stem}.xml', _replace_xml_n_channels),
        ]
        replace_dict: Dict[str, Tuple[int, int]] = {}
        for rel_path, replacer in updates:
            path = base / rel_path
            if not path.is_file():
                if debug_print:
                    print(f'path: "{path.as_posix()}": not found.')
                continue
            text = path.read_text(encoding='utf-8')
            if rel_path.endswith('.prb'):
                m = re.search(r'^total_nb_channels\s*=\s*(\d+)\s*$', text, flags=re.MULTILINE)
            elif rel_path.endswith('.params'):
                m = re.search(r'^nb_channels\s*=\s*(\d+)\s*$', text, flags=re.MULTILINE)
            else:
                m = re.search(r'<nChannels>(\d+)</nChannels>', text)
            if not m:
                print(f'path: "{path.as_posix()}": could not read current channel count')
                continue
            old_n = int(m.group(1))
            new_text = replacer(text, n_channels)
            if new_text == text:
                if debug_print:
                    print(f'path: "{path.as_posix()}": (new_text == old_text): (n_channels: {n_channels}).')
                continue
            replace_dict[path] = (old_n, n_channels)
            action = 'would update' if dry_run else 'updated'
            print(f'path: "{path.as_posix()}" ({rel_path}): {action}: n_channels {old_n} -> {n_channels}')
            if not dry_run:
                path.write_text(new_text, encoding='utf-8', newline='')
        ## END for rel_path, replacer in updat...

        return replace_dict


    @classmethod
    def prepare_for_spyking_circus(cls, basedir: Path, basename: str, n_channels: int, 
        valid_reference_session_basepath: Optional[Path]=None, ref_basename: Optional[str]=None,
        dry_run: bool = False, debug_print: bool = True) -> Dict[str, int]:
        """Set channel count in Spyking Circus / Neuroscope session config files.

        Parameters
        ----------
        basedir : Path
            Session folder (e.g. .../RatS/Day1Openfield).
        n_channels : int
            Value written to each target file that exists.
        session_stem : str, optional
            File prefix; default RatS-{basedir.name}.
        dry_run : bool
            If True, print would-be updates without writing.
        debug_print : bool
            If True, print when an expected target file is not found.

        Returns
        -------
        dict
            relative_path -> previous integer for files that existed and were changed.
        Prints each changed file; missing targets are reported when debug_print is True.


        Usage:
            # Example:
            _out = cls.prepare_for_spyking_circus(basedir=basedir, basename = sess.name, n_channels=n_channels)

        """
        if (valid_reference_session_basepath is not None) and (ref_basename is not None):
            ## copy probe/xml as needed
            cls.step_deploy_spyking_circus_files_from_template(valid_reference_session_basepath=valid_reference_session_basepath, ref_basename = ref_basename,
                                                target_session_basepath=basedir, target_basename = basename)
            
        else:
            print(f'WARNING: require valid_reference_session_basepath to copy files, but valid_reference_session_basepath is None')

        ## Replace the number of channels in the session's files to make them correct for the session
        ## INPUTS: basedir
        replace_dict = RawDataInitializationMixin.step_update_session_files_n_channels(basedir=basedir, basename=basename, n_channels=n_channels, dry_run=dry_run, debug_print=debug_print)
        print(replace_dict)

        return replace_dict


    @classmethod
    def _inspect_phy_folder(cls, phy_folder: Path) -> PhyFolderKind:
        if not phy_folder.is_dir():
            return "invalid"
        for filename in _PHY_REQUIRED_FILES:
            if not (phy_folder / filename).is_file():
                return "invalid"
        if (phy_folder / "cluster_si_unit_ids.tsv").is_file():
            return "si_phy_export"
        return "spyk_circ_phy_export"


    @classmethod
    def _detect_neuron_source_type(cls, phy_folder: Path, curation_review_path: Optional[Path] = None) -> NeuronSourceTypeResolved:
        if curation_review_path is not None and curation_review_path.is_file():
            return "sorting"
        if cls._inspect_phy_folder(phy_folder) == "invalid":
            raise FileNotFoundError(f"phy_folder is not a valid Phy export: {phy_folder}")
        return "spyk_circ"


    @classmethod
    def _resolve_neuron_load_paths(cls, config: NeuronLoadConfig, basedir: Path, basename: str) -> tuple[Path, Optional[Path], NeuronSourceTypeResolved]:
        basedir = windows_to_wsl_path_if_needed(basedir).resolve()
        if config.run_name is not None:
            sorting_root = basedir.joinpath("SORTING")
            phy_folder = config.phy_folder.resolve() if config.phy_folder is not None else sorting_root.joinpath(f"{config.run_name}_phy_curated").resolve()
            curation_review_path = config.curation_review_path.resolve() if config.curation_review_path is not None else sorting_root.joinpath(f"{config.run_name}_curation_review.csv").resolve()
            if not curation_review_path.is_file():
                raise FileNotFoundError(f"sorting neuron source requires curation_review_path; phy_folder={phy_folder}")
            return phy_folder, curation_review_path, "sorting"
        phy_folder = config.phy_folder
        curation_review_path = config.curation_review_path
        if phy_folder is None:
            phy_folder = basedir.joinpath("spyk-circ", basename, f"{basename}-merged.GUI").resolve()
        else:
            phy_folder = windows_to_wsl_path_if_needed(phy_folder).resolve()
        if curation_review_path is not None:
            curation_review_path = windows_to_wsl_path_if_needed(curation_review_path).resolve()
        if config.source_type == "sorting":
            if curation_review_path is None or not curation_review_path.is_file():
                raise FileNotFoundError(f"sorting neuron source requires curation_review_path; phy_folder={phy_folder}")
            return phy_folder, curation_review_path, "sorting"
        if config.source_type == "spyk_circ":
            if cls._inspect_phy_folder(phy_folder) == "invalid":
                raise FileNotFoundError(f"phy_folder is not a valid Phy export: {phy_folder}")
            return phy_folder, None, "spyk_circ"
        source_type = cls._detect_neuron_source_type(phy_folder, curation_review_path)
        return phy_folder, curation_review_path if source_type == "sorting" else None, source_type


    @classmethod
    def _read_phy_params(cls, phy_folder: Path) -> dict[str, str]:
        params: dict[str, str] = {}
        with (phy_folder / "params.py").open("r") as f:
            for line in f:
                line_values = line.replace("\n", "").replace('r"', '"').replace('"', "").split("=")
                params[line_values[0].strip()] = line_values[1].strip()
        return params


    @classmethod
    def _load_neurons_from_phyio(cls, phy_folder: Path, *, t_stop: float, include_groups: tuple[str, ...] = ("good", "mua")) -> object:
        from neuropy.core import Neurons
        from neuropy.io import PhyIO

        if cls._inspect_phy_folder(phy_folder) == "invalid":
            raise FileNotFoundError(f"phy_folder is not a valid Phy export: {phy_folder}")
        phy_data = PhyIO(phy_folder, include_groups=include_groups)
        if phy_data.spiketrains is None or len(phy_data.spiketrains) == 0:
            raise ValueError(f"no spiketrains found in Phy output at {phy_folder}")
        neuron_ids = phy_data.cluster_info["si_unit_id"].astype(int).values if "si_unit_id" in phy_data.cluster_info.columns else None
        return Neurons(np.array(phy_data.spiketrains, dtype=object), t_stop=t_stop, sampling_rate=phy_data.sampling_rate, peak_channels=phy_data.peak_channels, waveforms=np.array(phy_data.peak_waveforms, dtype="object"), shank_ids=np.array([int(v) for v in phy_data.shank_ids]), neuron_ids=neuron_ids)


    @classmethod
    def _load_neurons_from_sorting_phy_csv(cls, phy_folder: Path, curation_review_path: Path, *, t_stop: float, unit_filter: str) -> object:
        from neuropy.core import Neurons

        if cls._inspect_phy_folder(phy_folder) == "invalid":
            raise FileNotFoundError(f"phy_folder is not a valid Phy export: {phy_folder}")
        if not curation_review_path.is_file():
            raise FileNotFoundError(f"curation_review_path does not exist: {curation_review_path}")
        review = pd.read_csv(curation_review_path, index_col=0)
        review.index.name = "si_unit_id"
        selected_review = review.query(unit_filter)
        if selected_review.empty:
            raise ValueError(f"unit_filter={unit_filter!r} matched no units in {curation_review_path}")
        selected_si_unit_ids = selected_review.index.astype(int)
        params = cls._read_phy_params(phy_folder)
        sampling_rate = int(float(params["sample_rate"]))
        spktime = np.load(phy_folder / "spike_times.npy")
        clu_ids = np.load(phy_folder / "spike_clusters.npy")
        spk_templates_id = np.load(phy_folder / "spike_templates.npy")
        spk_templates = np.load(phy_folder / "templates.npy")
        cluster_info = pd.read_csv(phy_folder / "cluster_info.tsv", sep="\t")
        cluster_si = pd.read_csv(phy_folder / "cluster_si_unit_ids.tsv", sep="\t")
        selected_clusters = cluster_si[cluster_si["si_unit_id"].isin(selected_si_unit_ids)]
        missing_ids = set(selected_si_unit_ids) - set(selected_clusters["si_unit_id"].astype(int))
        if missing_ids:
            raise ValueError(f"Missing cluster mappings for si_unit_ids: {sorted(missing_ids)}")
        spiketrains, peak_waveforms, peak_channels, shank_ids, neuron_ids = [], [], [], [], []
        for row in selected_clusters.itertuples():
            clu_id, si_unit_id = int(row.cluster_id), int(row.si_unit_id)
            info_rows = cluster_info[cluster_info["si_unit_id"] == si_unit_id]
            if info_rows.empty:
                raise ValueError(f"cluster_info missing si_unit_id={si_unit_id}")
            info = info_rows.iloc[0]
            spike_locs = np.where(clu_ids == clu_id)[0]
            cell_template_id, counts = np.unique(spk_templates_id[spike_locs], return_counts=True)
            template = spk_templates[cell_template_id[np.argmax(counts)]].squeeze().T
            spiketrains.append(spktime[spike_locs] / sampling_rate)
            peak_waveforms.append(template[np.argmax(np.max(template, axis=1))])
            peak_channels.append(int(info["ch"]))
            shank_ids.append(int(info["sh"]))
            neuron_ids.append(si_unit_id)
        return Neurons(np.array(spiketrains, dtype=object), t_stop=t_stop, sampling_rate=sampling_rate, peak_channels=np.array(peak_channels), waveforms=np.array(peak_waveforms, dtype=object), shank_ids=np.array(shank_ids), neuron_ids=np.array(neuron_ids), extended_neuron_properties_df=selected_review.copy())


    @classmethod
    def _finalize_loaded_neurons(cls, sess, neurons: object, *, estimate_neuron_type: bool = True, save_neurons: bool = True) -> object:
        sess.neurons = neurons
        if estimate_neuron_type:
            try:
                from neuropy.utils import neurons_util
                neurons.spiketrains = np.array([np.squeeze(a_spike_train) for a_spike_train in neurons.spiketrains.tolist()], dtype=object)
                neuron_type = neurons_util.estimate_neuron_type(sess.neurons, plot=False)
                neurons.neuron_type = neuron_type[0]
            except Exception as e:
                print(f'WARNING: build_neurons_from_phy: failed to estimate neuron type: {e}')
        if save_neurons:
            neurons.filename = sess.filePrefix.with_suffix('.neurons')
            try:
                print(f'saving out to {neurons.filename.as_posix()}...')
                neurons.save()
                print(f'\tdone.')
            except Exception as e:
                print(f'WARNING: build_neurons_from_phy: failed to save neurons to "{neurons.filename.as_posix()}": {e}')
        return neurons


    @classmethod
    def build_neurons_from_phy(cls, sess, basedir: Path, *, phy_folder: Optional[Path] = None, curation_review_path: Optional[Path] = None, sorting_run_name: Optional[str] = None, neuron_load_config: Optional[NeuronLoadConfig] = None):
        """Loads neurons from Phy output and saves session file.

        Accepts a direct phy_folder path; folder type is determined by on-disk Phy export contents.
        CSV-filtered loading is used only when curation_review_path or sorting_run_name is explicitly provided.

        Modifies:
            sess.neurons

        Usage:

            cls.build_neurons_from_phy(sess, basedir=basedir)
            cls.build_neurons_from_phy(sess, basedir=basedir, phy_folder=basedir / "SORTING/folder_KS4_v1_phy_curated")
            cls.build_neurons_from_phy(sess, basedir=basedir, sorting_run_name="folder_KS4_v1")

        """
        config = neuron_load_config or NeuronLoadConfig(phy_folder=Path(phy_folder) if phy_folder is not None else None, curation_review_path=Path(curation_review_path) if curation_review_path is not None else None, run_name=sorting_run_name)
        basename = getattr(sess, "name", None) or getattr(getattr(sess, "config", None), "session_name", None)
        if basename is None and config.phy_folder is None and config.run_name is None:
            print('WARNING: build_neurons_from_phy: sess must have .name when phy_folder and run_name are not set')
            return None
        if basename is None:
            basename = "session"
        if sess.eegfile is None:
            print('WARNING: build_neurons_from_phy: sess.eegfile is None; cannot determine t_stop for Neurons')
            return None
        t_stop = sess.eegfile.duration
        try:
            resolved_phy_folder, resolved_review_path, source_type = cls._resolve_neuron_load_paths(config, basedir=Path(basedir), basename=basename)
        except FileNotFoundError as exc:
            print(f'WARNING: build_neurons_from_phy: {exc}')
            return None
        try:
            if source_type == "sorting":
                if resolved_review_path is None:
                    raise FileNotFoundError(f"sorting source requires curation_review_path; phy_folder={resolved_phy_folder}")
                neurons = cls._load_neurons_from_sorting_phy_csv(resolved_phy_folder, resolved_review_path, t_stop=t_stop, unit_filter=config.unit_filter)
                print(f"Loaded {neurons.n_neurons} neurons from sorting phy ({resolved_phy_folder.name}) using filter {config.unit_filter!r}")
            else:
                folder_kind = cls._inspect_phy_folder(resolved_phy_folder)
                neurons = cls._load_neurons_from_phyio(resolved_phy_folder, t_stop=t_stop, include_groups=config.include_groups)
                print(f"Loaded {neurons.n_neurons} neurons from phy export ({resolved_phy_folder.name}, kind={folder_kind}) groups={config.include_groups}")
        except (FileNotFoundError, ValueError) as exc:
            print(f'WARNING: build_neurons_from_phy: failed to load neurons from {resolved_phy_folder}: {exc}')
            return None
        return cls._finalize_loaded_neurons(sess, neurons, estimate_neuron_type=config.estimate_neuron_type, save_neurons=config.save_neurons)


    @classmethod
    def build_mua_pbe_artifact_epochs(cls, sess):
        """Builds MUA, PBE, and artifact epochs from neurons and EEG; saves session files.

        Modifies:
            sess.mua
            sess.recinfo artifact epochs

        Usage:

            cls.build_mua_pbe_artifact_epochs(sess)

        """
        # ==================================================================================================================================================================================================================================================================================== #
        # MUA Epochs                                                                                                                                                                                                                                                                           #
        # ==================================================================================================================================================================================================================================================================================== #

        if sess.neurons is not None:
            mua = sess.neurons.get_mua()
            mua.filename = sess.filePrefix.with_suffix(".mua.npy")
            if mua is not None:
                sess.mua = mua
                print(f'saving out to {mua.filename.as_posix()}...')
                mua.save()
                print(f'\tdone.')


        # ==================================================================================================================================================================================================================================================================================== #
        # PBE Epochs                                                                                                                                                                                                                                                                           #
        # ==================================================================================================================================================================================================================================================================================== #
        from neuropy.analyses.spkepochs import detect_pbe_epochs

        if sess.mua is not None:
            smth_mua = sess.mua.get_smoothed(sigma=0.02)
            pbe = detect_pbe_epochs(smth_mua)
            if pbe is not None:
                pbe.filename = sess.filePrefix.with_suffix('.pbe')
                print(f'saving out to {pbe.filename.as_posix()}...')
                pbe.save()
                print(f'\tdone.')


        # ==================================================================================================================================================================================================================================================================================== #
        # Artifact Epochs                                                                                                                                                                                                                                                                      #
        # ==================================================================================================================================================================================================================================================================================== #
        from neuropy.analyses.artifact import detect_artifact_epochs
        from neuropy.io.spykingcircusio import SpykingCircusIO

        if sess.eegfile is not None:
            signal = sess.eegfile.get_signal()
            art_epochs_file = sess.filePrefix.with_suffix('.artifact.npy')
            art_epochs_evt_path = sess.recinfo.source_file.with_suffix('.evt.art')
            art_epochs = Epoch.from_file(art_epochs_file)
            if art_epochs is None:
                art_epochs = detect_artifact_epochs(signal, thresh=8, edge_cutoff=2, merge=6)
                print(f'writing artifact epochs to {art_epochs_evt_path.as_posix()}...')
                sess.recinfo.write_epochs(epochs=art_epochs, ext='art')
                print(f'\tdone.')
                art_epochs.filename = art_epochs_file
                print(f'saving out to {art_epochs_file.as_posix()}...')
                art_epochs.save()
                print(f'\tdone.')

            ### Add in a buffer before and after each epoch if desired.
            _subfn_add_epoch_buffer(art_epochs.to_dataframe(), 0.2)
            print(f'writing buffered artifact epochs to {art_epochs_evt_path.as_posix()}...')
            sess.recinfo.write_epochs(art_epochs, 'art')
            print(f'\tdone.')

            ### Write `dead_times.txt` file for later processing with SpyKing Circus
            dead_times_txt_path: Path = sess.basepath / 'dead_times.txt'
            print(f'writing dead times to {dead_times_txt_path.as_posix()}...')
            SpykingCircusIO.write_epochs(dead_times_txt_path, epochs=art_epochs)

        return sess


    # ==================================================================================================================================================================================================================================================================================== #
    # Main run function                                                                                                                                                                                                                                                                    #
    # ==================================================================================================================================================================================================================================================================================== #
    @classmethod
    def run_all(cls, basedir: Path,
             basename: str = 'RatS-Day1Openfield', n_channels: int = 195, dat_file_sampling_rate: int = 30000, excluded_data_datetimes: List[str]=None,
             valid_reference_session_basepath: Optional[Path]=None, ref_basename: Optional[str]=None,
             phy_folder: Optional[Path] = None, curation_review_path: Optional[Path] = None,
             sorting_run_name: Optional[str] = None, neuron_load_config: Optional[NeuronLoadConfig] = None,
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

        basedir = Path(basedir)
        assert basedir.exists(), f"basedir: {basedir.as_posix()} does not exist! Is this the path for this computer?"

        session_xml_path: Path = find_first_file_rglob(basedir, '*.xml', recursive=False, raise_on_none_found=False)
        

        needs_session_xml_create: bool = (session_xml_path is None) or (not session_xml_path.exists())
        if needs_session_xml_create:
            print(f'session_xml_path: {session_xml_path.as_posix()} does not yet exist! Must copy from template session! Trying...')
            if basename is None:
                basename = session_xml_path.stem # 'Data/Bapun/RatS/Day1Openfield/RatS-Day1Openfield.xml' -> 'RatS-Day1Openfield'

            if (valid_reference_session_basepath is not None) and (ref_basename is not None):
                ## copy probe/xml as needed
                assert basename is not None
                neuroscope_obj = cls.step_deploy_session_root_xml_template(valid_reference_session_basepath=windows_to_wsl_path_if_needed(valid_reference_session_basepath), ref_basename = ref_basename,
                                                    target_session_basepath=basedir, target_basename = basename)
                
            else:
                print(f'WARNING: require valid_reference_session_basepath to copy files, but valid_reference_session_basepath is None')
        ## END if needs_session_xml_create...


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
        datetime_csv_out_path: Path = basedir.joinpath(f'{basename}.datetime.csv')
        concatenated_file_output_path: Path = basedir.joinpath(f'{basename}.dat').resolve()
        all_datetime_df = None
        all_found_files_dict = None

        if datetime_csv_out_path.exists():
            print(f'Skipping datetime CSV build; loading existing file: "{datetime_csv_out_path.as_posix()}"')
            included_only_datetime_df = pd.read_csv(datetime_csv_out_path)
            found_raw_data_paths: List[Path] = []
            if 'dat_file' in included_only_datetime_df.columns:
                found_raw_data_paths = [Path(v).resolve() for v in included_only_datetime_df['dat_file'].dropna().tolist() if Path(v).exists()]
        elif raw_data_path.exists():
            print(f'raw_data_path: "{raw_data_path.as_posix()}"')
            included_only_datetime_df, datetime_csv_out_path, found_raw_data_paths, all_datetime_df, all_found_files_dict = cls.build_session_datetime_csv(raw_data_path, basename=basename, excluded_data_datetimes=excluded_data_datetimes, n_channels=n_channels, sampling_rate=dat_file_sampling_rate)
        else:
            raise FileNotFoundError(f"Neither Raw_data directory ('{raw_data_path.as_posix()}') nor datetime CSV ('{datetime_csv_out_path.as_posix()}') exists.")

        included_only_datetime_df
        found_raw_data_paths

        ## make the "spyk-circ" output directory
        spyk_circ_output_dir: Path = basedir.joinpath('spyk-circ')
        spyk_circ_output_dir.mkdir(exist_ok=True, parents=True) ## dang I sure hope we're on Windows or I'll add some garbage paths :P

        if concatenated_file_output_path.exists():
            print(f'Skipping concatenation; existing .dat found: "{concatenated_file_output_path.as_posix()}"')
        elif len(found_raw_data_paths) > 0:
            concatenated_file_output_path = cls.step_perform_concat(found_raw_data_paths=found_raw_data_paths, basedir=basedir, basename=basename)
        else:
            print(f'Skipping concatenation; no output .dat and no available source files. Expected path: "{concatenated_file_output_path.as_posix()}"')
        print(f'have concatenated_file_output_path: "{concatenated_file_output_path.as_posix()}"')
        concatenated_file_output_path


        # sess.eegfile ## has an EEG file object
        output_process_resample_cmd: Optional[str] = cls.build_process_resample_command(sess, n_channels=n_channels, dat_sampling_rate_Hz=dat_file_sampling_rate, desired_eeg_sampling_rate_Hz=eeg_sampling_rate)
        output_process_resample_cmd

        ## POSITIONS:
        from neuropy.core import Position

        position_csvs_path: Path = sess.basepath.joinpath('Raw_data/position/CSVs')
        position: Position = RawDataInitializationMixin.build_initial_position_data(sess, position_csvs_path=position_csvs_path)
        


        _out = cls.prepare_for_spyking_circus(basedir=basedir, basename = sess.name, n_channels=n_channels, 
                                            valid_reference_session_basepath=valid_reference_session_basepath, ref_basename=ref_basename,
                                            dry_run=False, debug_print=True)



        cls.build_neurons_from_phy(sess, basedir=basedir, phy_folder=phy_folder, curation_review_path=curation_review_path, sorting_run_name=sorting_run_name, neuron_load_config=neuron_load_config)

        cls.build_mua_pbe_artifact_epochs(sess)


        # ==================================================================================================================================================================================================================================================================================== #
        # `.paradigm.npy`                                                                                                                                                                                                                                                                      #
        # ==================================================================================================================================================================================================================================================================================== #

        if (sess.paradigm is None): #  or (sess.paradigm != epochs_obj)

            ## Unless otherwise provided, try to get the latest common timestamp for all of the session's data fields for the t_start and the earliest for the t_stop
            session_t_start: float = np.nanmax([sess.position.t_start, sess.neurons.t_start])
            session_t_stop: float = np.nanmin([sess.position.t_stop, sess.neurons.t_stop])
            # (session_t_start, session_t_stop)
            # 0.0, 5511.2575

            # art_by_hand = Epoch(pd.DataFrame({"start": [246*60 + 31.1], "stop": [247*60 + 32.766], "label": "by_hand"}))
            epochs_df: pd.DataFrame = pd.DataFrame({'start':[session_t_start],'stop':[session_t_stop],'label':['maze']})
            epochs_df['duration'] = epochs_df['stop'] - epochs_df['start']
            epochs_obj: Epoch = Epoch(epochs_df)
            epochs_obj

            sess.paradigm = epochs_obj

            if epochs_obj is not None:
                epochs_obj.filename = sess.filePrefix.with_suffix('.paradigm.npy')
                print(f'saving out to {epochs_obj.filename.as_posix()}...')
                epochs_obj.save()
                print(f'\tdone.')




        return sess