import csv
import linecache
import re
from typing import Optional
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
import numpy as np
from datetime import datetime
import pandas as pd
from neuropy.utils import position_util, mathutil
from pathlib import Path
from ..core import Position


def getSampleRate(fileName):
    """Get Sample rate from csv header file - not set at 120Hz"""
    toprow = pd.read_csv(fileName, nrows=1, header=None)
    capture_FR = np.asarray(
        toprow[np.where(toprow == "Capture Frame Rate")[1][0] + 1][0], dtype=float
    )
    export_FR = np.asarray(
        toprow[np.where(toprow == "Export Frame Rate")[1][0] + 1][0], dtype=float
    )

    if capture_FR != export_FR:
        print("Careful! capture FR does NOT match export FR. Using export only.")

    return int(export_FR)


def getnframes(fileName):
    """Get nframes from csv header file"""
    toprow = pd.read_csv(fileName, nrows=1, header=None)
    nframes_take = np.asarray(
        toprow[np.where(toprow == "Total Frames in Take")[1][0] + 1][0], dtype=float
    )
    nframes_export = np.asarray(
        toprow[np.where(toprow == "Total Exported Frames")[1][0] + 1][0], dtype=float
    )

    if nframes_take != nframes_export:
        print(
            "CAREFUL! # frames in take does not match # frames exported. Using # frames exported for analysis!"
        )

    return int(nframes_export)


def getnframes_fbx(fileName):
    fileName = str(fileName)

    with open(fileName) as f:
        next(f)
        for i, line in enumerate(f):

            m = "".join(line)

            if "KeyCount" in m:
                # print("break at i = " + str(i))
                # line_frame = linecache.getline(fileName, i + 2).strip().split(" ")

                break

    return int(m.strip().split(":")[1].strip())


def getunits(fileName):
    """determine if position data is in centimeters or meters"""
    toprow = pd.read_csv(fileName, nrows=1, header=None)
    units = toprow[np.where(toprow == "Length Units")[1][0] + 1][0]

    return units


def posfromCSV(fileName):
    """Import position (and rotation, if exported) data from OptiTrack CSV file.

    Returns:
        x, y, z: 1D position arrays (cm)
        t: 1D relative-time array (seconds)
        rotation: 2D array shape (n_frames, 3 or 4) with rigid body rotation columns
            in CSV order (Euler X,Y,Z when 3 cols; Quaternion X,Y,Z,W when 4 cols), or
            None if the file has no rotation export.
        rotation_format: "euler" | "quaternion" | None matching `rotation`.
    """

    # ---- auto select which columns have rigid body position / rotation -------
    pos_columns = rigidbody_columns = rot_columns = None
    with open(fileName, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        line_count = 0
        for row in reader:
            if "Rigid Body" in row or "RigidBody" in row:
                rigidbody_columns = np.where(
                    np.bitwise_or(
                        np.array(row) == "Rigid Body", np.array(row) == "RigidBody"
                    )
                )[0]

            if "Position" in row:
                pos_columns = np.where(np.array(row) == "Position")[0]
                rot_columns = np.where(np.array(row) == "Rotation")[0]
                break
            line_count += 1

    rigidbody_pos_columns = np.intersect1d(pos_columns, rigidbody_columns)
    rigidbody_rot_columns = np.intersect1d(rot_columns, rigidbody_columns) if rot_columns is not None else np.array([], dtype=int)

    n_rot = int(len(rigidbody_rot_columns))
    if n_rot == 4:
        rotation_format = "quaternion"
        rot_col_names = ['rx', 'ry', 'rz', 'rw']
    elif n_rot == 3:
        rotation_format = "euler"
        rot_col_names = ['rx', 'ry', 'rz']
    else:
        rotation_format = None
        rot_col_names = []

    # second column is time so append that; include rotation cols too (variable width: 0/3/4)
    read_columns = np.unique(np.concatenate([[1], rigidbody_rot_columns, rigidbody_pos_columns])).astype(int)

    posdata = pd.read_csv(fileName, skiprows=line_count + 1, skip_blank_lines=False, usecols=read_columns)

    # pandas reads usecols in CSV-file (sorted) order; map original CSV indices -> df positions
    sorted_cols = np.sort(read_columns)
    t_idx = int(np.where(sorted_cols == 1)[0][0])
    rot_idxs = [int(np.where(sorted_cols == c)[0][0]) for c in rigidbody_rot_columns]
    pos_idxs = [int(np.where(sorted_cols == c)[0][0]) for c in rigidbody_pos_columns]

    t = np.asarray(posdata.iloc[:, t_idx])
    x0 = np.asarray(posdata.iloc[:, pos_idxs[0]])
    y0 = np.asarray(posdata.iloc[:, pos_idxs[1]])
    z0 = np.asarray(posdata.iloc[:, pos_idxs[2]])
    rot0 = {name: np.asarray(posdata.iloc[:, rot_idxs[i]]) for i, name in enumerate(rot_col_names)}

    pos_df: pd.DataFrame = pd.DataFrame(dict(t=t, x0=x0, y0=y0, z0=z0, **rot0))
    pos_df.dropna(axis='index', how='any', subset=['t'], inplace=True) ## drop any entries with NaN timestamps, that is irrecoverable
    t, x0, y0, z0 = pos_df['t'].to_numpy(), pos_df['x0'].to_numpy(), pos_df['y0'].to_numpy(), pos_df['z0'].to_numpy() ## extract cols back to vars
    rot0 = {name: pos_df[name].to_numpy() for name in rot_col_names}

    # if end frames are nan drop those
    # last_nan_region = contiguous_regions(np.isnan(x0))[-1]
    # todo: potential bug here where csv file has missing timestamps at end that go to NaN and then don't get accounted for.
    if np.isnan(x0[-1]):
        t, x0, y0, z0 = t[:-1], x0[:-1], y0[:-1], z0[:-1]
        rot0 = {name: arr[:-1] for name, arr in rot0.items()}

    xfill, yfill, zfill = interp_missing_pos(x0, y0, z0, t)

    # Linear-fill NaN gaps per rotation channel. NOTE: linear interp is a coarse fill for
    # Euler/quaternion data (slerp would be more correct); kept minimal here to mirror the
    # existing position interpolation pattern.
    rotfill = {}
    for name, arr in rot0.items():
        arr = arr.astype(float).copy()
        idnan = mathutil.contiguous_regions(np.isnan(arr))
        for ids in idnan:
            missing_ids = range(ids[0], ids[-1])
            bracket_ids = ids + [-1, 0]
            try:
                arr[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], arr[bracket_ids])
            except IndexError as e:
                print(f'WARN: skipping malformed rotation interp ids: {ids} for channel {name} instead of aborting entirely.\n\tOccured with error {e}.')
            except Exception:
                raise
        rotfill[name] = arr

    # Now convert to centimeters
    units = getunits(fileName)
    if units.lower() == "centimeters":
        x, y, z = xfill, yfill, zfill
    elif units.lower() == "meters":
        x, y, z = xfill * 100, yfill * 100, zfill * 100
    else:
        raise Exception(
            "position data needs to be exported in either centimeters or meters"
        )

    ## drop NaNs one more time
    pos_df: pd.DataFrame = pd.DataFrame(dict(t=t, x=x, y=y, z=z, **rotfill))
    pos_df.dropna(axis='index', how='any', subset=['t', 'x', 'y'], inplace=True) ## drop any entries with NaN timestamps, that is irrecoverable
    t, x, y, z = pos_df['t'].to_numpy(), pos_df['x'].to_numpy(), pos_df['y'].to_numpy(), pos_df['z'].to_numpy() ## extract cols back to vars

    if rotation_format is None:
        rotation = None
    else:
        rotation = np.column_stack([pos_df[name].to_numpy() for name in rot_col_names])

    return x, y, z, t, rotation, rotation_format


def interp_missing_pos(x, y, z, t):
    """Interpolate missing data points"""
    xgood, ygood, zgood = x, y, z
    idnan = mathutil.contiguous_regions(np.isnan(x))  # identify missing data points

    for ids in idnan:
        missing_ids = range(ids[0], ids[-1])
        bracket_ids = ids + [-1, 0]
        try:
            xgood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], x[bracket_ids])
            ygood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], y[bracket_ids])
            zgood[missing_ids] = np.interp(t[missing_ids], t[bracket_ids], z[bracket_ids])
        except IndexError as e:
            # Allows skiping a few malformed points instead of aborting entirely
            print(f'WARN: skipping malformed interpolation ids: {ids} instead of aborting entirely.\n\tOccured with error {e}.')
        except Exception as e:
            raise

    return xgood, ygood, zgood


def posfromFBX(fileName):
    fileName = str(fileName)

    xpos, ypos, zpos = [], [], []
    with open(fileName) as f:
        next(f)
        for i, line in enumerate(f):

            m = "".join(line)

            if "KeyCount" in m:
                # print("break at i = " + str(i))
                track_begin = i + 2
                line_frame = linecache.getline(fileName, i + 2).strip().split(" ")
                total_frames = int(line_frame[1]) - 1
                break

    with open(fileName) as f:
        for _ in range(track_begin):
            next(f)

        for i, line in enumerate(f):
            # print(line)
            if len(xpos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            xpos.extend(pos1)

        for line in f:

            if "KeyCount" in line:
                # print("break at i = " + str(i))
                break
            # else:  # NRK note: this errors out unpredictably on my computer. Is it necessary? doesn't the for break
            # automatically move onto the next line of code once you reach a "KeyCount" in the line being read?
            #     next(f)

        pos1 = []
        for i, line in enumerate(f):
            # print(i)
            if len(ypos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            ypos.extend(pos1)

        for line in f:

            if "KeyCount" in line:
                # print("break at i = " + str(i))
                break
            # else:
            #     next(f)
        pos1 = []

        for i, line in enumerate(f):
            # print(line)

            if len(zpos) > total_frames:
                break

            elif i < 1:
                # print(i)
                line = line.strip()
                m = line.split(",")
                pos1 = m[1::5]
                # print(pos1)

            else:
                line = line.strip()
                m = line.split(",")
                pos1 = m[2::5]

            # line = next(f)
            zpos.extend(pos1)

    xpos = [float(_) for _ in xpos]
    ypos = [float(_) for _ in ypos]
    zpos = [float(_) for _ in zpos]

    return np.asarray(xpos), np.asarray(ypos), np.asarray(zpos)


def getStartTime(fileName):
    """Get optitrack start time"""
    fileName = str(fileName)

    with open(fileName, newline="") as f:
        reader = csv.reader(f)
        row1 = next(reader)
        StartTime = [
            row1[i + 1] for i in range(len(row1)) if row1[i] == "Capture Start Time"
        ]
    # print(StartTime)
    tbegin = datetime.strptime(StartTime[0], "%Y-%m-%d %I.%M.%S.%f %p")
    return tbegin


def timestamps_from_oe(rec_folder, data_type="continuous"):
    """Gets timestamps for all recordings/experiments in a given recording folder. Assumes you have recorded
    in flat binary format in OpenEphys and left the directory structure intact. continuous data by default,
    set data_type='events' for TTL timestamps"""
    if isinstance(rec_folder, Path):
        oefolder = rec_folder
    else:
        oefolder = Path(rec_folder)

    # Identify and sort timestamp and settings files in ascending order
    if data_type in ["continuous"]:
        time_files = np.asarray(
            sorted(
                oefolder.glob("**/experiment*/**/" + data_type + "/**/timestamps.npy")
            )
        )
    else:
        raise ValueError("data_type must be " "continuous" "")
    set_files = np.asarray(sorted(oefolder.glob("**/settings*.xml")))
    sync_files = np.asarray(sorted(oefolder.glob("**/sync_messages.txt")))

    # Loop through and establish timeframes for each file
    times_abs = []
    for time, set_, sync_file in zip(time_files, set_files, sync_files):
        # load data
        timedata = np.load(time)
        myroot = ET.parse(set_).getroot()
        setdict = {}
        for elem in myroot[0]:
            setdict[elem.tag] = elem.text
        # setdict = XML2Dict(set_)
        SRuse, sync_start = get_sync_info(sync_file)

        # Identify absolute start times of each file...
        tbegin = datetime.strptime(setdict["DATE"], "%d %b %Y %H:%M:%S")
        tstamps = tbegin + pd.to_timedelta((timedata - sync_start) / SRuse, unit="sec")
        if len(times_abs) > 0 and tstamps[0] < times_abs[-1][-1]:
            raise Exception("Timestamps out of order - check directory structure!")
        times_abs.append(tstamps)

    return times_abs


def get_sync_info(_sync_file):
    sync_file_read = open(_sync_file).readlines()
    SR = int(
        sync_file_read[1][
            re.search("@", sync_file_read[1])
            .span()[1] : re.search("Hz", sync_file_read[1])
            .span()[0]
        ]
    )
    sync_start = int(
        sync_file_read[1][
            re.search("start time: ", sync_file_read[1])
            .span()[1] : re.search("@[0-9]*Hz", sync_file_read[1])
            .span()[0]
        ]
    )
    return SR, sync_start


class OptitrackIO:
    """

    Usage:
        from neuropy.core.position import Position

        csv_path = Path(r"W:\Data\Bapun\RatU\RatUDay5OpenfieldSD\Raw_data\position\CSV").resolve()
        assert csv_path.exists()
        _out: OptitrackIO = OptitrackIO(dirname=csv_path)
        pos_df: pd.DataFrame = _out.to_dataframe()
        # pos_df

        pos_obj: Position = Position(pos_df, metadata={'sampling_rate': _out.sampling_rate,
                                                    'source': 'from_csvs',
                                                    'source_files': _out.dirname.as_posix(),
                                                        'scale_factor': _out.scale_factor,
                                                        'datetime': _out.datetime,
                                                        'time': _out.time,
                                                        'override_included_csv_files': _out.override_included_csv_files,
                                                        })

        # pos_obj: Position = Position.init(traces=pos_df[['t', 'x', 'y', 'z', 'dt']].to_numpy(), sampling_rate=_out.sampling_rate, metadata={'source': 'from_csvs', 'source_files': _out.dirname.as_posix(), })
        pos_obj

    """
    def __init__(self, dirname, scale_factor=1.0, override_included_csv_files=None) -> None:
        self.dirname = dirname
        self.scale_factor = scale_factor
        self.datetime = None
        self.time = None
        self.override_included_csv_files = override_included_csv_files
        self._parse_folder()


    def _parse_folder(self):
        """get position data from files. All position related files should be in 'position' folder within basepath

        Parameters
        ----------
        method : str, optional
            method to grab file start times: "from_metadata" (default) grabs from metadata.csv file,
                                             "from_files" grabs from timestamps.npy files in open-ephys folders
        scale : float, optional
            scale the extracted coordinates, by default 1.0
        """

        if self.override_included_csv_files is None:
            found_files = sorted((self.dirname).glob("*.csv"))
        else:
            found_files = sorted(self.override_included_csv_files)
            
        assert len(found_files) > 0, f"found no files!"
        
        sampling_rate = getSampleRate(found_files[0])

        # ------- collecting timepoints related to position tracking ------
        posfiles = np.asarray(found_files)
        posfilestimes = np.asarray(
            [
                datetime.strptime(file.stem, "Take %Y-%m-%d %I.%M.%S %p")
                for file in posfiles
            ]
        )
        filesort_ind = np.argsort(posfilestimes).astype(int)
        posfiles = posfiles[filesort_ind]

        postime, posx, posy, posz = [], [], [], []
        posrx, posry, posrz, posrw = [], [], [], []
        datetime_starts, datetime_stops, datetime_nframes = [], [], []
        rotation_format = None

        for file in posfiles:
            print(file)
            tbegin = getStartTime(file)

            if file.with_suffix(".fbx").is_file():
                # Get time ranges for position files
                nframes_pos = getnframes_fbx(file.with_suffix(".fbx"))
                duration = pd.Timedelta(nframes_pos / sampling_rate, unit="sec")
                tend = tbegin + duration
                trange = pd.date_range(start=tbegin, end=tend, periods=nframes_pos)

                x, y, z = posfromFBX(file.with_suffix(".fbx"))
                assert len(x) == nframes_pos
                postime.extend(trange)

                ## FBX path doesn't parse rotation; pad with NaN to preserve length alignment
                file_rotation = None
                file_rotation_format = None

            else:
                x, y, z, trelative, file_rotation, file_rotation_format = posfromCSV(file)
                # Make sure you arent't just importing the header, if so engage except
                assert len(x) > 0
                nframes_pos = len(x)
                trange = tbegin + pd.to_timedelta(trelative, unit="s")
                postime.extend(trange)
                tend = trange[-1]

            datetime_starts.append(tbegin)
            datetime_stops.append(tend)
            datetime_nframes.append(nframes_pos)
            posx.extend(x)
            posy.extend(y)
            posz.extend(z)

            ## ---- accumulate rotation (rx, ry, rz, [rw]); pad with NaN if absent ----
            if file_rotation_format is not None:
                if rotation_format is None:
                    rotation_format = file_rotation_format
                elif rotation_format != file_rotation_format:
                    print(f'WARN: rotation_format mismatch across files (saw "{rotation_format}" then "{file_rotation_format}" in {file.name}). Falling back to rotation_format=None.')
                    rotation_format = "__mismatch__"

            if file_rotation is not None:
                posrx.extend(file_rotation[:, 0])
                posry.extend(file_rotation[:, 1])
                posrz.extend(file_rotation[:, 2])
                if file_rotation.shape[1] >= 4:
                    posrw.extend(file_rotation[:, 3])
                else:
                    posrw.extend(np.full(nframes_pos, np.nan))
            else:
                posrx.extend(np.full(nframes_pos, np.nan))
                posry.extend(np.full(nframes_pos, np.nan))
                posrz.extend(np.full(nframes_pos, np.nan))
                posrw.extend(np.full(nframes_pos, np.nan))

        postime = pd.to_datetime(postime)
        posx = np.asarray(posx)
        posy = np.asarray(posy)
        posz = np.asarray(posz)

        assert len(postime) == len(posx)

        self.x = posx * self.scale_factor
        self.y = posy * self.scale_factor
        self.z = posz * self.scale_factor
        self.datetime_array = postime
        self.datetime_starts = datetime_starts
        self.datetime_stops = datetime_stops
        self.datetime_nframes = datetime_nframes
        self.sampling_rate = sampling_rate

        ## ---- store rotation channels (unscaled); rotation_format is "euler"/"quaternion"/None ----
        if rotation_format == "__mismatch__":
            rotation_format = None
        self.rotation_format = rotation_format
        self.rx = np.asarray(posrx)
        self.ry = np.asarray(posry)
        self.rz = np.asarray(posrz)
        self.rw = np.asarray(posrw) if rotation_format == "quaternion" else None

    def get_position_at_datetimes(self, dt):

        x = np.interp(dt, self.datetime_array, self.x)
        y = np.interp(dt, self.datetime_array, self.y)
        z = np.interp(dt, self.datetime_array, self.z)

        return x, y, z

    def old_stuff(self):
        """get position data from files. All position related files should be in 'position' folder within basepath

        Parameters
        ----------
        method : str, optional
            method to grab file start times: "from_metadata" (default) grabs from metadata.csv file,
                                             "from_files" grabs from timestamps.npy files in open-ephys folders
        scale : float, optional
            scale the extracted coordinates, by default 1.0
        """
        sRate = self._obj.sampfreq  # .dat file sampling frequency
        basePath = Path(self._obj.basePath)
        metadata = self._obj.loadmetadata

        # ------- collecting timepoints related to .dat file  --------
        data_time = []
        # transfer start times from the settings*.xml file and nframes in .dat file to each row of the metadata file
        tracking_sRate = position_util.getSampleRate(
            sorted((self._obj.basePath / "position").glob("*.csv"))[0]
        )
        durations = []
        if method == "from_metadata":
            for i, file_time in enumerate(metadata["StartTime"][:nfiles]):
                tbegin = datetime.strptime(file_time, "%Y-%m-%d_%H-%M-%S")
                nframes = metadata["nFrames"][i]
                duration = pd.Timedelta(nframes / sRate, unit="sec")
                tend = tbegin + duration
                trange = pd.date_range(
                    start=tbegin,
                    end=tend,
                    periods=int(duration.total_seconds() * tracking_sRate),
                )
                data_time.extend(trange)

        # grab timestamps directly from timestamps.npy files. Assumes you have preserved the OE file structure.
        elif method == "from_files":
            times_all = timestamps_from_oe(basePath, data_type="continuous")
            for i, times in enumerate(times_all):
                tbegin, tend = times[0], times[-1]
                duration = tend - tbegin
                durations.append(duration)
                trange = pd.date_range(
                    start=tbegin,
                    end=tend,
                    periods=int(duration.total_seconds() * tracking_sRate),
                )
                data_time.extend(trange)
        data_time = pd.to_datetime(data_time)

        # ------- deleting intervals that were deleted from .dat file after concatenating
        ndeletedintervals = metadata.count()["deletedStart (minutes)"]
        for i in range(ndeletedintervals):
            tnoisy_begin = data_time[0] + pd.Timedelta(
                metadata["deletedStart (minutes)"][i], unit="m"
            )
            tnoisy_end = data_time[0] + pd.Timedelta(
                metadata["deletedEnd (minutes)"][i], unit="m"
            )

            del_index = np.where((data_time > tnoisy_begin) & (data_time < tnoisy_end))[
                0
            ]

            data_time = np.delete(data_time, del_index)

        # ------- collecting timepoints related to position tracking ------
        posFolder = basePath / "position"
        posfiles = np.asarray(sorted(posFolder.glob("*.csv")))
        posfilestimes = np.asarray(
            [
                datetime.strptime(file.stem, "Take %Y-%m-%d %I.%M.%S %p")
                for file in posfiles
            ]
        )
        filesort_ind = np.argsort(posfilestimes).astype(int)
        posfiles = posfiles[filesort_ind]

        postime, posx, posy, posz = [], [], [], []

        for file in posfiles:
            print(file)
            tbegin = position_util.getStartTime(file)

            if file.with_suffix(".fbx").is_file():
                # Get time ranges for position files
                nframes_pos = position_util.getnframes(file)
                duration = pd.Timedelta(nframes_pos / tracking_sRate, unit="sec")
                tend = tbegin + duration
                trange = pd.date_range(start=tbegin, end=tend, periods=nframes_pos)

                x, y, z = position_util.posfromFBX(file.with_suffix(".fbx"))

                postime.extend(trange)

            else:  # First try to load everything from CSV directly
                x, y, z, trelative = position_util.posfromCSV(file)
                # Make sure you arent't just importing the header, if so engage except
                assert len(x) > 0
                trange = tbegin + pd.to_timedelta(trelative, unit="s")
                postime.extend(trange)

            posx.extend(x)
            posy.extend(y)
            posz.extend(z)
        postime = pd.to_datetime(postime[: len(posx)])
        posx = np.asarray(posx)
        posy = np.asarray(posy)
        posz = np.asarray(posz)

        # -------- interpolating positions for recorded data ------------
        xdata = np.interp(data_time, postime, posx) * scale
        ydata = np.interp(data_time, postime, posy) * scale
        zdata = np.interp(data_time, postime, posz) * scale

        time = np.linspace(0, len(xdata) / tracking_sRate, len(xdata))
        posVar = {
            "x": xdata,
            "y": zdata,
            "z": ydata,  # keep this data in case you are interested in rearing activity
            "time": time,
            "datetime": data_time,
            "trackingsRate": tracking_sRate,
        }

        self.x = xdata
        self.y = zdata
        self.z = ydata
        self.time = time
        self.tracking_srate = tracking_sRate

        self.save()


    def to_dataframe(self) -> pd.DataFrame:
        pos_df = pd.DataFrame({'t': self.datetime_array, 'x': self.x, 'y': self.y, 'z': self.z, 'rx': self.rx, 'ry': self.ry, 'rz': self.rz})
        if self.rotation_format == "quaternion" and self.rw is not None:
            pos_df['rw'] = self.rw
        pos_df['dt'] = pos_df['t'].copy() ## convert datetime times to 'dt' column
        pos_df['t'] = (pos_df['t'] - np.nanmin(pos_df['dt'])).dt.total_seconds() ## minimum (first) time to 't' (seconds) column
        pos_df.attrs.update({'srate': self.sampling_rate, 'scale_factor': self.scale_factor, 'rotation_format': self.rotation_format})
        return pos_df
    

    def to_position_obj(self, pos_obj_save_path: Optional[Path]=None):
        """
        
        """
        from neuropy.core.position import Position
        
        pos_df: pd.DataFrame = self.to_dataframe()
        pos_df = pos_df.dropna(how='any', subset=['t', 'x', 'y'], inplace=False) ## drop any NaN values (rotation cols intentionally not in subset)
        pos_obj: Position = Position(pos_df, metadata={'sampling_rate': self.sampling_rate,
                                                    'source': 'from_csvs',
                                                    'source_files': self.dirname.as_posix(),
                                                        'scale_factor': self.scale_factor,
                                                        'datetime': self.datetime,
                                                        'time': self.time,
                                                        'override_included_csv_files': self.override_included_csv_files,
                                                        'rotation_format': self.rotation_format,
                                                        })

        # pos_obj: Position = Position.init(traces=pos_df[['t', 'x', 'y', 'z', 'dt']].to_numpy(), sampling_rate=_out.sampling_rate, metadata={'source': 'from_csvs', 'source_files': _out.dirname.as_posix(), })
        if pos_obj_save_path is not None:
            pos_obj.filename = pos_obj_save_path ## set the filename
            print(f'trying to save pos_obj to "{pos_obj_save_path.as_posix()}"')
            pos_obj.save()
            print(f'\tdone.')
            
        return pos_obj
    

    