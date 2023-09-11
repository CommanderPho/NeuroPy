import ast

import numpy as np
from glob import glob
import os
import re
from pathlib import Path
from xml.etree import ElementTree
import pandas as pd
from datetime import datetime, timezone
from dateutil import tz


def get_us_start(settings_file: str, from_zone="UTC", to_zone="America/Detroit"):
    """Get microsecond time second precision start time from Pho/Timestamp plugin"""

    experiment_meta = XML2Dict(settings_file)

    start_us = experiment_meta["SIGNALCHAIN"]["PROCESSOR"][
        "Utilities/PhoStartTimestamp Processor"
    ]["PhoStartTimestampPlugin"]["RecordingStartTimestamp"]["startTime"]
    dt_start_utc = datetime.strptime(start_us[:-1], "%Y-%m-%d_%H:%M:%S.%f").replace(
        tzinfo=tz.gettz("UTC")
    )
    to_zone = tz.gettz("America/Detroit")

    return dt_start_utc.astimezone(to_zone)


def get_dat_timestamps(basepath: str or Path, sync: bool = False):
    """
    Gets timestamps for each frame in your dat file(s) in a given directory.

    IMPORTANT: in the event your .dat file has less frames than you timestamps.npy file,
    you MUST create a "dropped_end_frames.txt" file with the # of missing frames in the same folder
    to properly account for this offset.

    :param basepath: str, path to parent directory, holding your 'experiment' folder(s).
    :param sync: True = use 'synchronized_timestamps.npy' file, default = False
    :return:
    """
    basepath = Path(basepath)

    timestamp_files = get_timestamp_files(basepath, type="continuous", sync=sync)

    timestamps = []
    for file in timestamp_files:
        set_file = get_settings_filename(file)  # get settings file name
        set_folder = get_set_folder(file)
        try:
            start_time = get_us_start(set_folder / set_file)
            # print("Using precise start time from Pho/Timestamp plugin")
        except KeyError:
            try:
                experiment_meta = XML2Dict(set_folder / set_file)  # Get meta data
                start_time = pd.Timestamp(
                    experiment_meta["INFO"]["DATE"]
                )  # get start time from meta-data
            except FileNotFoundError:
                print(
                    "WARNING:"
                    + str(set_folder / set_file)
                    + " not found. Inferring start time from directory structure. PLEASE CHECK!"
                )
                # Find folder with timestamps
                m = re.search(
                    "[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}",
                    str(set_folder),
                )
                start_time = pd.to_datetime(m.group(0), format="%Y-%m-%d_%H-%M-%S")

        SR, sync_frame = parse_sync_file(
            file.parents[3] / "recording1/sync_messages.txt"
        )  # Get SR and sync frame info
        print("start time = " + str(start_time))
        stamps = np.load(file)  # load in timestamps

        # Remove any dropped end frames.
        if (file.parent / "dropped_end_frames.txt").exists():
            with open((file.parent / "dropped_end_frames.txt"), "rb") as fp:
                nfile = fp.readlines(0)
                pattern = re.compile("[0-9]+")
                ndropped = int(pattern.search(str(nfile[0])).group(0))

            print(f"Dropping last {ndropped} frames per dropped_end_frames.txt file")
            stamps = stamps[0:-ndropped]

        timestamps.append(
            (
                start_time + pd.to_timedelta((stamps - sync_frame) / SR, unit="sec")
            ).to_frame(index=False)
        )  # Add in absolute timestamps, keep index of acquisition system

    return pd.concat(timestamps)


def get_timestamp_files(
    basepath: str or Path, type: str in ["continuous", "TTL"], sync: bool = False
):
    """
    Identify all timestamp files of a certain type
    :param basepath: str of Path object of folder containing timestamp file(s)
    :param type: 'continuous' or 'TTL'
    :param sync: False(default) for 'timestamps.npy', True for 'synchronized_timestamps.npy'
    :return: list of all files in that directory matching the inputs
    """

    timestamps_list = sorted(basepath.glob("**/*timestamps.npy"))
    continuous_bool = ["continuous" in str(file_name) for file_name in timestamps_list]
    TTL_bool = ["TTL" in str(file_name) for file_name in timestamps_list]
    sync_bool = ["synchronized" in str(file_name) for file_name in timestamps_list]
    no_sync_bool = np.bitwise_not(sync_bool)

    if type == "continuous" and not sync:
        file_inds = np.where(np.bitwise_and(continuous_bool, no_sync_bool))[0]
    elif type == "continuous" and sync:
        file_inds = np.where(np.bitwise_and(continuous_bool, sync_bool))[0]
    elif type == "TTL" and not sync:
        file_inds = np.where(np.bitwise_and(TTL_bool, no_sync_bool))[0]
    elif type == "TTL" and sync:
        file_inds = np.where(np.bitwise_and(TTL_bool, sync_bool))[0]

    return [timestamps_list[ind] for ind in file_inds]


def get_lfp_timestamps(dat_times_or_folder, SRdat=30000, SRlfp=1250):
    """
    Gets all timestamps corresponding to a downsampled lfp or eeg file
    :param dat_times_or_folder: str, path to parent directory, holding your 'experiment' folder(s).
    OR pandas dataframe of timestamps from .dat file.
    :param SRdat: sample rate for .dat file
    :param SRlfp: sample rate for .lfp file
    :return:
    """

    if isinstance(dat_times_or_folder, (str, Path)):
        dat_times = get_dat_timestamps(dat_times_or_folder)
    elif isinstance(dat_times_or_folder, (pd.DataFrame, pd.Series)):
        dat_times = dat_times_or_folder

    assert (
        np.round(SRdat / SRlfp) == SRdat / SRlfp
    ), "SRdat file must be an integer multiple of SRlfp "
    return dat_times.iloc[slice(0, None, int(SRdat / SRlfp))]


def load_all_ttl_events(basepath: str or Path, sync: bool = False, **kwargs):
    """Loads TTL events from digital input port on an OpenEphys box or Intan Recording Controller in BINARY format.
    Assumes you have left the directory structure intact! Flexible - can load from just one recording or all recordings.
    Combines all events into one dataframe with datetimes.

    :param TTLpath: folder where TTL files live
    :param sync: continuous data timestamps to use for alignment. False(default) = use 'timestamps.npy',
    True = use 'synchronized_timestamps.npy'
    :param kwargs: accepts all kwargs to load_ttl_events
    """
    basepath = Path(basepath)
    TTLpaths = sorted(basepath.glob("**/TTL*"))  # get all TTL folders
    # Grab corresponding continuous data folders
    exppaths = [file.parents[3] for file in TTLpaths]

    # Concatenate everything together into one list
    events_all, nframes_dat = [], []
    for TTLfolder, expfolder in zip(TTLpaths, exppaths):
        events = load_ttl_events(TTLfolder, **kwargs)
        events_all.append(events)
        nframes_dat.append(get_dat_timestamps(expfolder, sync=sync))

    # Now loop through and make everything into a datetime in case you are forced to use system times to synchronize everything later
    times_list = []
    for ide, events in enumerate(events_all):
        times_list.append(events_to_datetime(events))
        # NRK todo: start here 11/1/2021 - make a continuous running index here to match up with concatenated .dat files!

    # NRK todo: add in recording # as a column for easy reference

    return pd.concat(times_list)


def load_ttl_events(TTLfolder, zero_timestamps=True, event_names="", sync_info=True):
    """Loads TTL events for one recording folder and spits out a dictionary.

    :param TTLfolder: folder where your TTLevents live, recorded in BINARY format.
    :param zero_timestamps: True (default) = subtract start time in sync_messages.txt. This will align everything
    to the first frame in your .dat file.
    :param event_names: can pass a dictionary to keep track of what each event means, e.g.
    event_names = {1: 'optitrack_start', 2: 'lick'} would tell you channel1 = input from optitrack and
    channel2 = animal lick port activations
    :param sync_info: True (default), grabs sync related info if TTlfolder is in openephys directory structure
    :return: channel_states, channels, full_words, and timestamps in ndarrays
    """
    TTLfolder = Path(TTLfolder)

    # Load event times and states into a dict
    events = dict()
    for varname in ["channel_states", "channels", "full_words", "timestamps"]:
        events[varname] = np.load(TTLfolder / (varname + ".npy"))

    # Get sync info
    if sync_info:
        sync_file = TTLfolder.parents[2] / "sync_messages.txt"
        SR, record_start = parse_sync_file(sync_file)
        events["SR"] = SR

        # Zero timestamps
        if zero_timestamps:
            events["timestamps"] = events["timestamps"] - record_start

        # Grab start time from .xml file and keep it with events just in case
        settings_file = TTLfolder.parents[4] / get_settings_filename(TTLfolder)
        try:
            events["start_time"] = pd.to_datetime(
                XML2Dict(settings_file)["INFO"]["DATE"]
            )
        except FileNotFoundError:
            print("Settings file: " + str(settings_file) + " NOT FOUND")

            # Find start time using filename
            p = re.compile(
                "[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-2]+[0-9]+-[0-6]+[0-9]+-[0-6]+[0-9]+"
            )
            events["start_time"] = pd.to_datetime(
                p.search(str(settings_file)).group(0), format="%Y-%m-%d_%H-%M-%S"
            )

            # Print to screen to double check!
            print(
                str(events["start_time"])
                + " loaded from folder structure, be sure to double check!"
            )

    # Last add in event_names dict
    events["event_names"] = event_names

    return events


def events_to_datetime(events):
    """Parses out channel_states and timestamps and calculates absolute datetimes for all events"""

    # First grab relevant keys from full events dictionary
    sub_dict = {key: events[key] for key in ["channel_states", "timestamps"]}
    sub_dict["datetimes"] = events["start_time"] + pd.to_timedelta(
        events["timestamps"] / [events["SR"]], unit="sec"
    )

    # Now dump any event names into the appropriate rows
    if events["event_names"] is not None:
        event_names = np.empty_like(events["timestamps"], dtype=np.dtype(("U", 10)))
        for key in events["event_names"]:
            # Allocate appropriate name to all timestamps for a channel
            event_names[events["channel_states"] == key] = events["event_names"][key]
        sub_dict["event_name"] = event_names

    return pd.DataFrame.from_dict(sub_dict)


def parse_sync_file(sync_file):
    """Grab synchronization info for a given session
    :param sync_file: path to 'sync_messages.txt' file in recording folder tree for that recording.
    :return: sync_frame: int, sync frame # when you hit the record button relative to hitting the play button
             SR: int, sampling rate in Hz. Subtract from TTL timestamps to get dat frame #.
    """

    # Read in file
    sync_lines = open(sync_file).readlines()

    try:
        # Grab sampling rate and sync time based on file structure
        SR = int(
            sync_lines[1][
                re.search("@", sync_lines[1])
                .span()[1] : re.search("Hz", sync_lines[1])
                .span()[0]
            ]
        )
        sync_frame = int(
            sync_lines[1][
                re.search("start time: ", sync_lines[1])
                .span()[1] : re.search("@[0-9]*Hz", sync_lines[1])
                .span()[0]
            ]
        )
    except IndexError:  # Fill in from elsewhere if sync_messages missing info
        parent_dir = Path(sync_file).parent
        timestamp_files = sorted(parent_dir.glob("**/continuous/**/timestamps.npy"))
        assert len(timestamp_files) == 1, "Too many timestamps.npy files"
        sync_frame = np.load(timestamp_files[0])[0]

        structure_file = parent_dir / "structure.oebin"
        with open(structure_file) as f:
            data = f.read()
        structure = ast.literal_eval(data)
        SR = structure["continuous"][0]["sample_rate"]

    return SR, sync_frame


def get_set_folder(child_dir):
    """Gets the folder where your settings.xml file and experiment folders should live."""
    child_dir = Path(child_dir)
    expfolder_id = np.where(
        [
            str(child_dir.parents[id]).find("experiment") > -1
            for id in range(len(child_dir.parts) - 1)
        ]
    )[0].max()

    return child_dir.parents[expfolder_id + 1]


def get_settings_filename(child_dir):
    """Infers the settings file name from a child directory, e.g. the continuous or event recording folder

    :param child_dir: any directory below the top-level directory. Must include the "experiment#' directory!
    :return:
    """

    child_dir = Path(child_dir)
    expfolder = child_dir.parts[
        np.where(["experiment" in folder for folder in child_dir.parts])[0][0]
    ]

    if expfolder[-1] == "1":
        return "settings.xml"
    else:
        return "settings_" + expfolder[10:] + ".xml"


def LoadTTLEvents_full(
    Folder, Processor=None, Experiment=None, Recording=None, TTLport=1, mode="r+"
):
    """Load TTL in events recorded in binary format. Copied from https://github.com/open-ephys/analysis-tools
    Python3.Binary module for loading binary files and adjusted for TTL events. Keeps track of processor and other
    metadata, but probably too onerous for daily use"""
    Files = sorted(glob(Folder + "/**/TTL_" + str(TTLport) + "/*.npy", recursive=True))
    # InfoFiles = sorted(glob(Folder + '/*/*/structure.oebin'))

    event_data, timing_data = {}, {}
    print("Loading events from recording on TTL", TTLport, "...")
    for F, File in enumerate(Files):
        try:
            Exp, Rec, _, Proc, _, npy_file = File.split("/")[-6:]
            sync_file = open(
                os.path.join(
                    File.split("/" + Exp + "/" + Rec)[0] + "/" + Exp + "/" + Rec,
                    "sync_messages.txt",
                )
            ).readlines()
        except ValueError:  # for windows machines use the correct delimiter
            Exp, Rec, _, Proc, _, npy_file = File.split("\\")[-6:]
            sync_file = open(
                os.path.join(
                    File.split("\\" + Exp + "\\" + Rec)[0] + "\\" + Exp + "\\" + Rec,
                    "sync_messages.txt",
                )
            ).readlines()

        Exp = str(int(Exp[10:]) - 1)
        Rec = str(int(Rec[9:]) - 1)
        Proc = Proc.split(".")[-2].split("-")[-1]
        if "_" in Proc:
            Proc = Proc.split("_")[0]

        # Info = literal_eval(open(InfoFiles[F]).read())
        # ProcIndex = [Info['continuous'].index(_) for _ in Info['continuous']
        #              if str(_['recorded_processor_id']) == Proc][0]

        if Proc not in event_data.keys():
            event_data[Proc], timing_data[Proc] = {}, {}
        if Exp not in event_data[Proc]:
            event_data[Proc][Exp], timing_data[Proc][Exp] = {}, {}
        if Rec not in event_data[Proc][Exp]:
            event_data[Proc][Exp][Rec], timing_data[Proc][Exp][Rec] = {}, {}

        timing_data[Proc][Exp][Rec]["Rate"] = sync_file[1][
            re.search("@", sync_file[1])
            .span()[1] : re.search("Hz", sync_file[1])
            .span()[0]
        ]
        timing_data[Proc][Exp][Rec]["start_time"] = sync_file[1][
            re.search("start time: ", sync_file[1])
            .span()[1] : re.search("@[0-9]*Hz", sync_file[1])
            .span()[0]
        ]

        if Experiment:
            if int(Exp) != Experiment - 1:
                continue

        if Recording:
            if int(Rec) != Recording - 1:
                continue

        if Processor:
            if Proc != Processor:
                continue

        event_data[Proc][Exp][Rec][npy_file[:-4]] = np.load(File)

    return event_data, timing_data


"""
Created on 20170704 21:15:19

@author: Thawann Malfatti

Loads info from the settings.xml file.

Examples:
    File = '/Path/To/Experiment/settings.xml

    # To get all info the xml file can provide:
    AllInfo = SettingsXML.XML2Dict(File)

    # AllInfo will be a dictionary following the same structure of the XML file.

    # To get the sampling rate used in recording:
    Rate = SettingsXML.GetSamplingRate(File)

    # To get info only about channels recorded:
    RecChs = SettingsXML.GetRecChs(File)[0]

    # To get also the processor names:
    RecChs, PluginNames = SettingsXML.GetRecChs(File)

    # RecChs will be a dictionary:
    #
    # RecChs
    #     ProcessorNodeId
    #         ChIndex
    #             'name'
    #             'number'
    #             'gain'
    #         'PluginName'

"""


def FindRecProcs(Ch, Proc, RecChs):
    ChNo = Ch["number"]
    Rec = Proc["CHANNEL"][ChNo]["SELECTIONSTATE"]["record"]

    if Rec == "1":
        if Proc["NodeId"] not in RecChs:
            RecChs[Proc["NodeId"]] = {}
        RecChs[Proc["NodeId"]][ChNo] = Ch

    return RecChs


def Root2Dict(El):
    Dict = {}
    if list(El):
        for SubEl in El:
            if SubEl.keys():
                if SubEl.get("name"):
                    if SubEl.tag not in Dict:
                        Dict[SubEl.tag] = {}
                    Dict[SubEl.tag][SubEl.get("name")] = Root2Dict(SubEl)

                    Dict[SubEl.tag][SubEl.get("name")].update(
                        {K: SubEl.get(K) for K in SubEl.keys() if K != "name"}
                    )

                else:
                    Dict[SubEl.tag] = Root2Dict(SubEl)
                    Dict[SubEl.tag].update(
                        {K: SubEl.get(K) for K in SubEl.keys() if K != "name"}
                    )

            else:
                if SubEl.tag not in Dict:
                    Dict[SubEl.tag] = Root2Dict(SubEl)
                else:
                    No = len([k for k in Dict if SubEl.tag in k])
                    Dict[SubEl.tag + "_" + str(No + 1)] = Root2Dict(SubEl)
    else:
        if El.items():
            return dict(El.items())
        else:
            return El.text

    return Dict


def XML2Dict(File):
    Tree = ElementTree.parse(File)
    Root = Tree.getroot()
    Info = Root2Dict(Root)

    return Info


def GetSamplingRate(File):
    Info = XML2Dict(File)
    Error = "Cannot parse sample rate. Check your settings.xml file at SIGNALCHAIN>PROCESSOR>Sources/Rhythm FPGA."
    SignalChains = [_ for _ in Info.keys() if "SIGNALCHAIN" in _]

    try:
        for SignalChain in SignalChains:
            if "Sources/Rhythm FPGA" in Info[SignalChain]["PROCESSOR"].keys():
                if (
                    "SampleRateString"
                    in Info[SignalChain]["PROCESSOR"]["Sources/Rhythm FPGA"]["EDITOR"]
                ):
                    Rate = Info[SignalChain]["PROCESSOR"]["Sources/Rhythm FPGA"][
                        "EDITOR"
                    ]["SampleRateString"]
                    Rate = float(Rate.split(" ")[0]) * 1000
                elif (
                    Info[SignalChain]["PROCESSOR"]["Sources/Rhythm FPGA"]["EDITOR"][
                        "SampleRate"
                    ]
                    == "17"
                ):
                    Rate = 30000
                elif (
                    Info[SignalChain]["PROCESSOR"]["Sources/Rhythm FPGA"]["EDITOR"][
                        "SampleRate"
                    ]
                    == "16"
                ):
                    Rate = 25000
                else:
                    Rate = None
            else:
                Rate = None

        if not Rate:
            print(Error)
            return None
        else:
            return Rate

    except Exception as Ex:
        print(Ex)
        print(Error)
        return None


def GetRecChs(File):
    Info = XML2Dict(File)
    RecChs = {}
    ProcNames = {}

    if len([k for k in Info if "SIGNALCHAIN" in k]) > 1:
        for S in [k for k in Info if "SIGNALCHAIN_" in k]:
            for P, Proc in Info[S]["PROCESSOR"].items():
                Info["SIGNALCHAIN"]["PROCESSOR"][P + "_" + S[-1]] = Proc
            del Info[S]
    #     print('There are more than one signal chain in file. )
    #     Ind = input(')

    for P, Proc in Info["SIGNALCHAIN"]["PROCESSOR"].items():
        if "isSource" in Proc:
            if Proc["isSource"] == "1":
                SourceProc = P[:]
        else:
            if Proc["name"].split("/")[0] == "Sources":
                SourceProc = P[:]

        if "CHANNEL_INFO" in Proc and Proc["CHANNEL_INFO"]:
            for Ch in Proc["CHANNEL_INFO"]["CHANNEL"].values():
                RecChs = FindRecProcs(Ch, Proc, RecChs)

        elif "CHANNEL" in Proc:
            for Ch in Proc["CHANNEL"].values():
                RecChs = FindRecProcs(Ch, Proc, RecChs)

        else:
            continue

        if "pluginName" in Proc:
            ProcNames[Proc["NodeId"]] = Proc["pluginName"]
        else:
            ProcNames[Proc["NodeId"]] = Proc["name"]

    if Info["SIGNALCHAIN"]["PROCESSOR"][SourceProc]["CHANNEL_INFO"]:
        SourceProc = Info["SIGNALCHAIN"]["PROCESSOR"][SourceProc]["CHANNEL_INFO"][
            "CHANNEL"
        ]
    else:
        SourceProc = Info["SIGNALCHAIN"]["PROCESSOR"][SourceProc]["CHANNEL"]

    for P, Proc in RecChs.items():
        for C, Ch in Proc.items():
            if "gain" not in Ch:
                RecChs[P][C].update(
                    [c for c in SourceProc.values() if c["number"] == C][0]
                )

    return (RecChs, ProcNames)


if __name__ == "__main__":
    events_folder = "/data3/Trace_FC/Recording_Rats/Django/2023_03_09_recall1/1_tone_recall/2023-03-09_12-14-31/Record Node 103/experiment1/recording1/events/Intan_Rec._Controller-100.0/TTL_1"
    load_all_ttl_events(events_folder)
    pass
