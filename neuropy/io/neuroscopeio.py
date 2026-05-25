import shutil
import numpy as np
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as Etree
from .. import core


class NeuroscopeIO:
    def __init__(self, xml_filename) -> None:
        self.source_file = Path(xml_filename)
        self.eeg_filename = self.source_file.with_suffix(".eeg")
        self.dat_filename = self.source_file.with_suffix(".dat")
        self.skipped_channels = None
        self.channel_groups = None
        self.discarded_channels = None
        self._parse_xml_file()
        self._good_channels()


    def _parse_xml_file(self):

        tree = Etree.parse(self.source_file)
        myroot = tree.getroot()
        nbits = int(myroot.find("acquisitionSystem").find("nBits").text)

        dat_sampling_rate = n_channels = None
        for sf in myroot.findall("acquisitionSystem"):
            dat_sampling_rate = int(sf.find("samplingRate").text)
            n_channels = int(sf.find("nChannels").text)

        eeg_sampling_rate = None
        for val in myroot.findall("fieldPotentials"):
            eeg_sampling_rate = int(val.find("lfpSamplingRate").text)

        channel_groups, skipped_channels = [], []
        for x in myroot.findall("anatomicalDescription"):
            for y in x.findall("channelGroups"):
                for z in y.findall("group"):
                    chan_group = []
                    for chan in z.findall("channel"):
                        if int(chan.attrib["skip"]) == 1:
                            skipped_channels.append(int(chan.text))

                        chan_group.append(int(chan.text))
                    if chan_group:
                        channel_groups.append(np.array(chan_group))

        discarded_channels = np.setdiff1d(
            np.arange(n_channels), np.concatenate(channel_groups)
        )

        self.sig_dtype = nbits
        self.dat_sampling_rate = dat_sampling_rate
        self.eeg_sampling_rate = eeg_sampling_rate
        self.n_channels = n_channels
        self.channel_groups = np.array(channel_groups, dtype="object")
        self.discarded_channels = discarded_channels
        self.skipped_channels = np.array(skipped_channels)


    def _good_channels(self):
        good_chan = []
        for n in range(self.n_channels):
            if n not in self.discarded_channels and n not in self.skipped_channels:
                good_chan.append(n)

        self.good_channels = np.array(good_chan)


    def backup_xml_file(self, override_backup_path: Optional[Path]=None) -> Path:
        """Copy the on-disk neuroscope .xml to a backup path, leaving the source unchanged.

        Copies bytes from ``self.source_file`` (not the in-memory parsed state). Intended to be
        called before ``update_xml_file()``. By default writes ``{source}.xml.pre_edit.bak`` next
        to the source file and skips copying if that backup already exists. When
        ``override_backup_path`` is provided, always copies to that path (overwriting if needed).
        """
        source_path = Path(self.source_file).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Cannot backup missing neuroscope xml file: {source_path}")
        if override_backup_path is not None:
            backup_path = Path(override_backup_path).resolve()
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, backup_path)
            return backup_path
        backup_path = source_path.with_suffix(source_path.suffix + ".pre_edit.bak")
        if not backup_path.is_file():
            shutil.copy2(source_path, backup_path)
        return backup_path


    def _validate_channel_metadata(self) -> None:
        if self.channel_groups is None or self.skipped_channels is None or self.discarded_channels is None or self.n_channels is None:
            raise ValueError("Cannot update xml: channel metadata not initialized")
        source_path = Path(self.source_file).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Cannot update missing neuroscope xml file: {source_path}")
        channel_groups = [np.asarray(g, dtype=int).ravel() for g in self.channel_groups]
        grouped_channels = np.concatenate(channel_groups) if len(channel_groups) > 0 else np.array([], dtype=int)
        if grouped_channels.size > 0:
            if grouped_channels.size != len(np.unique(grouped_channels)):
                raise ValueError("Duplicate channel indices in channel_groups")
            if np.any(grouped_channels < 0) or np.any(grouped_channels >= self.n_channels):
                raise ValueError(f"channel_groups contains indices outside [0, {self.n_channels})")
        expected_discarded = np.setdiff1d(np.arange(self.n_channels), grouped_channels)
        discarded = np.asarray(self.discarded_channels, dtype=int).ravel()
        if not np.array_equal(np.sort(expected_discarded), np.sort(discarded)):
            raise ValueError("discarded_channels is inconsistent with channel_groups and n_channels")
        grouped_set = set(int(c) for c in grouped_channels)
        skipped_channels = np.asarray(self.skipped_channels, dtype=int).ravel()
        if not all(int(c) in grouped_set for c in skipped_channels):
            raise ValueError("skipped_channels contains indices not present in channel_groups")


    def update_xml_file(self) -> Path:
        """Inverse of ``_parse_xml_file``: persist in-memory channel metadata to the on-disk .xml file.

        Rebuilds ``anatomicalDescription/channelGroups`` from ``self.channel_groups`` and
        ``self.skipped_channels``. ``self.discarded_channels`` is validated but not written
        separately (it is derived from channels omitted from all groups). Call
        ``backup_xml_file()`` before editing. Does not modify ``acquisitionSystem``,
        ``fieldPotentials``, or ``spikeDetection``.

        Raises
        ------
        ValueError
            If channel metadata is missing, inconsistent, or the xml has no channel groups section.
        FileNotFoundError
            If ``self.source_file`` does not exist on disk.
        """
        self._validate_channel_metadata()
        source_path = Path(self.source_file).resolve()
        tree = Etree.parse(source_path)
        root = tree.getroot()
        channel_groups_elements = [cg for ad in root.findall("anatomicalDescription") for cg in ad.findall("channelGroups")]
        if not channel_groups_elements:
            raise ValueError(f"No anatomicalDescription/channelGroups in {source_path}")
        skipped_set = set(int(c) for c in np.asarray(self.skipped_channels).ravel())
        for cg_elem in channel_groups_elements:
            for group_elem in list(cg_elem.findall("group")):
                cg_elem.remove(group_elem)
        primary_channel_groups = channel_groups_elements[0]
        for chan_group in self.channel_groups:
            group_elem = Etree.SubElement(primary_channel_groups, "group")
            for chan_idx in np.asarray(chan_group, dtype=int).ravel():
                chan_elem = Etree.SubElement(group_elem, "channel")
                chan_elem.text = str(int(chan_idx))
                chan_elem.set("skip", "1" if int(chan_idx) in skipped_set else "0")
        tree.write(source_path)
        self._good_channels()
        return source_path


    def __str__(self) -> str:
        return (
            f"filename: {self.source_file} \n"
            f"# channels: {self.n_channels}\n"
            f"sampling rate: {self.dat_sampling_rate}\n"
            f"lfp Srate (downsampled): {self.eeg_sampling_rate}\n"
        )


    def set_datetime(self, datetime_epoch):
        """Often a resulting recording file is creating after concatenating different blocks.
        This method takes Epoch array containing datetime.
        """
        pass


    def write_neurons(self, neurons: core.Neurons):
        """To view spikes in neuroscope, spikes are exported to .clu.1 and .res.1 files in the basepath.
        You can order the spikes in a way to view sequential activity in neuroscope.

        Parameters
        ----------
        spks : list
            list of spike times.
        """

        spks = neurons.spiketrains
        srate = neurons.sampling_rate
        nclu = len(spks)
        spk_frame = np.concatenate([(cell * srate).astype(int) for cell in spks])
        clu_id = np.concatenate([[_] * len(spks[_]) for _ in range(nclu)])

        sort_ind = np.argsort(spk_frame)
        spk_frame = spk_frame[sort_ind]
        clu_id = clu_id[sort_ind]
        clu_id = np.append(nclu, clu_id)

        file_clu = self.source_file.with_suffix(".clu.1")
        file_res = self.source_file.with_suffix(".res.1")

        with file_clu.open("w") as f_clu, file_res.open("w") as f_res:
            for item in clu_id:
                f_clu.write(f"{item}\n")
            for frame in spk_frame:
                f_res.write(f"{frame}\n")


    def write_epochs(self, epochs: core.Epoch, ext=".epc"):
        with self.source_file.with_suffix(f".evt.{ext}").open("w") as a:
            for event in epochs.to_dataframe().itertuples():
                a.write(f"{event.start*1000} start\n{event.stop*1000} stop\n")


    def write_position(self, position: core.Position):
        # neuroscope only displays positive values so translating the coordinates
        x, y = position.x, position.y
        x = self.x + abs(min(self.x))
        y = self.y + abs(min(self.y))
        print(max(x))
        print(max(y))

        filename = self._obj.files.filePrefix.with_suffix(".pos")
        with filename.open("w") as f:
            for xpos, ypos in zip(x, y):
                f.write(f"{xpos} {ypos}\n")


    def to_dict(self, recurrsively=False):
        return {
            "source_file": self.source_file,
            "channel_groups": self.channel_groups,
            "skipped_channels": self.skipped_channels,
            "discarded_channels": self.discarded_channels,
            "n_channels": self.n_channels,
            "dat_sampling_rate": self.dat_sampling_rate,
            "eeg_sampling_rate": self.eeg_sampling_rate,
        }
