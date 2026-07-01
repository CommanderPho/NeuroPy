from __future__ import annotations

import numpy as np
from pathlib import Path
import pandas as pd
from typing import List, Optional, Sequence


class PhyIO:
    """ handles the output of the software Phy, which is used to provide an interactive GUI to manually curate cells for the spike-sorting step
    """
    def __init__(self, dirname: Path, include_groups=("mua", "good")) -> None:
        self._source_dir = dirname
        self._sampling_rate: Optional[int] = None
        self._spiketrains: Optional[np.ndarray] = None
        self._waveforms: Optional[np.ndarray] = None
        self._peak_waveforms: Optional[List[np.ndarray]] = None
        self._peak_channels: Optional[np.ndarray] = None
        self._n_channels: Optional[int] = None
        self._n_features_per_channel: Optional[int] = None
        self._cluster_info: Optional[pd.DataFrame] = None
        self._amplitudes: Optional[np.ndarray] = None
        self._shank_ids: Optional[np.ndarray] = None
        self._include_groups = include_groups
        self._parse_folder()


    @property
    def source_dir(self) -> Path:
        return self._source_dir


    @property
    def sampling_rate(self) -> Optional[int]:
        return self._sampling_rate


    @property
    def spiketrains(self) -> Optional[np.ndarray]:
        return self._spiketrains


    @property
    def waveforms(self) -> Optional[np.ndarray]:
        return self._waveforms


    @property
    def peak_waveforms(self) -> Optional[List[np.ndarray]]:
        return self._peak_waveforms


    @property
    def peak_channels(self) -> Optional[np.ndarray]:
        return self._peak_channels


    @property
    def n_channels(self) -> Optional[int]:
        return self._n_channels


    @property
    def n_features_per_channel(self) -> Optional[int]:
        return self._n_features_per_channel


    @property
    def cluster_info(self) -> Optional[pd.DataFrame]:
        return self._cluster_info


    @property
    def amplitudes(self) -> Optional[np.ndarray]:
        return self._amplitudes


    @property
    def shank_ids(self) -> Optional[np.ndarray]:
        return self._shank_ids


    @property
    def include_groups(self) -> Sequence[str]:
        return self._include_groups


    def _parse_folder(self):
        params = {}
        with (self._source_dir / "params.py").open("r") as f:
            for line in f:
                line_values = (
                    line.replace("\n", "")
                    .replace('r"', '"')
                    .replace('"', "")
                    .split("=")
                )
                params[line_values[0].strip()] = line_values[1].strip()

        self._sampling_rate = int(float(params["sample_rate"]))
        self._n_channels = int(params["n_channels_dat"])
        if "n_features_per_channel" in params:
            self._n_features_per_channel = int(params["n_features_per_channel"])
        elif (self._source_dir / "pc_features.npy").is_file():
            self._n_features_per_channel = int(np.load(self._source_dir / "pc_features.npy", mmap_mode="r").shape[1])
        else:
            self._n_features_per_channel = None

        spktime = np.load(self._source_dir / "spike_times.npy")
        clu_ids = np.load(self._source_dir / "spike_clusters.npy")
        spk_templates_id = np.load(self._source_dir / "spike_templates.npy")
        spk_templates = np.load(self._source_dir / "templates.npy")
        cluinfo = pd.read_csv(self._source_dir / "cluster_info.tsv", delimiter="\t")
        similarity = np.load(self._source_dir / "similar_templates.npy")

        # if self.include_noise_clusters:
        #     cluinfo = cluinfo[
        #         cluinfo["group"].isin(["mua", "good", "noise"])
        #     ].reset_index(drop=True)
        # else:
        #     cluinfo = cluinfo[cluinfo["group"].isin(["mua", "good"])].reset_index(
        #         drop=True
        #     )

        cluinfo = cluinfo[cluinfo["group"].isin(self._include_groups)].reset_index(
            drop=True
        )
        if "id" not in cluinfo:
            print(
                "WARN: id column does not exist in cluster_info.tsv. Using cluster_id column instead."
            )
            cluinfo["id"] = cluinfo["cluster_id"]

        self._cluster_info = cluinfo.copy()
        self._amplitudes = np.asarray(cluinfo["amp"].values)
        self._peak_channels = np.asarray(cluinfo["ch"].values)
        self._shank_ids = np.asarray(cluinfo["sh"].values)

        if not self._cluster_info.empty:
            spiketrains, template_waveforms = [], []
            for clu in cluinfo.itertuples():
                clu_spike_location = np.where(clu_ids == clu.id)[0]
                spkframes = spktime[clu_spike_location]
                cell_template_id, counts = np.unique(
                    spk_templates_id[clu_spike_location], return_counts=True
                )
                spiketrains.append(spkframes / self._sampling_rate)
                template_waveforms.append(
                    spk_templates[cell_template_id[np.argmax(counts)]].squeeze().T
                )

            self._spiketrains = np.array(spiketrains, dtype="object")
            self._waveforms = np.array(template_waveforms)
            self._peak_waveforms = [
                wav[np.argmax(np.max(wav, axis=1))] for wav in template_waveforms
            ]


    def __setstate__(self, state):
        legacy_to_private = (
            ("source_dir", "_source_dir"),
            ("sampling_rate", "_sampling_rate"),
            ("spiketrains", "_spiketrains"),
            ("waveforms", "_waveforms"),
            ("peak_waveforms", "_peak_waveforms"),
            ("peak_channels", "_peak_channels"),
            ("n_channels", "_n_channels"),
            ("n_features_per_channel", "_n_features_per_channel"),
            ("cluster_info", "_cluster_info"),
            ("amplitudes", "_amplitudes"),
            ("shank_ids", "_shank_ids"),
            ("include_groups", "_include_groups"),
        )
        for legacy_key, private_key in legacy_to_private:
            if legacy_key in state and private_key not in state:
                state[private_key] = state.pop(legacy_key)
        self.__dict__.update(state)
