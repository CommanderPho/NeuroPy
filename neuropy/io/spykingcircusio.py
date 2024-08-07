import numpy as np
import h5py
from ..core import Epoch


class SpykingCircusIO:
    def __init__(self) -> None:
        pass

    def load_rough_mua(self, mua_filename=None):
        """Load in multi-unit activity (MUA) generated by running spyking-circus with the 'thresholding' flag.
        Note that this differs from MUA designated in phy after manual spike sorting (q = 6),
        so it is defined as Spikes.rough_mua"""

        # First grab the mua file if not specified.
        if mua_filename is None:
            mua_filename = list(self._obj.basePath.glob("**/*.mua.hdf5"))
            assert (
                len(mua_filename) == 1
            ), "More than one .mua.hdf5 file found in directory tree. Re-organize and try again!"

        # Now load it in
        self.rough_mua = h5py.File(mua_filename[0], "r+")

    def roughmua2neuroscope(self, chans, shankIDs):
        """Exports all threshold crossings on electrodes entered to view in neuroscope to .clu.shankID and .res.shankID
        files for viewing in neuroscope.  Must run Spikes.load_mua first. Cluster ids correspond to electrodes.
        Chans = channel numbers in neuroscope."""
        assert len(chans) == len(
            shankIDs
        ), "electrodes and shankIDs must be the same length"

        # re-order/name the channels to match that in the .mua.hdf5 file
        chans_reorder = np.asarray(
            [np.where(chan == self._obj.goodchans)[0][0] for chan in chans]
        )
        for shank in np.unique(shankIDs):

            # Build up array of mua threshold crossings
            spikes, chanIDs = [], []
            for mua_chan, chan in zip(
                chans_reorder[shankIDs == shank], np.asarray(chans)[shankIDs == shank]
            ):

                spikes.extend(self.rough_mua["spiketimes"]["elec_" + str(mua_chan)][:])
                chanIDs.extend(
                    chan
                    * np.ones_like(
                        self.rough_mua["spiketimes"]["elec_" + str(mua_chan)[:]]
                    )
                )

            # sort by spike-times
            sort_ids = np.asarray(spikes).argsort()
            spikes_sorted = np.asarray(spikes)[sort_ids]
            nclu = len(chans)
            chanIDs_sorted = np.append(nclu, np.asarray(chanIDs)[sort_ids])

            # Now write to clu and res files
            mua_filePrefix = self._obj.basePath / (
                self._obj.files.filePrefix.name + "_mua"
            )
            file_clu = mua_filePrefix.with_suffix(".clu." + str(shank))
            file_res = mua_filePrefix.with_suffix(".res." + str(shank))
            with file_clu.open("w") as f_clu, file_res.open("w") as f_res:
                for item in chanIDs_sorted:
                    f_clu.write(f"{item}\n")
                for frame in spikes_sorted:
                    f_res.write(f"{frame}\n")

    @staticmethod
    def write_epochs(file, epochs: Epoch):
        with file.open("w") as a:
            for event in epochs.to_dataframe().itertuples():
                a.write(f"{event.start*1000} {event.stop*1000}\n")

        print(f"{file.name} created")
