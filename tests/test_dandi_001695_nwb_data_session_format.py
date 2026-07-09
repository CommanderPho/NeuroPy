from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import matplotlib.collections as matplotlib_collections


for alias_name, alias_value in {"bool": bool, "float": float, "int": int}.items():
    if alias_name not in np.__dict__:
        setattr(np, alias_name, alias_value)

if not hasattr(matplotlib_collections, "BrokenBarHCollection"):
    matplotlib_collections.BrokenBarHCollection = object


def test_dandi_nwb_001695_format_is_registered_and_loader_is_exposed():
    from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass
    from neuropy.core.session.data_session_loader import DataSessionLoader

    registered_formats = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    assert registered_formats["dandi_nwb_001695"] is DANDI001695NWBDataSessionFormatRegisteredClass
    assert hasattr(DataSessionLoader, "dandi_nwb_001695_session")


def test_dandi_nwb_001695_context_and_session_name_are_parsed_from_subject_folder(tmp_path):
    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

    basedir = tmp_path / "HighDensityCrossBrain" / "001695" / "sub-M01"
    basedir.mkdir(parents=True)
    nwb_filename = "sub-M01_ses-20240312T100000_behavior+ecephys.nwb"
    (basedir / nwb_filename).touch()

    context = DANDI001695NWBDataSessionFormatRegisteredClass.parse_session_basepath_to_context(basedir, nwb_filename=nwb_filename)

    assert DANDI001695NWBDataSessionFormatRegisteredClass.get_session_name(basedir) == "ses-20240312T100000"
    assert context.format_name == "dandi_nwb_001695"
    assert context.animal == "M01"
    assert context.exper_name == "001695"
    assert context.session_name == "ses-20240312T100000"


def test_find_nwb_file_uses_first_sorted_file_when_multiple_files_without_override(tmp_path):
    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

    basedir = tmp_path / "sub-M01"
    basedir.mkdir()
    first_file = basedir / "sub-M01_ses-20240308T100000_ecephys.nwb"
    second_file = basedir / "sub-M01_ses-20240312T100000_behavior+ecephys.nwb"
    first_file.touch()
    second_file.touch()

    with pytest.warns(UserWarning, match="Multiple NWB files"):
        assert DANDI001695NWBDataSessionFormatRegisteredClass.find_nwb_file(basedir) == first_file


def test_find_nwb_file_uses_override_when_multiple_files_exist(tmp_path):
    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

    basedir = tmp_path / "sub-M01"
    basedir.mkdir()
    first_file = basedir / "sub-M01_ses-20240308T100000_ecephys.nwb"
    second_file = basedir / "sub-M01_ses-20240312T100000_behavior+ecephys.nwb"
    first_file.touch()
    second_file.touch()

    assert DANDI001695NWBDataSessionFormatRegisteredClass.find_nwb_file(basedir, nwb_filename=second_file.name) == second_file


def test_dandi_nwb_001754_find_nwb_file_uses_first_sorted_file_when_multiple_files_without_override(tmp_path):
    from neuropy.core.session.Formats.Specific.DANDI001754NWBDataSessionFormat import DANDI001754NWBDataSessionFormatRegisteredClass

    basedir = tmp_path / "sub-Rat1"
    basedir.mkdir()
    first_file = basedir / "sub-Rat1_ses-19980413T163700_ecephys.nwb"
    second_file = basedir / "sub-Rat1_ses-19980425T124500_behavior+ecephys.nwb"
    first_file.touch()
    second_file.touch()

    with pytest.warns(UserWarning, match="Multiple NWB files"):
        assert DANDI001754NWBDataSessionFormatRegisteredClass.find_nwb_file(basedir) == first_file


def test_wake_epoch_labels_overlap_position_filters_to_epochs_with_position():
    from types import SimpleNamespace

    from neuropy.core.epoch import Epoch
    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

    epochs_df = pd.DataFrame({
        'start': [1.0, 4335.0],
        'stop': [2204.0, 8270.0],
        'label': ['WAKE0', 'WAKE3'],
        'behavior': ['wake', 'wake'],
        'duration': [2203.0, 3935.0],
    })
    sess = SimpleNamespace(epochs=Epoch(epochs_df), position=SimpleNamespace(time=np.linspace(4357.0, 6261.0, 10)))
    activity_labels = DANDI001695NWBDataSessionFormatRegisteredClass._get_activity_epoch_labels(sess)

    assert activity_labels == ['WAKE3']


def test_get_ecephys_t0_uses_minimum_spike_time():
    from types import SimpleNamespace

    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

    units_df = pd.DataFrame({'spike_times': [[5.0, 6.0], [0.25, 1.0]]})
    nwbf = SimpleNamespace(units=SimpleNamespace(to_dataframe=lambda: units_df))

    assert DANDI001695NWBDataSessionFormatRegisteredClass._get_ecephys_t0(nwbf) == 0.25


def test_sleep_states_paradigm_conversion():
    from types import SimpleNamespace

    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

    sleep_df = pd.DataFrame({
        'start_time': [0.0, 10.0, 20.0, 30.0],
        'stop_time': [10.0, 20.0, 30.0, 40.0],
        'state': ['WAKE', 'NREM', 'REM', 'Ripple'],
    })
    intervals = SimpleNamespace(SleepStates=SimpleNamespace(to_dataframe=lambda: sleep_df))
    nwbf = SimpleNamespace(intervals=intervals)

    paradigm = DANDI001695NWBDataSessionFormatRegisteredClass._load_paradigm_from_nwb(nwbf, t0=0.0, epoch_label_mode='sleep_states')
    result_df = paradigm.to_dataframe()

    assert result_df['label'].tolist() == ['WAKE0', 'NREM0', 'REM0', 'Ripple0']
    assert result_df['behavior'].tolist() == ['wake', 'nrem', 'rem', 'ripple']


def test_load_neurons_filters_by_cell_area():
    from types import SimpleNamespace

    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

    units_df = pd.DataFrame({
        'spike_times': [[1.0, 2.0], [3.0], [4.0, 5.0, 6.0]],
        'cell_area': ['CA1', 'CA3', 'CA1'],
        'cell_type': ['Pyramidal Cell', 'Pyramidal Cell', 'Narrow Interneuron'],
    }, index=[0, 1, 2])
    nwbf = SimpleNamespace(units=SimpleNamespace(to_dataframe=lambda: units_df))

    neurons = DANDI001695NWBDataSessionFormatRegisteredClass._load_neurons_from_nwb(nwbf, t0=0.0, t_stop=10.0, unit_location_filter='CA1')

    assert len(neurons) == 2
    assert neurons.neuron_ids.tolist() == [0, 2]


@pytest.mark.skipif(not Path(r'H:\Data\DANDI\HighDensityCrossBrain\001695\sub-M01').is_dir(), reason='DANDI 001695 data not available locally')
def test_live_load_behavior_ecephys_session_smoke():
    from neuropy.core.session.Formats.Specific.DANDI001695NWBDataSessionFormat import DANDI001695NWBDataSessionFormatRegisteredClass

    basedir = Path(r'H:\Data\DANDI\HighDensityCrossBrain\001695\sub-M01')
    sess = DANDI001695NWBDataSessionFormatRegisteredClass.get_session(
        basedir=basedir,
        override_parameters_flat_keypaths_dict={'nwb.nwb_filename': 'sub-M01_ses-20240312T100000_behavior+ecephys.nwb', 'nwb.export_root': str(basedir.parent.parent / 'export')},
    )
    assert sess.neurons.n_neurons > 0
    assert len(sess.position.time) > 0
    assert 'WAKE0' in sess.epochs.get_unique_labels()
    activity_epoch_labels = DANDI001695NWBDataSessionFormatRegisteredClass._get_activity_epoch_labels(sess)
    assert len(activity_epoch_labels) > 0
    assert 'WAKE3' in activity_epoch_labels
    assert float(sess.position.time[0]) > 100.0
    spike_times = sess.neurons.spiketrains[0]
    assert float(np.min(spike_times)) >= 0.0
