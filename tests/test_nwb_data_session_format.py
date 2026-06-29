from pathlib import Path

import numpy as np
import pytest
import matplotlib.collections as matplotlib_collections


for alias_name, alias_value in {"bool": bool, "float": float, "int": int}.items():
    if alias_name not in np.__dict__:
        setattr(np, alias_name, alias_value)

if not hasattr(matplotlib_collections, "BrokenBarHCollection"):
    matplotlib_collections.BrokenBarHCollection = object


def test_dandi_nwb_format_is_registered_and_loader_is_exposed():
    from neuropy.core.session.Formats.BaseDataSessionFormats import DataSessionFormatRegistryHolder
    from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass
    from neuropy.core.session.data_session_loader import DataSessionLoader

    registered_formats = DataSessionFormatRegistryHolder.get_registry_data_session_type_class_name_dict()

    assert registered_formats["dandi_nwb"] is NWBDataSessionFormatRegisteredClass
    assert hasattr(DataSessionLoader, "dandi_nwb_session")


def test_dandi_nwb_context_and_session_name_are_parsed_from_subject_folder(tmp_path):
    from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass

    basedir = tmp_path / "download" / "000978" / "sub-JDS-SingleDay-ER1"
    basedir.mkdir(parents=True)

    context = NWBDataSessionFormatRegisteredClass.parse_session_basepath_to_context(basedir)

    assert NWBDataSessionFormatRegisteredClass.get_session_name(basedir) == "ER1_SingleDay"
    assert context.format_name == "dandi_nwb"
    assert context.animal == "ER1"
    assert context.exper_name == "000978"
    assert context.session_name == "SingleDay"


def test_find_nwb_file_uses_override_when_multiple_files_exist(tmp_path):
    from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass

    basedir = tmp_path / "sub-JDS-SingleDay-ZT2"
    basedir.mkdir()
    first_file = basedir / "sub-JDS-SingleDay-ZT2_obj-1dss6zi_behavior+ecephys.nwb"
    second_file = basedir / "sub-JDS-SingleDay-ZT2_obj-u40err_behavior+ecephys.nwb"
    first_file.touch()
    second_file.touch()

    assert NWBDataSessionFormatRegisteredClass.find_nwb_file(basedir, nwb_filename=second_file.name) == second_file


def test_find_nwb_file_requires_override_filename_to_exist(tmp_path):
    from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass

    basedir = tmp_path / "sub-JDS-SingleDay-ZT2"
    basedir.mkdir()
    (basedir / "sub-JDS-SingleDay-ZT2_obj-1dss6zi_behavior+ecephys.nwb").touch()

    with pytest.raises(FileNotFoundError):
        NWBDataSessionFormatRegisteredClass.find_nwb_file(basedir, nwb_filename="missing.nwb")


def test_nwb_track_graph_linearization_on_synthetic_w_track_position(tmp_path):
    pytest.importorskip("track_linearization")

    import pandas as pd
    from types import SimpleNamespace

    from neuropy.core import Epoch, Position
    from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass

    n_samples = 100
    t = np.linspace(0.0, 10.0, n_samples)
    x = np.full(n_samples, 100.0)
    y = np.linspace(30.0, 90.0, n_samples)
    position = Position.from_separate_arrays(t, x, y=y)

    epochs_df = pd.DataFrame({'start': [0.0, 5.0], 'stop': [5.0, 10.0], 'label': ['maze0', 'sleep0'], 'behavior': ['maze', 'sleep'], 'duration': [5.0, 5.0]})
    epochs = Epoch(epochs_df)

    basedir = tmp_path / "download" / "000978" / "sub-JDS-SingleDay-ER1"
    basedir.mkdir(parents=True)
    context = NWBDataSessionFormatRegisteredClass.parse_session_basepath_to_context(basedir)
    preprocessing_parameters = NWBDataSessionFormatRegisteredClass.build_default_preprocessing_parameters()
    config = SimpleNamespace(preprocessing_parameters=preprocessing_parameters, get_context=lambda: context)

    file_prefix = tmp_path / "export" / "000978" / "ER1" / "ER1_SingleDay"
    file_prefix.parent.mkdir(parents=True, exist_ok=True)

    session = SimpleNamespace(position=position, epochs=epochs, paradigm=epochs, config=config, basepath=basedir, filePrefix=file_prefix, get_context=lambda: context)

    NWBDataSessionFormatRegisteredClass._compute_linear_position_if_possible(session)

    pos_df = session.position.to_dataframe()
    maze_mask = (pos_df['t'] >= 0.0) & (pos_df['t'] <= 5.0)
    sleep_mask = (pos_df['t'] > 5.0) & (pos_df['t'] <= 10.0)

    assert pos_df.loc[maze_mask, 'lin_pos'].notna().all()
    assert pos_df.loc[sleep_mask, 'lin_pos'].isna().all()
    assert 'track_segment_id' in pos_df.columns
    assert pos_df.loc[maze_mask, 'linearization_method'].eq('track_graph').all()


def test_linearize_position_df_track_graph_accepts_track_definition_key():
    pytest.importorskip("track_linearization")

    import pandas as pd

    from neuropy.utils import position_util

    pos_df = pd.DataFrame({'t': np.linspace(0.0, 1.0, 20), 'x': np.full(20, 100.0), 'y': np.linspace(30.0, 90.0, 20)})
    out_df = position_util.linearize_position_df(pos_df, method='track_graph', track_definition='w_maze')

    assert out_df['lin_pos'].notna().all()
    assert 'track_segment_id' in out_df.columns
    assert out_df['linearization_method'].eq('track_graph').all()


def test_position_needs_track_graph_recompute_detects_stale_isomap_cache(tmp_path):
    pytest.importorskip("track_linearization")

    import pandas as pd
    from types import SimpleNamespace

    from neuropy.core import Epoch, Position
    from neuropy.core.session.Formats.Specific.NWBDataSessionFormat import NWBDataSessionFormatRegisteredClass

    t = np.linspace(0.0, 1.0, 10)
    position = Position.from_separate_arrays(t, np.linspace(40.0, 150.0, 10), y=np.linspace(10.0, 120.0, 10), lin_pos=np.linspace(0.0, 100.0, 10))
    epochs = Epoch(pd.DataFrame({'start': [0.0], 'stop': [1.0], 'label': ['maze0'], 'behavior': ['maze'], 'duration': [1.0]}))

    basedir = tmp_path / "download" / "000978" / "sub-JDS-SingleDay-ER1"
    basedir.mkdir(parents=True)
    context = NWBDataSessionFormatRegisteredClass.parse_session_basepath_to_context(basedir)
    preprocessing_parameters = NWBDataSessionFormatRegisteredClass.build_default_preprocessing_parameters()
    config = SimpleNamespace(preprocessing_parameters=preprocessing_parameters, get_context=lambda: context)
    session = SimpleNamespace(position=position, epochs=epochs, paradigm=epochs, config=config, basepath=basedir, filePrefix=tmp_path / "ER1_SingleDay", get_context=lambda: context)

    assert NWBDataSessionFormatRegisteredClass._position_needs_track_graph_recompute(session) is True
