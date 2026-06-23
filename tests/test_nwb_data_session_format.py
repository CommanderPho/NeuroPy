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
