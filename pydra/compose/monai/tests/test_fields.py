"""Tests for pydra.compose.monai.fields.arg and .out."""
import typing as ty
from fileformats.medimage import NiftiGzX
from pydra.compose import monai


def test_arg_stores_path_kwarg():
    a = monai.arg(name="T1w", type=NiftiGzX, path="anat/T1w")
    assert a.path == "anat/T1w"


def test_arg_path_defaults_to_none():
    a = monai.arg(name="T1w", type=NiftiGzX)
    assert a.path is None


def test_arg_type_and_help_propagate():
    a = monai.arg(name="T1w", type=NiftiGzX, help="T1-weighted image")
    assert a.type is NiftiGzX
    assert a.help == "T1-weighted image"


def test_out_stores_path_kwarg():
    o = monai.out(name="mask", type=NiftiGzX, path="anat/mask")
    assert o.path == "anat/mask"


def test_out_path_defaults_to_none():
    o = monai.out(name="mask", type=NiftiGzX)
    assert o.path is None


def test_out_type_and_help_propagate():
    o = monai.out(name="mask", type=NiftiGzX, help="binary mask")
    assert o.type is NiftiGzX
    assert o.help == "binary mask"


def test_arg_accepts_any_type():
    a = monai.arg(name="weights", type=ty.Any)
    assert a.type is ty.Any
