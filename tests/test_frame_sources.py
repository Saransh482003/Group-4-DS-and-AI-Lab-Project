import os
import shutil
import pytest
from app.sources.frame_folder_source import FrameFolderSource

@pytest.fixture
def temp_flat_folder(tmp_path):
    d = tmp_path / "flat"
    d.mkdir()
    (d / "image1.jpg").touch()
    (d / "image2.png").touch()
    (d / "not_image.txt").touch()
    return str(d)

@pytest.fixture
def temp_multi_folder(tmp_path):
    d = tmp_path / "multi"
    d.mkdir()
    seq1 = d / "seq1"
    seq1.mkdir()
    (seq1 / "img1.jpg").touch()
    seq2 = d / "seq2"
    seq2.mkdir()
    (seq2 / "img2.jpg").touch()
    (seq2 / "img3.jpg").touch()
    return str(d)

def test_frame_folder_flat(temp_flat_folder):
    source = FrameFolderSource(temp_flat_folder)
    assert source.open() == True
    assert len(source.image_files) == 2

    # Check proper ordering
    assert source.image_files[0][0] == "seq_0"
    assert source.image_files[0][1].endswith("image1.jpg")

def test_frame_folder_multi(temp_multi_folder):
    source = FrameFolderSource(temp_multi_folder)
    assert source.open() == True
    assert len(source.image_files) == 3

    # Seq1
    assert source.image_files[0][0] == "seq1"
    assert source.image_files[0][1].endswith("img1.jpg")

    # Seq2
    assert source.image_files[1][0] == "seq2"
    assert source.image_files[1][1].endswith("img2.jpg")
    assert source.image_files[2][0] == "seq2"
    assert source.image_files[2][1].endswith("img3.jpg")

def test_empty_folder(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    source = FrameFolderSource(str(d))
    assert source.open() == False
