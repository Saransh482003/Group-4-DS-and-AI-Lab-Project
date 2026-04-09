import pytest
from unittest.mock import patch, MagicMock
from app.utils.kaggle_data import ensure_egoblind_dataset
from app.config.pipeline_schema import KaggleSettings

@patch("app.utils.kaggle_data.check_kaggle_credentials", return_value=True)
@patch("app.utils.kaggle_data.subprocess.run")
@patch("app.utils.paths.get_base_dir")
def test_ensure_egoblind_download(mock_get_base_dir, mock_run, mock_check, tmp_path):
    mock_get_base_dir.return_value = tmp_path

    settings = KaggleSettings(
        dataset_slug="test/ds",
        local_cache_root="cache",
        extracted_folder_name="test_folder",
        auto_download_if_missing=True
    )

    # 1. First call should download
    res = ensure_egoblind_dataset(settings)
    assert mock_run.called
    assert res.endswith("extracted")

    # 2. Second call should cache hit
    mock_run.reset_mock()
    res2 = ensure_egoblind_dataset(settings)
    assert not mock_run.called
    assert res2 == res

@patch("app.utils.paths.get_base_dir")
def test_ensure_egoblind_no_auto_download(mock_get_base_dir, tmp_path):
    mock_get_base_dir.return_value = tmp_path

    settings = KaggleSettings(
        dataset_slug="test/ds",
        local_cache_root="cache",
        extracted_folder_name="test_folder",
        auto_download_if_missing=False
    )

    res = ensure_egoblind_dataset(settings)
    assert res == ""
