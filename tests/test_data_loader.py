import os
import pytest
import pandas as pd
import sys

# 將 src 目錄加入到 sys.path 中，以便 pytest 可以找到 data_loader 模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_loader import load_data

@pytest.fixture
def mock_data_dir(tmpdir):
    """建立一個 mock 的資料目錄，並在其中創建測試檔案。"""
    raw_dir = tmpdir.mkdir("raw")
    raw_dir.join("ai_generated_test_01.txt").write("This is AI text.")
    raw_dir.join("human_written_test_01.txt").write("This is human text.")
    raw_dir.join("human_written_test_02.txt").write("Another human text.")
    raw_dir.join("ignored_file.md").write("This file should be ignored.")
    return str(raw_dir)

def test_load_data_returns_dataframe(mock_data_dir):
    """測試 load_data 是否返回一個 Pandas DataFrame。"""
    df = load_data(mock_data_dir)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_load_data_columns(mock_data_dir):
    """測試返回的 DataFrame 是否包含 'text' 和 'label' 欄位。"""
    df = load_data(mock_data_dir)
    assert 'text' in df.columns
    assert 'label' in df.columns

def test_load_data_labels(mock_data_dir):
    """測試 'label' 欄位是否只包含 'ai' 和 'human'。"""
    df = load_data(mock_data_dir)
    unique_labels = df['label'].unique()
    assert set(unique_labels) == {'ai', 'human'}

def test_load_data_content(mock_data_dir):
    """測試載入的資料筆數和內容是否正確。"""
    df = load_data(mock_data_dir)
    assert len(df) == 3
    assert df[df['label'] == 'ai'].shape[0] == 1
    assert df[df['label'] == 'human'].shape[0] == 2

def test_load_data_empty_dir(tmpdir):
    """測試當目錄為空時，是否返回一個空的 DataFrame。"""
    empty_dir = tmpdir.mkdir("empty_raw")
    df = load_data(str(empty_dir))
    assert isinstance(df, pd.DataFrame)
    assert df.empty
