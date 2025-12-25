import os
import pytest
import pandas as pd
import numpy as np
import sys

# 將 src 目錄加入到 sys.path 中，以便 pytest 可以找到 feature_engineering 模組
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from feature_engineering import extract_features, apply_feature_extraction

@pytest.fixture
def mock_dataframe():
    """建立一個 mock 的 DataFrame 用於測試。"""
    data = {
        'text': [
            "This is the first sentence. This is the second one!",
            "Another sentence, with numbers 123 and UPPERCASE.",
            "",
            None
        ],
        'label': ['human', 'ai', 'human', 'ai']
    }
    return pd.DataFrame(data)

def test_extract_features_on_valid_text():
    """測試 extract_features 對有效文本的處理。"""
    text = "Hello world. This is a test."
    features = extract_features(text)
    
    # 檢查是否返回字典
    assert isinstance(features, dict)
    
    # 檢查是否包含所有預期的特徵鍵
    expected_keys = [
        'text_length', 'word_count', 'sentence_count', 'avg_word_length',
        'avg_sentence_length', 'punctuation_count', 'uppercase_count',
        'digit_count', 'punctuation_ratio', 'uppercase_ratio', 'digit_ratio'
    ]
    assert all(key in features for key in expected_keys)
    
    # 檢查幾個具體的值
    assert features['text_length'] == 28
    assert features['word_count'] == 6
    assert features['sentence_count'] == 2
    assert features['punctuation_count'] == 2
    assert features['uppercase_count'] == 2

def test_extract_features_on_empty_string():
    """測試 extract_features 對空字串的處理。"""
    text = ""
    features = extract_features(text)
    assert features == {}

def test_extract_features_on_none():
    """測試 extract_features 對 None 的處理。"""
    text = None
    features = extract_features(text)
    assert features == {}

def test_apply_feature_extraction(mock_dataframe):
    """測試 apply_feature_extraction 是否能成功新增特徵欄位。"""
    # 移除包含 None 的行，因為 apply 無法處理
    df = mock_dataframe.dropna(subset=['text']).copy()
    
    featured_df = apply_feature_extraction(df)
    
    # 檢查是否返回 DataFrame
    assert isinstance(featured_df, pd.DataFrame)
    
    # 檢查欄位數量是否正確
    original_cols = len(df.columns)
    expected_feature_cols = 11  # 我們提取了 11 個特徵
    assert len(featured_df.columns) == original_cols + expected_feature_cols
    
    # 檢查第一行資料的某個特徵值是否正確
    # "This is the first sentence. This is the second one!"
    assert featured_df.loc[0, 'word_count'] == 10
    assert featured_df.loc[0, 'sentence_count'] == 2
