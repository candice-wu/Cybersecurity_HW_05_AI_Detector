import pandas as pd
import numpy as np
import re
from string import punctuation

# 導入我們自己的 data_loader 模組
from .data_loader import load_data

def extract_features(text):
    """
    從單一文本中提取基礎統計特徵和風格特徵。

    Args:
        text (str): 輸入的文本字串。

    Returns:
        dict: 包含所有提取特徵的字典。
    """
    if not isinstance(text, str) or not text:
        return {}

    # 1. 基礎統計特徵
    words = re.findall(r'\b\w+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]

    num_chars = len(text)
    num_words = len(words)
    num_sentences = len(sentences)

    avg_word_length = np.mean([len(word) for word in words]) if num_words > 0 else 0
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if num_sentences > 0 else 0

    # 2. 風格特徵
    num_punctuations = sum(1 for char in text if char in punctuation)
    num_uppercase = sum(1 for char in text if char.isupper())
    num_digits = sum(1 for char in text if char.isdigit())
    
    # 比例特徵
    punc_ratio = num_punctuations / num_chars if num_chars > 0 else 0
    upper_ratio = num_uppercase / num_chars if num_chars > 0 else 0
    digit_ratio = num_digits / num_chars if num_chars > 0 else 0

    features = {
        'text_length': num_chars,
        'word_count': num_words,
        'sentence_count': num_sentences,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'punctuation_count': num_punctuations,
        'uppercase_count': num_uppercase,
        'digit_count': num_digits,
        'punctuation_ratio': punc_ratio,
        'uppercase_ratio': upper_ratio,
        'digit_ratio': digit_ratio,
    }
    return features

def apply_feature_extraction(df):
    """
    對 DataFrame 中的 'text' 欄位應用特徵提取。

    Args:
        df (pd.DataFrame): 包含 'text' 欄位的 DataFrame。

    Returns:
        pd.DataFrame: 附加了特徵欄位的 DataFrame。
    """
    features_df = df['text'].apply(lambda text: pd.Series(extract_features(text)))
    return pd.concat([df, features_df], axis=1)

if __name__ == '__main__':
    # 載入資料
    raw_df = load_data()
    
    if not raw_df.empty:
        # 提取特徵
        featured_df = apply_feature_extraction(raw_df)
        
        print("特徵提取完成。")
        print("\n資料範例 (包含特徵):")
        print(featured_df.head())
        
        print("\n特徵欄位:")
        print(featured_df.columns.tolist())
        
        # 顯示特定欄位的統計資訊，以比較 ai 和 human 的差異
        print("\n'平均詞長' 的分組統計:")
        print(featured_df.groupby('label')['avg_word_length'].describe())
        
        print("\n'大寫字母比例' 的分組統計:")
        print(featured_df.groupby('label')['uppercase_ratio'].describe())
    else:
        print("無法載入資料或資料為空。")
