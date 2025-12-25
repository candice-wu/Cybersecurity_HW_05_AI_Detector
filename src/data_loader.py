import os
import re
import pandas as pd

def load_data(data_dir='data/raw'):
    """
    從指定的資料夾讀取 AI 生成和人類撰寫的文本資料。

    Args:
        data_dir (str): 包含原始文本檔案的目錄路徑。

    Returns:
        pd.DataFrame: 包含 'text' (文本內容) 和 'label' ('ai' 或 'human') 的 DataFrame。
    """
    data = []
    for filename in os.listdir(data_dir):
        if filename.endswith('.txt'):
            filepath = os.path.join(data_dir, filename)
            label = None
            if filename.startswith('ai_generated'):
                label = 'ai'
            elif filename.startswith('human_written'):
                label = 'human'
            
            if label:
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # 初步清理：去除多餘空白行和首尾空白
                    text = re.sub(r'\n\s*\n', '\n', text).strip()
                    data.append({'text': text, 'label': label})
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # 僅為測試用途，確保此模組能獨立運行
    df = load_data()
    print("載入資料範例:")
    print(df.head())
    print("\n資料分佈:")
    print(df['label'].value_counts())
    print(f"\n總資料筆數: {len(df)}")
