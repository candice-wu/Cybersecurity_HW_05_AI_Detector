# 📄 AI 文本偵測器 (AI Text Detector)

本專案旨在開發一個能夠區分由人工智慧（AI）生成與人類撰寫文本的應用程式。應用程式基於機器學習模型，並透過 Streamlit 提供一個互動式的 Web 介面，讓使用者可以輸入文本進行即時分析，並以視覺化的方式呈現分析結果與模型效能。

**[前往 Streamlit Demo 網站](https://hw05-ai-detector.streamlit.app)**

---

## 💡 主要功能亮點

-   **AI 文本檢測**: 偵測輸入文本是由 AI 生成還是人類撰寫，並顯示 AI 生成機率（透過儀表板圖）。
-   **互動式 Web 介面**: 使用 Streamlit 提供使用者友善的圖形介面，支援文本輸入和檔案上傳。
-   **模型效能儀表板**: 視覺化呈現當前模型的效能評估指標（準確率、分類報告、混淆矩陣）及特徵重要性。
-   **模型訓練功能**: 支援在應用程式側邊欄中重新訓練模型，以便使用最新的資料進行模型更新。
-   **詳細分析與解釋**: 提供判定原因的簡要說明（基於統計特徵）及系統性能指標（處理時間、分析字數、處理速度），增強透明度。
-   **系統資訊顯示**: 側邊欄顯示開發者、Python 環境及關鍵函式庫版本，便於追蹤與維護。

---

## 🥅 CRISP-DM 專案框架

本專案遵循 CRISP-DM (跨產業數據探勘標準流程) 框架進行開發，其生命週期包含以下六個階段：

### 1. 商業理解 (Business Understanding)
*   **背景**: 隨著大型語言模型（LLM）的普及，AI 生成內容已無處不在。雖然這帶來了效率，但也引發了對內容真實性的擔憂，例如在學術誠信、新聞報導、內容審核等領域。
*   **目標**: 開發一個使用者友好的工具，能夠快速、準確地判斷一段文本是由 AI 生成還是人類撰寫，為使用者提供一個可靠的參考依據。

### 2. 資料理解 (Data Understanding)
*   **資料來源**: 專案初期的訓練資料位於 `data/raw/` 目錄下，包含開發者提供的少量 AI 生成與人類撰寫的 `.txt` 示範檔案。
*   **資料特性**: 資料被分為 `ai` 和 `human` 兩個類別。目前資料量有限，主要用於驗證開發流程。為了建構一個強健的模型，未來需要從更多元的來源（如新聞、論文、小說、社群媒體）擴充真實世界的資料。

### 3. 資料準備 (Data Preparation)
*   **資料載入與清理**: `src/data_loader.py` 模組負責從 `data/raw/` 目錄讀取所有 `.txt` 檔案。它會根據檔名 (`ai_generated_*`, `human_written_*`) 自動為每筆資料標記類別，並進行初步的文字清理（例如去除多餘的空白字元）。
*   **特徵工程**: `src/feature_engineering.py` 模組是本專案目前的核心。它採用「自建特徵法」，從文本中提取以下維度的統計與風格特徵：
    ```json
    [
        "text_length", "word_count", "sentence_count", 
        "avg_word_length", "avg_sentence_length", 
        "punctuation_count", "uppercase_count", "digit_count",
        "punctuation_ratio", "uppercase_ratio", "digit_ratio"
    ]
    ```

### 4. 模型建立 (Modeling)
*   **模型選擇**: 本專案目前採用基於「自建特徵」的傳統監督式學習方法。我們選用 `scikit-learn` 函式庫中的 `RandomForestClassifier`（隨機森林分類器）作為核心演算法，因其在處理表格型特徵時具有良好的效能與解釋性。
*   **模型訓練**: `src/model.py` 模組封裝了完整的模型訓練流程，包括資料分割、模型訓練、儲存與載入。訓練好的模型會被序列化並儲存為 `model.joblib` 檔案。

### 5. 模型評估 (Evaluation)
*   **評估指標**: 我們使用一套標準的分類模型評估指標來衡量模型的效能，包括：準確率 (Accuracy)、精確率 (Precision)、召回率 (Recall)、F1 分數 (F1-score) 以及混淆矩陣 (Confusion Matrix)。
*   **結果呈現**: 模型的整體效能評估結果會被視覺化，並顯示在 Streamlit 應用程式的「模型儀表板」中，方便使用者了解當前模型的表現。

### 6. 系統部署 (Deployment)
*   **應用介面**: 使用 Streamlit 框架開發，主程式為 `5114050013_hw5.py`。它提供了一個互動式的 Web 介面，讓使用者可以輕易地進行文本分析、查看結果與管理模型。
*   **部署計畫**: 根據 `tasks.md` 的規劃，未來可將此應用程式部署至 Streamlit Cloud 或其他雲端平台，提供公開的線上服務。

---

## 🛠️ 技術棧

-   **程式語言**: Python (核心開發語言)
-   **網頁框架**: Streamlit (用於構建互動式 Web UI)
-   **機器學習**: scikit-learn (用於模型訓練、評估與預測，核心模型為 RandomForestClassifier)
-   **數據處理**: Pandas, NumPy (用於數據載入、清洗和特徵處理)
-   **自然語言處理**: (目前主要透過自建特徵，未來可引入 NLTK, spaCy 以擴展語言學分析)
-   **數據視覺化**: Plotly (用於儀表板圖、混淆矩陣), Altair (用於句長分佈等靜態圖表)
-   **模型序列化**: Joblib (用於模型儲存與載入)
-   **程式碼規範**: Black, isort (用於自動化程式碼格式化與導入排序)
-   **單元測試**: Pytest (用於確保程式碼的正確性與穩定性)

---

## 🎡 專案架構

```
.
├── 5114050013_hw5.py         # Streamlit 應用程式主程式，負責 UI 邏輯與功能整合
├── data/
│   ├── raw/                  # 原始資料集，存放 AI 生成與人類撰寫的文本檔案
│   └── processed/            # (預留) 存放處理後的資料
├── src/                      # 核心功能模組
│   ├── data_loader.py        # 負責資料載入與初步清理
│   ├── feature_engineering.py# 負責文本特徵提取
│   ├── model.py              # 負責模型訓練與預測
│   └── visualization.py      # 負責所有視覺化圖表的繪製
├── tests/                    # 單元測試檔案
│   ├── test_data_loader.py   # data_loader 的單元測試
│   └── test_feature_engineering.py # feature_engineering 的單元測試
├── model.joblib              # 訓練完成後儲存的模型檔案 (被 .gitignore 忽略)
├── requirements.txt          # 專案依賴套件清單
└── README.md                 # 專案說明文件
```

---

## 系統部署 (Deployment)

### 🚀 本地端執行 (Local Deployment)

以下步驟指導您如何在本地環境中啟動並運行 AI 文本偵測器：

1.  **複製專案儲存庫**:
    ```bash
    git clone https://github.com/candice-wu/Cybersecurity_HW_05_AI_Detector.git
    cd Cybersecurity_HW_05_AI_Detector
    ```

2.  **建立並啟用 Python 虛擬環境**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # 或在 Windows 上: venv\Scripts\activate
    ```

3.  **安裝專案依賴套件**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **產生模型檔案**:
    *   **重要提示**: 在啟動應用程式之前，您需要先在本機端執行一次模型訓練，以產生 `model.joblib` 模型檔案。請在終端機中執行以下指令：
    ```bash
    python -m src.model
    ```

5.  **啟動 Streamlit 應用程式**:
    ```bash
    streamlit run 5114050013_hw5.py
    ```
    應用程式將在您的預設瀏覽器中開啟。

### ☁️ 雲端部署 (Cloud Deployment)

在將此專案推送到您的遠端 GitHub 儲存庫後，您可以依照以下步驟將其免費部署為一個公開的 Web 應用程式：

1.  **登入 Streamlit Community Cloud**
    前往 [share.streamlit.io](https://share.streamlit.io/) 並使用您的 GitHub 帳號登入。

2.  **建立新應用程式**
    點擊頁面右上方的「New app」按鈕。

3.  **連結您的儲存庫**
    - **Repository**：選擇您存放此專案的 GitHub 儲存庫。
    - **Branch**：選擇您要部署的分支（例如 `main` 或 `master`）。
    - **Main file path**：確認應用程式的主檔案路徑為 `5114050013_hw5.py`。

4.  **部署！**
    點擊「Deploy!」。
    Streamlit 將會開始建置您的應用程式，這可能需要幾分鐘的時間。

5.  **取得應用程式網址**
    部署成功後，您會得到一個公開的網址。請使用此網址更新本文件最上方的「前往 Streamlit Demo 網站」連結。

---

## 🏜️ 未來可改善計畫

-   **模型升級**: 引入 `transformers` 函式庫，使用更先進的預訓練語言模型（如 RoBERTa, BERT, DeBERTa）來進行偵測。這將從依賴統計特徵轉變為利用深度語意理解，有望大幅提升模型的準確度和泛化能力。
-   **特徵擴充**: 在現有基礎上，增加更豐富的語言學特徵，例如：
    *   **詞彙豐富度 (TTR - Type-Token Ratio)**：識別 AI 生成文本可能用詞較為單一的模式。
    *   **語法複雜度**: 透過句法樹分析等方法，評估文本的語法結構複雜度。
    *   **困惑度 (Perplexity)**：使用一個標準的語言模型來評估文本的流暢度與自然度。
-   **資料集增強**: 建立一個可持續擴充、多樣化且標記精確的資料集，並研究資料增強（Data Augmentation）技術，以產生更多樣化的訓練樣本，提升模型對不同風格文本的適應性。
-   **使用者回饋機制**: 在介面上新增一個功能，允許使用者對模型的預測結果進行回饋（例如提供「判斷錯誤」按鈕）。這些回饋資料可以被收集起來，作為未來模型再訓練的重要參考，實現模型的持續優化。

---


## 📝 **授權聲明 Authorization**

This project is for educational purposes only.

Last updated: 2025-12-25

---

🌀 **致謝**
- 國立中興大學 陳煥教授
- Gemini Cli
- Perplexity