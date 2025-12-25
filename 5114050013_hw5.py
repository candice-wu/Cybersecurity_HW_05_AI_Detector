import streamlit as st
import os
import pandas as pd
import time
import sys

# 導入函式庫以獲取版本號
import sklearn
import plotly
import altair as alt

# 從 src 導入模組
from src.model import train_model, load_model, predict_text, FEATURE_COLUMNS
from src.visualization import plot_gauge_chart, plot_confusion_matrix, display_metrics

# --- 應用程式設定 ---
st.set_page_config(
    page_title="AI 文本偵測器",
    page_icon="🤖",
    layout="wide"
)

# --- 會話狀態初始化 ---
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'model_info' not in st.session_state:
    # 在啟動時就載入一次模型資訊
    st.session_state.model_info = load_model()


# --- 全局模型載入 ---
@st.cache_resource
def get_model_from_info(model_info):
    """從已載入的 model_info 中提取模型和標籤編碼器。"""
    if model_info:
        return model_info.get("model"), model_info.get("label_encoder")
    return None, None

model, le = get_model_from_info(st.session_state.model_info)

# --- 側邊欄 ---
st.sidebar.title("⚙️ 系統資訊")

with st.sidebar.expander("🤖 模型資訊", expanded=True):
    if model:
        st.success("模型已成功載入。")
    else:
        st.error("模型檔案 `model.joblib` 遺失！請先在本機端執行 `python -m src.model` 來產生模型檔案。")

with st.sidebar.expander("ℹ️ 開發與環境"):
    st.write("**開發者:** Candice Wu")
    st.write(f"**Python 版本:** {sys.version.split(' ')[0]}")
    st.write(f"**Streamlit 版本:** {st.__version__}")
    st.write(f"**Scikit-learn 版本:** {sklearn.__version__}")
    st.write(f"**Pandas 版本:** {pd.__version__}")
    st.write(f"**Plotly 版本:** {plotly.__version__}")
    st.write(f"**Altair 版本:** {alt.__version__}")

st.sidebar.markdown("---")
st.sidebar.write("© 2025 Candice Wu. All Rights Reserved.")
st.sidebar.write("最後更新: 2025-12-25")


# --- 主頁面 ---
st.title("🤖 AI 文本偵測器")
st.write("檢測輸入的文字內容是由 AI 生成還是人類撰寫。")

# --- 文本分析輸入區 ---
st.header("🔍 輸入文本進行分析")
user_input = st.text_area(
    "請在此處貼上您要分析的文本：",
    height=200,
    placeholder="在此輸入或貼上文本..."
)
uploaded_file = st.file_uploader("或上傳一個 .txt 文件進行分析", type="txt")

if st.button("開始分析", type="primary"):
    text_to_analyze = ""
    if user_input:
        text_to_analyze = user_input
    elif uploaded_file is not None:
        text_to_analyze = uploaded_file.read().decode("utf-8")

    if not text_to_analyze.strip():
        st.warning("請輸入或上傳有效的文本內容。")
    elif model is None:
        st.error("模型檔案遺失，無法進行分析。")
    else:
        start_time = time.time()
        with st.spinner("正在分析文本..."):
            prediction, confidence = predict_text(text_to_analyze, model, le)
            ai_prob = confidence if prediction.lower() == 'ai' else 1 - confidence
        
        end_time = time.time()
        processing_time = end_time - start_time
        word_count = len(text_to_analyze.split())
        words_per_sec = word_count / processing_time if processing_time > 0 else 0

        st.session_state.analysis_results = {
            "text": text_to_analyze,
            "prediction": prediction,
            "confidence": confidence,
            "ai_prob": ai_prob,
            "processing_time": processing_time,
            "word_count": word_count,
            "words_per_sec": words_per_sec
        }

# --- 結果顯示區 ---
if st.session_state.analysis_results:
    st.markdown("---")
    st.header("📊 分析結果")
    
    results = st.session_state.analysis_results
    ai_prob = results["ai_prob"]
    
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.plotly_chart(plot_gauge_chart(ai_prob), use_container_width=True)

    with col2:
        st.subheader("判定結果")
        if results["prediction"].lower() == "ai":
            st.error(f"AI 生成 ({results['confidence']*100:.2f}%)")
        else:
            st.success(f"人類撰寫 ({results['confidence']*100:.2f}%)")
        st.write("此結果基於模型的機率分佈。")

    with st.expander("🔍 判定原因與評比指標"):
        st.write("""
            我們的模型透過分析文本的多項統計與風格特徵來做出判斷。它並非理解文本的語意，而是識別 AI 生成內容與人類寫作在模式上的差異。
            主要評比的特徵維度包括：
        """)
        st.json(FEATURE_COLUMNS)
        st.write("模型會綜合這些特徵的數值，與訓練資料中學習到的模式進行比對，最終給出一個可能性判斷。")

    with st.expander("⏱️ 系統性能指標"):
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        perf_col1.metric("處理時間", f"{results['processing_time']:.2f} 秒")
        perf_col2.metric("分析字數", f"{results['word_count']} 字")
        perf_col3.metric("處理速度", f"{results['words_per_sec']:.0f} 字/秒")

# --- 模型效能顯示區 ---
if st.session_state.model_info:
    st.markdown("---")
    st.header("📈 當前模型效能")
    
    model_results = st.session_state.model_info
    y_test_labels_upper = [label.capitalize() for label in model_results["y_test"]]
    y_pred_labels_upper = [label.capitalize() for label in model_results["y_pred"]]
    class_labels = sorted(list(set(y_test_labels_upper)))

    display_metrics(y_test_labels_upper, y_pred_labels_upper, labels=class_labels)
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.write("#### 特徵重要性")
        st.dataframe(pd.Series(model_results["model"].feature_importances_, index=model_results["feature_columns"]).sort_values(ascending=False).round(4))
    with m_col2:
        st.write("#### 混淆矩陣")
        st.plotly_chart(plot_confusion_matrix(y_test_labels_upper, y_pred_labels_upper, labels=class_labels), use_container_width=True)
else:
    st.markdown("---")
    st.warning("找不到模型效能資訊。請確保 `model.joblib` 包含評估數據。")
