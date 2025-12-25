import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import altair as alt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 導入我們自己的模組
from .data_loader import load_data
from .feature_engineering import apply_feature_extraction


def plot_gauge_chart(probability: float, title: str = "AI 生成機率"):
    """
    使用 Plotly 繪製儀表板圖來顯示 AI 生成機率。

    Args:
        probability (float): AI 生成的機率 (值應在 0 和 1 之間)。
        title (str): 圖表的標題。

    Returns:
        plotly.graph_objects.Figure: 繪製好的 Plotly Figure 對象。
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={'suffix': '%', 'font': {'size': 50}},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'lightgreen'},
                {'range': [50, 75], 'color': 'yellow'},
                {'range': [75, 100], 'color': 'red'}
            ],
        }
    ))
    fig.update_layout(height=300)
    return fig

def plot_confusion_matrix(y_true, y_pred, labels, title="混淆矩陣"):
    """
    使用 Plotly 繪製混淆矩陣。

    Args:
        y_true (array-like): 真實標籤。
        y_pred (array-like): 預測標籤。
        labels (list): 類別標籤列表，例如 ['human', 'ai']。
        title (str): 圖表的標題。

    Returns:
        plotly.graph_objects.Figure: 繪製好的 Plotly Figure 對象。
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = go.Figure(data=go.Heatmap(
                   z=cm,
                   x=labels,
                   y=labels,
                   hoverongaps=False,
                   colorscale='Blues',
                   text=cm,
                   texttemplate="%{text}"
                   ))
    fig.update_layout(
        title=title,
        xaxis_title="預測標籤",
        yaxis_title="真實標籤",
        xaxis=dict(side='top')
    )
    return fig

def display_metrics(y_true, y_pred, labels):
    """
    在 Streamlit 中顯示模型評估指標。

    Args:
        y_true (array-like): 真實標籤。
        y_pred (array-like): 預測標籤。
        labels (list): 類別標籤列表。
    """
    st.subheader("📊 模型評估指標")
    accuracy = accuracy_score(y_true, y_pred)
    
    # 使用 st.markdown 和 HTML 來模擬 st.metric 並自訂顏色
    color = "#DD5C6A" if accuracy >= 0.5 else "#3ABBDE"
    st.markdown(f'<h6>準確率 (Accuracy)</h6><h3 style="color:{color};">{accuracy:.2f}</h3>', unsafe_allow_html=True)


    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
    
    report_df = pd.DataFrame(report).transpose()
    
    # 將索引和欄位名稱改為首字母大寫，並確保 'ai'/'human' 轉換為 'AI'/'Human'
    def format_report_name(name):
        name_lower = name.lower()
        if name_lower == 'ai':
            return 'AI'
        elif name_lower == 'human':
            return 'Human'
        else:
            return name.capitalize()

    report_df.index = [format_report_name(name) for name in report_df.index]
    report_df.columns = [format_report_name(name) for name in report_df.columns]
    
    st.write("**分類報告 (Classification Report)：**")
    # 格式化 DataFrame
    formatted_df = report_df.style.format({
        "Precision": "{:.2f}",
        "Recall": "{:.2f}",
        "F1-score": "{:.2f}",
        "Support": "{:g}" # 維持 Support 為整數
    })
    st.dataframe(formatted_df)

def plot_sentence_length_distribution(df, label_col='label', feature_col='avg_sentence_length', title="句長分佈 (AI vs Human)"):
    """
    使用 Altair 繪製句長分佈的盒鬚圖，並返回統計數據。

    Args:
        df (pd.DataFrame): 包含標籤和句長特徵的 DataFrame。
        label_col (str): 標籤所在的欄位名稱。
        feature_col (str): 句長特徵所在的欄位名稱。
        title (str): 圖表的標題。

    Returns:
        tuple: (altair.Chart, pd.DataFrame)，分別為圖表對象和統計數據 DataFrame。
    """
    # 複製 DataFrame 以避免修改原始資料
    plot_df = df.copy()
    # 將標籤修改為首字母大寫
    plot_df[label_col] = plot_df[label_col].map({'ai': 'AI', 'human': 'Human'})
    
    chart = alt.Chart(plot_df).mark_boxplot(extent='min-max').encode(
        # 將 x 和 y 對調以實現橫向排列
        x=alt.X(f'{feature_col}:Q', title='平均句長'),
        y=alt.Y(f'{label_col}:N', title='文本類別', sort=['AI', 'Human']), # 保持 Y 軸順序
        color=alt.Color(f'{label_col}:N', legend=None, scale=alt.Scale(domain=['AI', 'Human'], range=['#DD5C6A', '#9FCE63'])) # 使用自訂顏色
    ).properties(
        title=title
    )
    
    # 計算統計數據 (更穩健的方法，避免 describe() 的內部 unstack)
    grouped = plot_df.groupby(label_col)[feature_col]
    stats_df = pd.DataFrame({
        '數量': grouped.count(),
        '平均值': grouped.mean(),
        '標準差': grouped.std(),
        '最小值': grouped.min(),
        'Q1': grouped.quantile(0.25),
        '中位數 (Q2)': grouped.median(),
        'Q3': grouped.quantile(0.75),
        '最大值': grouped.max()
    })
    
    return chart, stats_df


if __name__ == '__main__':
    st.set_page_config(layout="wide")
    st.title("視覺化模組測試")

    # 模擬儀表板圖
    st.header("儀表板圖測試")
    st.plotly_chart(plot_gauge_chart(0.88))


    # 模擬資料載入與特徵提取
    raw_df = load_data(data_dir='data/raw') # 從專案根目錄執行
    
    if not raw_df.empty:
        featured_df = apply_feature_extraction(raw_df)
        
        # 模擬模型預測結果
        le = LabelEncoder()
        featured_df['label_encoded'] = le.fit_transform(featured_df['label'])
        
        # 將 y_true 的標籤統一轉換為 'AI' 和 'Human'
        label_map = {'ai': 'AI', 'human': 'Human'}
        y_true = featured_df['label'].map(label_map)
        class_labels = sorted(y_true.unique().tolist())

        y_pred = y_true.copy()
        num_errors = int(len(y_pred) * 0.1)
        if num_errors > 0:
            human_indices = y_pred.index[y_pred == 'Human']
            if len(human_indices) >= num_errors:
                error_indices = np.random.choice(human_indices, num_errors, replace=False)
                y_pred.loc[error_indices] = 'AI'

        col1, col2 = st.columns(2)

        with col1:
            st.header("混淆矩陣測試")
            st.plotly_chart(plot_confusion_matrix(y_true, y_pred, labels=class_labels), use_container_width=True)

        with col2:
            st.header("模型評估指標測試")
            display_metrics(y_true, y_pred, labels=class_labels)

        st.markdown("---") # 分隔線

        col3, col4 = st.columns(2)

        with col3:
            st.header("句長分佈圖測試")
            dist_chart, dist_stats = plot_sentence_length_distribution(featured_df)
            st.altair_chart(dist_chart, use_container_width=True, theme="streamlit")
        
        with col4:
            st.header("句長分佈對應統計數據:")
            # 格式化統計數據 DataFrame
            formatted_stats_df = dist_stats.style.format({
                "平均值": "{:.2f}",
                "標準差": "{:.2f}",
                "最小值": "{:.2f}",
                "Q1": "{:.2f}",
                "中位數 (Q2)": "{:.2f}",
                "Q3": "{:.2f}",
                "最大值": "{:.2f}",
                "數量": "{:g}"
            })
            st.dataframe(formatted_stats_df)
        
    else:
        st.error("無法載入資料或資料為空，無法進行視覺化測試。")
