"""基于 Streamlit 的垃圾邮件智能识别界面。"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from spam_classifier import TEXT_COLUMNS, load_model, predict_messages

MODEL_PATH = Path("models/spam_classifier.joblib")
REPORT_PATH = Path("models/training_report.json")

st.set_page_config(
    page_title="垃圾邮件智能识别系统",
    page_icon="📨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f6f9ff 0%, #ffffff 35%);
        font-size: 18px;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .hero {
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        background: linear-gradient(125deg, #172554 0%, #1d4ed8 58%, #06b6d4 100%);
        color: white;
        box-shadow: 0 18px 45px rgba(29, 78, 216, 0.18);
        margin-bottom: 1.5rem;
    }
    .hero-badge {
        display: inline-block;
        padding: .35rem .75rem;
        border-radius: 999px;
        background: rgba(255,255,255,.15);
        border: 1px solid rgba(255,255,255,.28);
        font-size: .82rem;
        letter-spacing: .04em;
        margin-bottom: .9rem;
    }
    .hero h1 {
        color: white !important;
        font-size: clamp(2rem, 4vw, 3.2rem);
        margin: 0 0 .6rem 0;
        line-height: 1.15;
    }
    .hero p {
        margin: 0;
        color: rgba(255,255,255,.82);
        font-size: 1.15rem;
        line-height: 1.75;
    }
    h2 {
        font-size: 1.8rem !important;
    }
    h3 {
        font-size: 1.45rem !important;
    }
    div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stMarkdownContainer"] li {
        font-size: 1.06rem;
        line-height: 1.75;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5eaf3;
        padding: 1rem 1.1rem;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, .05);
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 14px;
        border-color: #cbd5e1;
        background: rgba(255,255,255,.9);
        font-size: 1.08rem;
        line-height: 1.7;
        padding: 1rem;
    }
    div[data-testid="stTextArea"] label p,
    div[data-testid="stFileUploader"] label p {
        font-size: 1.05rem !important;
        font-weight: 600;
    }
    div[data-testid="stButton"] button p,
    div[data-testid="stDownloadButton"] button p {
        font-size: 1.08rem !important;
        font-weight: 650;
    }
    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #93c5fd;
        padding: .8rem;
        border-radius: 16px;
    }
    .result-card {
        padding: 1.25rem 1.4rem;
        border-radius: 16px;
        margin-top: 1rem;
        font-size: 1.05rem;
        font-weight: 600;
    }
    .result-spam {
        color: #991b1b;
        background: #fef2f2;
        border: 1px solid #fecaca;
    }
    .result-ham {
        color: #166534;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
    }
    .section-note {
        color: #64748b;
        font-size: 1.02rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    div[data-baseweb="tab-list"] {
        gap: .35rem;
    }
    button[data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding-left: 1.15rem;
        padding-right: 1.15rem;
    }
    button[data-baseweb="tab"] p {
        font-size: 1.05rem !important;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">MACHINE LEARNING · NLP</div>
        <h1>垃圾邮件智能识别系统</h1>
        <p>面向中英文短信与邮件，对比词频与 TF-IDF 特征，结合朴素贝叶斯、逻辑回归和 SVM 识别可疑信息。</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("尚未找到训练模型。请先在项目目录执行 `python train.py --demo`，或使用正式数据训练。")
    st.stop()

model = get_model()

with st.sidebar:
    st.header("系统信息")
    st.success("模型已加载")
    st.write("**模型文件**")
    st.code(str(MODEL_PATH), language=None)
    st.write("**识别流程**")
    st.caption("文本清洗 → TF-IDF → 分类模型 → 概率输出")

single_tab, batch_tab, report_tab = st.tabs(["🔎 单条识别", "📚 批量识别", "📊 模型报告"])

with single_tab:
    st.subheader("检测一条消息")
    st.markdown('<div class="section-note">粘贴短信或邮件正文，系统将给出分类结果与垃圾概率。</div>', unsafe_allow_html=True)
    message = st.text_area(
        "输入短信或邮件内容",
        height=210,
        placeholder="例如：恭喜您获得免费奖品，请点击链接立即领取……",
    )
    if st.button("立即检测", type="primary", width="stretch", icon="🔍"):
        if not message.strip():
            st.warning("请输入需要识别的内容。")
        else:
            result = predict_messages(model, [message]).iloc[0]
            probability = float(result["spam_probability"])
            left, right = st.columns(2)
            left.metric("识别结果", result["prediction"])
            right.metric("垃圾概率", f"{probability:.2%}")
            st.progress(probability, text="模型判定为垃圾邮件的概率")
            if result["prediction"] == "垃圾邮件":
                st.markdown(
                    '<div class="result-card result-spam">⚠️ 此内容可能是垃圾邮件。请勿随意点击链接、转账或提供个人信息。</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card result-ham">✅ 此内容更可能是正常消息，但仍请核实陌生链接和发件人身份。</div>',
                    unsafe_allow_html=True,
                )

with batch_tab:
    st.subheader("批量检测")
    st.markdown('<div class="section-note">上传包含 text、message、email 或 content 列的 CSV 文件。</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded, encoding="utf-8-sig")
            lowered = {str(column).lower(): column for column in batch.columns}
            text_col = next((lowered[name] for name in TEXT_COLUMNS if name in lowered), None)
            if text_col is None:
                st.error("未找到文本列，请使用 text、message、email 或 content 作为列名。")
            else:
                output = batch.copy()
                results = predict_messages(model, output[text_col].fillna("").astype(str).tolist())
                output["识别结果"] = results["prediction"]
                output["垃圾概率"] = results["spam_probability"].round(6)
                st.dataframe(output, width="stretch", alt="批量邮件识别结果")
                csv_data = output.to_csv(index=False).encode("utf-8-sig")
                st.download_button("下载识别结果", csv_data, "spam_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"文件处理失败：{exc}")

with report_tab:
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        st.subheader("训练摘要")
        if report["samples"] <= 100:
            st.warning("当前是小型演示模型。请运行官方数据下载与训练脚本，以获得可靠的实验结果。")
        else:
            st.success("当前模型已使用完整 SMS Spam Collection 英文数据集和 TREC06C 中文邮件数据集进行训练。")
        cols = st.columns(4)
        cols[0].metric("总样本", report["samples"])
        cols[1].metric("垃圾邮件", report["spam_samples"])
        cols[2].metric("正常邮件", report["ham_samples"])
        cols[3].metric("最优组合", report.get("best_model_name", report["best_algorithm"]))

        if report.get("language_metrics"):
            st.subheader("中英文分项表现")
            language_table = pd.DataFrame(report["language_metrics"]).T
            language_test_samples = report.get("language_test_samples", {})
            language_table.insert(
                0,
                "数据集总量",
                [report.get("language_samples", {}).get(name, 0) for name in language_table.index],
            )
            language_table.insert(
                1,
                "测试样本数",
                [
                    language_test_samples.get(
                        name,
                        sum(sum(row) for row in report["language_metrics"][name]["confusion_matrix"]),
                    )
                    for name in language_table.index
                ],
            )
            st.dataframe(
                language_table[
                    [
                        "数据集总量",
                        "测试样本数",
                        "accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "roc_auc",
                    ]
                ],
                width="stretch",
                alt="中文邮件和英文短信在独立测试集上的分类表现",
            )
            st.caption(
                "数据集总量为去重后的完整样本数；测试样本数及各项指标来自独立测试集。"
                "TREC06C 中文邮件与 UCI 英文短信分别参与分层划分。"
            )

        table = pd.DataFrame(report["metrics"]).T
        if "display_name" in table.columns:
            table.index = table["display_name"]
        if "cv_f1_mean" in table.columns:
            st.subheader("分层五折交叉验证")
            st.markdown(
                '<div class="section-note">模型选择以平均 F1 为主；若结果接近，则依次参考 Precision、ROC-AUC 和 F1 标准差。</div>',
                unsafe_allow_html=True,
            )
            cv_table = table[
                [
                    "cv_accuracy_mean",
                    "cv_precision_mean",
                    "cv_recall_mean",
                    "cv_f1_mean",
                    "cv_f1_std",
                    "cv_roc_auc_mean",
                ]
            ].rename(
                columns={
                    "cv_accuracy_mean": "平均 Accuracy",
                    "cv_precision_mean": "平均 Precision",
                    "cv_recall_mean": "平均 Recall",
                    "cv_f1_mean": "平均 F1",
                    "cv_f1_std": "F1 标准差",
                    "cv_roc_auc_mean": "平均 ROC-AUC",
                }
            )
            st.dataframe(
                cv_table.sort_values("平均 F1", ascending=False),
                width="stretch",
                alt="六种模型组合的五折交叉验证指标",
            )

        st.subheader("独立测试集最终指标")
        st.dataframe(
            table[["accuracy", "precision", "recall", "f1", "roc_auc"]],
            width="stretch",
            alt="六种模型组合在独立测试集上的最终指标",
        )
        st.subheader("测试集模型性能对比")
        comparison = table[["accuracy", "precision", "recall", "f1", "roc_auc"]].T
        st.bar_chart(
            comparison,
            height=380,
            stack=False,
            x_label="评价指标",
            y_label="得分（0–1）",
            alt="六种模型组合在独立测试集上的性能对比",
        )
        st.caption("Precision、Recall 和 F1 均以垃圾邮件为正类计算。")
        best_key = report["best_algorithm"]
        best_metrics = report["metrics"][best_key]
        true_negative, false_positive = best_metrics["confusion_matrix"][0]
        false_negative, true_positive = best_metrics["confusion_matrix"][1]
        matrix = pd.DataFrame(
            best_metrics["confusion_matrix"],
            index=["实际正常邮件", "实际垃圾邮件"],
            columns=["模型预测正常", "模型预测垃圾"],
        )
        st.subheader("最优模型混淆矩阵")
        st.caption(report.get("best_model_name", best_key))
        st.dataframe(
            matrix,
            width="stretch",
            alt="最优模型混淆矩阵，对角线为正确识别，非对角线为误报和漏报",
        )
        st.info("阅读方法：先看左侧的实际类别，再看上方的模型预测类别。对角线表示判断正确，另外两个格子表示误判。")
        result_cols = st.columns(4)
        result_cols[0].metric(
            "正确识别正常",
            f"{true_negative:,} 条",
            help="实际为正常邮件，模型也预测为正常。",
        )
        result_cols[1].metric(
            "误报",
            f"{false_positive:,} 条",
            help="实际为正常邮件，但模型错误地预测为垃圾。",
        )
        result_cols[2].metric(
            "漏报",
            f"{false_negative:,} 条",
            help="实际为垃圾邮件，但模型错误地预测为正常。",
        )
        result_cols[3].metric(
            "正确识别垃圾",
            f"{true_positive:,} 条",
            help="实际为垃圾邮件，模型也预测为垃圾。",
        )
        st.caption(
            f"测试集共 {matrix.to_numpy().sum():,} 条：正确识别 "
            f"{true_negative + true_positive:,} 条，误判 {false_positive + false_negative:,} 条。"
        )
    else:
        st.info("训练报告不存在，请重新运行训练脚本。")
