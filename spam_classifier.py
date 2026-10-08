"""垃圾邮件分类系统的核心训练和推理逻辑。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABEL_COLUMNS = ("label", "v1", "target", "category", "class")
TEXT_COLUMNS = ("text", "v2", "message", "sms", "email", "content")


def clean_text(text: object) -> str:
    """规范化文本，同时保留对垃圾识别有帮助的 URL/数字等标记。"""
    value = str(text).lower().strip()
    value = re.sub(r"https?://\S+|www\.\S+", " URL ", value)
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.\w+\b", " EMAIL ", value)
    value = re.sub(r"\b\d+(?:[.,]\d+)?\b", " NUMBER ", value)
    value = re.sub(r"[^\w\s!?$%￥]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def _find_column(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {str(column).strip().lower(): column for column in columns}
    return next((lowered[name] for name in candidates if name in lowered), None)


def _normalise_label(value: object) -> int:
    label = str(value).strip().lower()
    spam_values = {"spam", "1", "true", "yes", "junk", "垃圾", "垃圾邮件"}
    ham_values = {"ham", "0", "false", "no", "normal", "正常", "正常邮件", "非垃圾邮件"}
    if label in spam_values:
        return 1
    if label in ham_values:
        return 0
    raise ValueError(f"无法识别标签：{value!r}，请使用 spam/ham 或 1/0。")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """加载 UCI 原始 TSV 或带表头的 CSV/TSV 数据。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")

    if path.name.lower() == "smsspamcollection":
        frame = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    else:
        separator = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
        frame = pd.read_csv(path, sep=separator, encoding="utf-8-sig")
        label_col = _find_column(list(frame.columns), LABEL_COLUMNS)
        text_col = _find_column(list(frame.columns), TEXT_COLUMNS)
        if label_col is None or text_col is None:
            # 兼容无表头的两列 TSV。
            raw = pd.read_csv(path, sep=separator, header=None, encoding="utf-8-sig")
            if raw.shape[1] < 2:
                raise ValueError("数据至少需要标签和文本两列。")
            frame = raw.iloc[:, :2]
            frame.columns = ["label", "text"]
        else:
            frame = frame[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})

    frame = frame.dropna(subset=["label", "text"]).copy()
    frame["label"] = frame["label"].map(_normalise_label)
    frame["text"] = frame["text"].astype(str)
    frame = frame[frame["text"].str.strip().ne("")].drop_duplicates(subset=["label", "text"])
    if frame["label"].nunique() != 2:
        raise ValueError("训练数据必须同时包含正常邮件和垃圾邮件。")
    return frame.reset_index(drop=True)


def make_pipeline(algorithm: str, feature_method: str = "tfidf") -> Pipeline:
    vectorizer_options = {
        "preprocessor": clean_text,
        "lowercase": False,
        # 字符级 n-gram 不依赖英文空格分词，也能学习“刷单”“验证码”等中文片段。
        "analyzer": "char",
        "ngram_range": (2, 5),
        "min_df": 2,
        "max_df": 0.98,
        "max_features": 60_000,
    }
    if feature_method == "tfidf":
        vectorizer: Any = TfidfVectorizer(**vectorizer_options, sublinear_tf=True)
    elif feature_method == "count":
        vectorizer = CountVectorizer(**vectorizer_options)
    else:
        raise ValueError(f"不支持的特征方法：{feature_method}")

    if algorithm == "naive_bayes":
        classifier: Any = MultinomialNB(alpha=0.5)
    elif algorithm == "logistic_regression":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=42,
        )
    elif algorithm == "svm":
        # 概率校准使网页端可以展示易理解的垃圾概率。
        classifier = CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3)
    else:
        raise ValueError(f"不支持的算法：{algorithm}")
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def evaluate(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict[str, Any]:
    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)[:, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 6),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


@dataclass
class TrainingResult:
    best_algorithm: str
    model_path: Path
    report_path: Path
    report: dict[str, Any]


def train_and_save(
    data_path: str | Path,
    output_dir: str | Path = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    chinese_data_path: str | Path | None = None,
) -> TrainingResult:
    english_frame = load_dataset(data_path)
    english_frame["language"] = "英文短信"
    if chinese_data_path is None:
        trec_candidate = Path(data_path).parent / "trec06c.csv"
        sample_candidate = Path(data_path).parent / "chinese_training_messages.csv"
        if trec_candidate.exists():
            chinese_data_path = trec_candidate
        elif sample_candidate.exists():
            chinese_data_path = sample_candidate

    frames = [english_frame]
    sources = [str(Path(data_path))]
    if chinese_data_path is not None:
        chinese_frame = load_dataset(chinese_data_path)
        chinese_frame["language"] = "中文邮件"
        frames.append(chinese_frame)
        sources.append(str(Path(chinese_data_path)))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["label", "text"])
    if len(frame) < 10:
        raise ValueError("数据量过少，至少需要 10 条样本。")
    class_counts = frame["label"].value_counts()
    if class_counts.min() < 4:
        raise ValueError("每个类别至少需要 4 条样本，以便分层划分和 SVM 概率校准。")

    x_train, x_test, y_train, y_test = train_test_split(
        frame["text"],
        frame["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=frame["language"].astype(str) + "_" + frame["label"].astype(str),
    )

    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, Any]] = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    score_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    display_names = {
        "count_naive_bayes": "词频 + 朴素贝叶斯",
        "count_logistic_regression": "词频 + 逻辑回归",
        "count_svm": "词频 + SVM",
        "tfidf_naive_bayes": "TF-IDF + 朴素贝叶斯",
        "tfidf_logistic_regression": "TF-IDF + 逻辑回归",
        "tfidf_svm": "TF-IDF + SVM",
    }
    experiments = [
        (feature_method, algorithm)
        for feature_method in ("count", "tfidf")
        for algorithm in ("naive_bayes", "logistic_regression", "svm")
    ]
    print(
        f"[准备完成] 共 {len(frame)} 条数据，训练集 {len(x_train)} 条，"
        f"测试集 {len(x_test)} 条；开始比较 {len(experiments)} 个模型组合。",
        flush=True,
    )
    for experiment_number, (feature_method, algorithm) in enumerate(experiments, start=1):
            experiment = f"{feature_method}_{algorithm}"
            model = make_pipeline(algorithm, feature_method)
            print(
                f"[模型 {experiment_number}/{len(experiments)}] "
                f"{display_names[experiment]}：开始 5 折交叉验证",
                flush=True,
            )
            fold_scores = {name: [] for name in score_names}
            for fold_number, (fold_train, fold_valid) in enumerate(
                cv.split(x_train, y_train), start=1
            ):
                print(
                    f"  [交叉验证 {fold_number}/5] 正在训练...",
                    flush=True,
                )
                fold_model = clone(model)
                fold_model.fit(x_train.iloc[fold_train], y_train.iloc[fold_train])
                fold_metric = evaluate(
                    fold_model,
                    x_train.iloc[fold_valid],
                    y_train.iloc[fold_valid],
                )
                for score_name in score_names:
                    fold_scores[score_name].append(fold_metric[score_name])
                print(
                    f"  [交叉验证 {fold_number}/5] 完成，"
                    f"F1={fold_metric['f1']:.4f}",
                    flush=True,
                )

            print("  [最终拟合] 正在使用完整训练集训练...", flush=True)
            model.fit(x_train, y_train)
            models[experiment] = model
            metrics[experiment] = evaluate(model, x_test, y_test)
            metrics[experiment]["feature_method"] = feature_method
            metrics[experiment]["algorithm"] = algorithm
            metrics[experiment]["display_name"] = display_names[experiment]
            for score_name in score_names:
                values = np.asarray(fold_scores[score_name], dtype=float)
                metrics[experiment][f"cv_{score_name}_mean"] = round(float(values.mean()), 6)
                metrics[experiment][f"cv_{score_name}_std"] = round(float(values.std()), 6)
            print(
                f"[模型 {experiment_number}/{len(experiments)}] 完成，"
                f"交叉验证平均 F1={metrics[experiment]['cv_f1_mean']:.4f}",
                flush=True,
            )

    best_algorithm = max(
        metrics,
        key=lambda name: (
            metrics[name]["cv_f1_mean"],
            metrics[name]["cv_precision_mean"],
            metrics[name]["cv_roc_auc_mean"],
            -metrics[name]["cv_f1_std"],
        ),
    )
    print(f"[选择模型] 最优组合：{display_names[best_algorithm]}", flush=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "spam_classifier.joblib"
    report_path = output_dir / "training_report.json"
    joblib.dump(models[best_algorithm], model_path)
    print(f"[保存模型] {model_path}", flush=True)

    language_metrics: dict[str, dict[str, Any]] = {}
    best_model = models[best_algorithm]
    for language in sorted(frame["language"].unique()):
        language_mask = frame.loc[x_test.index, "language"].eq(language)
        if language_mask.sum() >= 2 and y_test.loc[language_mask].nunique() == 2:
            language_metrics[language] = evaluate(
                best_model,
                x_test.loc[language_mask],
                y_test.loc[language_mask],
            )

    report: dict[str, Any] = {
        "dataset": str(Path(data_path)),
        "data_sources": sources,
        "samples": int(len(frame)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "spam_samples": int((frame["label"] == 1).sum()),
        "ham_samples": int((frame["label"] == 0).sum()),
        "best_algorithm": best_algorithm,
        "best_model_name": display_names[best_algorithm],
        "selection_method": {
            "validation": "Stratified 5-fold cross-validation on training set",
            "primary_metric": "mean F1",
            "tie_breakers": ["mean precision", "mean ROC-AUC", "lower F1 standard deviation"],
            "test_set_usage": "Final evaluation only",
        },
        "language_samples": {
            language: int(count) for language, count in frame["language"].value_counts().items()
        },
        "language_test_samples": {
            language: int(frame.loc[x_test.index, "language"].eq(language).sum())
            for language in sorted(frame["language"].unique())
        },
        "language_metrics": language_metrics,
        "metrics": metrics,
        "labels": {"0": "正常邮件", "1": "垃圾邮件"},
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[训练完成] 评估报告已保存到 {report_path}", flush=True)
    return TrainingResult(best_algorithm, model_path, report_path, report)


def load_model(path: str | Path = "models/spam_classifier.joblib") -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"模型不存在：{path}。请先运行 train.py 训练模型。")
    return joblib.load(path)


def predict_messages(model: Pipeline, messages: list[str]) -> pd.DataFrame:
    if not messages:
        return pd.DataFrame(columns=["text", "prediction", "spam_probability"])
    probabilities = model.predict_proba(messages)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return pd.DataFrame(
        {
            "text": messages,
            "prediction": np.where(predictions == 1, "垃圾邮件", "正常邮件"),
            "spam_probability": probabilities,
        }
    )
