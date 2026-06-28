# -*- coding: utf-8 -*-
"""脑卒中数据可视化与预测分析主程序。"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "brain_stroke.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
RANDOM_STATE = 42

NUMERIC_COLUMNS = ["age", "avg_glucose_level", "bmi"]
BINARY_COLUMNS = ["hypertension", "heart_disease"]
CATEGORICAL_COLUMNS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS

CN_NAME = {
    "age": "年龄",
    "avg_glucose_level": "平均血糖水平",
    "bmi": "BMI",
    "hypertension": "高血压",
    "heart_disease": "心脏病",
    "gender": "性别",
    "ever_married": "婚姻状况",
    "work_type": "工作类型",
    "Residence_type": "居住类型",
    "smoking_status": "吸烟状态",
    "stroke": "脑卒中",
}

COLOR_NO = "#4C78A8"
COLOR_YES = "#E45756"
COLOR_ACCENT = "#59A14F"
COLOR_PURPLE = "#8F63B8"


def configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Noto Sans CJK SC",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"未找到数据文件：{path}")
    data = pd.read_csv(path)
    required = set(FEATURE_COLUMNS + ["stroke"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"数据缺少字段：{sorted(missing)}")
    data = data.drop_duplicates().reset_index(drop=True)
    for column in NUMERIC_COLUMNS + BINARY_COLUMNS + ["stroke"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=FEATURE_COLUMNS + ["stroke"]).reset_index(drop=True)
    data["hypertension"] = data["hypertension"].astype(int)
    data["heart_disease"] = data["heart_disease"].astype(int)
    data["stroke"] = data["stroke"].astype(int)
    return data


def add_derived_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    bins = [-np.inf, 17, 34, 49, 64, 74, np.inf]
    labels = ["0-17岁", "18-34岁", "35-49岁", "50-64岁", "65-74岁", "75岁及以上"]
    result["age_group"] = pd.cut(result["age"], bins=bins, labels=labels)
    result["stroke_label"] = result["stroke"].map({0: "未发生脑卒中", 1: "发生脑卒中"})
    return result


def save_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close()


def annotate_bars(ax, fmt: str = "{:.1f}", suffix: str = "") -> None:
    for patch in ax.patches:
        value = patch.get_height()
        ax.annotate(
            fmt.format(value) + suffix,
            (patch.get_x() + patch.get_width() / 2, value),
            ha="center",
            va="bottom",
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=9,
        )


def make_overview_tables(data: pd.DataFrame) -> None:
    overview = pd.DataFrame(
        {
            "项目": ["样本数", "字段数", "重复记录数", "缺失单元格数", "脑卒中样本数", "脑卒中占比"],
            "结果": [
                len(data),
                data.shape[1],
                int(data.duplicated().sum()),
                int(data.isna().sum().sum()),
                int(data["stroke"].sum()),
                f"{data['stroke'].mean() * 100:.2f}%",
            ],
        }
    )
    overview.to_csv(OUTPUT_DIR / "data_overview.csv", index=False, encoding="utf-8-sig")

    numeric_summary = data[NUMERIC_COLUMNS].describe().T.round(2)
    numeric_summary.index = [CN_NAME[item] for item in numeric_summary.index]
    numeric_summary.to_csv(OUTPUT_DIR / "numeric_summary.csv", encoding="utf-8-sig")

    by_stroke = data.groupby("stroke")[NUMERIC_COLUMNS].agg(["mean", "median", "std"]).round(2)
    by_stroke.to_csv(OUTPUT_DIR / "numeric_by_stroke.csv", encoding="utf-8-sig")

    categorical_rows = []
    for column in BINARY_COLUMNS + CATEGORICAL_COLUMNS:
        grouped = data.groupby(column, dropna=False)["stroke"].agg(["count", "sum", "mean"]).reset_index()
        grouped.columns = ["类别", "样本数", "脑卒中人数", "脑卒中率"]
        grouped.insert(0, "变量", CN_NAME[column])
        grouped["脑卒中率"] = (grouped["脑卒中率"] * 100).round(2)
        categorical_rows.append(grouped)
    pd.concat(categorical_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "categorical_stroke_rates.csv", index=False, encoding="utf-8-sig"
    )


def plot_data_quality(data: pd.DataFrame) -> None:
    missing = data.isna().sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    ax.bar([CN_NAME.get(x, x) for x in missing.index], missing.values, color=COLOR_NO)
    ax.set_title("各字段缺失值数量", fontsize=15, pad=12)
    ax.set_ylabel("缺失数量")
    ax.set_xlabel("字段")
    ax.tick_params(axis="x", rotation=35)
    annotate_bars(ax, fmt="{:.0f}")
    ax.grid(axis="y", alpha=0.2)
    save_figure("01_data_quality.png")


def plot_stroke_distribution(data: pd.DataFrame) -> None:
    counts = data["stroke"].value_counts().sort_index()
    labels = ["未发生脑卒中", "发生脑卒中"]
    plt.figure(figsize=(8, 5.5))
    ax = plt.gca()
    bars = ax.bar(labels, counts.values, color=[COLOR_NO, COLOR_YES], width=0.55)
    ax.set_title("脑卒中目标变量分布", fontsize=15, pad=12)
    ax.set_ylabel("样本数量")
    ax.grid(axis="y", alpha=0.2)
    total = len(data)
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + total * 0.012,
            f"{value}人\n({value / total * 100:.2f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    save_figure("02_stroke_distribution.png")


def plot_numeric_distributions(data: pd.DataFrame) -> None:
    chart_info = [
        ("age", "年龄分布对比", "年龄（岁）", "03_age_distribution_by_stroke.png", (0, 85)),
        (
            "avg_glucose_level",
            "平均血糖水平分布对比",
            "平均血糖水平",
            "04_glucose_distribution_by_stroke.png",
            None,
        ),
        ("bmi", "BMI分布对比", "BMI", "05_bmi_distribution_by_stroke.png", None),
    ]
    for column, title, xlabel, filename, xlim in chart_info:
        plt.figure(figsize=(9, 5.5))
        ax = plt.gca()
        no_values = data.loc[data["stroke"] == 0, column]
        yes_values = data.loc[data["stroke"] == 1, column]
        bins = 28 if column != "age" else np.arange(0, 86, 4)
        ax.hist(no_values, bins=bins, density=True, alpha=0.55, color=COLOR_NO, label="未发生脑卒中")
        ax.hist(yes_values, bins=bins, density=True, alpha=0.55, color=COLOR_YES, label="发生脑卒中")
        ax.axvline(no_values.mean(), color=COLOR_NO, linestyle="--", linewidth=1.8)
        ax.axvline(yes_values.mean(), color=COLOR_YES, linestyle="--", linewidth=1.8)
        ax.set_title(title, fontsize=15, pad=12)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("概率密度")
        if xlim:
            ax.set_xlim(*xlim)
        ax.legend()
        ax.grid(alpha=0.2)
        save_figure(filename)


def plot_age_group_rate(data: pd.DataFrame) -> None:
    rate = data.groupby("age_group", observed=False)["stroke"].agg(["count", "sum", "mean"]).reset_index()
    rate["rate"] = rate["mean"] * 100
    rate.to_csv(OUTPUT_DIR / "age_group_stroke_rate.csv", index=False, encoding="utf-8-sig")
    plt.figure(figsize=(10, 5.5))
    ax = plt.gca()
    bars = ax.bar(rate["age_group"].astype(str), rate["rate"], color=COLOR_PURPLE)
    ax.set_title("不同年龄组脑卒中率", fontsize=15, pad=12)
    ax.set_xlabel("年龄组")
    ax.set_ylabel("脑卒中率（%）")
    ax.grid(axis="y", alpha=0.2)
    for bar, value, count in zip(bars, rate["rate"], rate["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.2f}%\nn={count}", ha="center", fontsize=9)
    save_figure("06_stroke_rate_by_age_group.png")


def plot_binary_rates(data: pd.DataFrame) -> None:
    for idx, column in enumerate(["hypertension", "heart_disease"], start=7):
        rate = data.groupby(column)["stroke"].agg(["count", "sum", "mean"]).reset_index()
        rate["rate"] = rate["mean"] * 100
        labels = ["否", "是"]
        title = "高血压与脑卒中率" if column == "hypertension" else "心脏病与脑卒中率"
        filename = "07_stroke_rate_by_hypertension.png" if column == "hypertension" else "08_stroke_rate_by_heart_disease.png"
        plt.figure(figsize=(8, 5.5))
        ax = plt.gca()
        bars = ax.bar(labels, rate["rate"], color=[COLOR_NO, COLOR_YES], width=0.55)
        ax.set_title(title, fontsize=15, pad=12)
        ax.set_ylabel("脑卒中率（%）")
        ax.grid(axis="y", alpha=0.2)
        for bar, value, count in zip(bars, rate["rate"], rate["count"]):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.2f}%\nn={count}", ha="center", fontsize=10)
        save_figure(filename)


def plot_categorical_rate(data: pd.DataFrame, column: str, title: str, filename: str, order=None) -> None:
    rate = data.groupby(column)["stroke"].agg(["count", "sum", "mean"]).reset_index()
    rate["rate"] = rate["mean"] * 100
    if order:
        rate[column] = pd.Categorical(rate[column], categories=order, ordered=True)
        rate = rate.sort_values(column)
    else:
        rate = rate.sort_values("rate", ascending=False)
    label_map = {
        "formerly smoked": "曾经吸烟",
        "smokes": "目前吸烟",
        "never smoked": "从不吸烟",
        "Unknown": "未知",
        "Private": "私营企业",
        "Self-employed": "个体经营",
        "Govt_job": "政府工作",
        "children": "儿童",
        "Yes": "是",
        "No": "否",
        "Urban": "城市",
        "Rural": "农村",
        "Male": "男性",
        "Female": "女性",
    }
    labels = [label_map.get(str(x), str(x)) for x in rate[column]]
    plt.figure(figsize=(10, 5.7))
    ax = plt.gca()
    bars = ax.bar(labels, rate["rate"], color=COLOR_ACCENT)
    ax.set_title(title, fontsize=15, pad=12)
    ax.set_ylabel("脑卒中率（%）")
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", rotation=20)
    for bar, value, count in zip(bars, rate["rate"], rate["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.18, f"{value:.2f}%\nn={count}", ha="center", fontsize=8.8)
    save_figure(filename)


def plot_correlation_heatmap(data: pd.DataFrame) -> None:
    numeric = data[NUMERIC_COLUMNS + BINARY_COLUMNS + ["stroke"]].copy()
    corr = numeric.corr()
    labels = [CN_NAME[x] for x in corr.columns]
    plt.figure(figsize=(8.2, 6.8))
    ax = plt.gca()
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_title("数值变量相关系数热力图", fontsize=15, pad=14)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(value) > 0.5 else "black")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson相关系数")
    save_figure("13_correlation_heatmap.png")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("bin", "passthrough", BINARY_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ],
        verbose_feature_names_out=False,
    )


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.85, 161)
    f1_values = [f1_score(y_true, probability >= threshold) for threshold in thresholds]
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def train_models(data: pd.DataFrame):
    X = data[FEATURE_COLUMNS]
    y = data["stroke"]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    model_definitions = {
        "逻辑回归": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "决策树": DecisionTreeClassifier(
            class_weight="balanced",
            max_depth=5,
            min_samples_leaf=12,
            random_state=RANDOM_STATE,
        ),
        "随机森林": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            max_depth=8,
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "梯度提升": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=2,
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    fitted = {}
    probabilities = {}
    predictions = {}

    for name, estimator in model_definitions.items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
        fit_kwargs = {}
        if name == "梯度提升":
            fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_train)
        pipeline.fit(X_train, y_train, **fit_kwargs)

        valid_probability = pipeline.predict_proba(X_valid)[:, 1]
        threshold, valid_f1 = choose_threshold(y_valid, valid_probability)
        test_probability = pipeline.predict_proba(X_test)[:, 1]
        test_prediction = (test_probability >= threshold).astype(int)
        matrix = confusion_matrix(y_test, test_prediction)

        results.append(
            {
                "模型": name,
                "验证集最优阈值": round(threshold, 3),
                "验证集F1": round(valid_f1, 4),
                "准确率": round(accuracy_score(y_test, test_prediction), 4),
                "精确率": round(precision_score(y_test, test_prediction), 4),
                "召回率": round(recall_score(y_test, test_prediction), 4),
                "F1值": round(f1_score(y_test, test_prediction), 4),
                "ROC-AUC": round(roc_auc_score(y_test, test_probability), 4),
                "PR-AUC": round(average_precision_score(y_test, test_probability), 4),
                "TN": int(matrix[0, 0]),
                "FP": int(matrix[0, 1]),
                "FN": int(matrix[1, 0]),
                "TP": int(matrix[1, 1]),
            }
        )
        fitted[name] = pipeline
        probabilities[name] = test_probability
        predictions[name] = test_prediction

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    # 医疗筛查任务优先召回率，且随机森林便于解释，因此选作主模型。
    main_model_name = "随机森林"
    main_model = fitted[main_model_name]
    main_threshold = float(result_frame.loc[result_frame["模型"] == main_model_name, "验证集最优阈值"].iloc[0])
    joblib.dump(main_model, OUTPUT_DIR / "stroke_random_forest.joblib")

    metadata = {
        "main_model": main_model_name,
        "threshold": main_threshold,
        "train_size": int(len(X_train)),
        "validation_size": int(len(X_valid)),
        "test_size": int(len(X_test)),
        "positive_rate": float(y.mean()),
        "feature_columns": FEATURE_COLUMNS,
        "random_state": RANDOM_STATE,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    (OUTPUT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return result_frame, fitted, probabilities, predictions, X_test, y_test, main_model_name


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    selected = metrics.set_index("模型")[["准确率", "召回率", "F1值", "ROC-AUC", "PR-AUC"]]
    x = np.arange(len(selected.index))
    width = 0.15
    plt.figure(figsize=(11.5, 6.2))
    ax = plt.gca()
    for idx, column in enumerate(selected.columns):
        ax.bar(x + (idx - 2) * width, selected[column], width=width, label=column)
    ax.set_xticks(x, selected.index)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("指标值")
    ax.set_title("四种分类模型测试集性能对比", fontsize=15, pad=12)
    ax.legend(ncol=3, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    save_figure("14_model_comparison.png")


def plot_roc_curves(y_test, probabilities: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(8.2, 6.3))
    ax = plt.gca()
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        auc_value = roc_auc_score(y_test, probability)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name}（AUC={auc_value:.3f}）")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="随机分类")
    ax.set_xlabel("假阳性率")
    ax.set_ylabel("真正率（召回率）")
    ax.set_title("模型ROC曲线对比", fontsize=15, pad=12)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    save_figure("15_roc_curves.png")


def plot_pr_curves(y_test, probabilities: dict[str, np.ndarray]) -> None:
    plt.figure(figsize=(8.2, 6.3))
    ax = plt.gca()
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probability)
        ap = average_precision_score(y_test, probability)
        ax.plot(recall, precision, linewidth=2, label=f"{name}（AP={ap:.3f}）")
    ax.axhline(y_test.mean(), linestyle="--", color="gray", label="正类基准比例")
    ax.set_xlabel("召回率")
    ax.set_ylabel("精确率")
    ax.set_title("模型精确率-召回率曲线", fontsize=15, pad=12)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    save_figure("16_precision_recall_curves.png")


def plot_confusion_matrix(y_test, prediction: np.ndarray, model_name: str) -> None:
    matrix = confusion_matrix(y_test, prediction)
    plt.figure(figsize=(6.5, 5.7))
    ax = plt.gca()
    display = ConfusionMatrixDisplay(matrix, display_labels=["未发生", "发生"])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"{model_name}混淆矩阵", fontsize=15, pad=12)
    save_figure("17_random_forest_confusion_matrix.png")


def plot_feature_importance(main_model: Pipeline) -> None:
    preprocessor = main_model.named_steps["preprocess"]
    estimator = main_model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importance = estimator.feature_importances_
    feature_frame = pd.DataFrame({"特征": feature_names, "重要性": importance})

    def translate_feature(name: str) -> str:
        mapping = {
            "age": "年龄",
            "avg_glucose_level": "平均血糖",
            "bmi": "BMI",
            "hypertension": "高血压",
            "heart_disease": "心脏病",
            "gender_Female": "性别_女性",
            "gender_Male": "性别_男性",
            "ever_married_No": "婚姻_未婚",
            "ever_married_Yes": "婚姻_已婚",
            "work_type_Govt_job": "工作_政府",
            "work_type_Private": "工作_私营",
            "work_type_Self-employed": "工作_个体",
            "work_type_children": "工作_儿童",
            "Residence_type_Rural": "居住_农村",
            "Residence_type_Urban": "居住_城市",
            "smoking_status_Unknown": "吸烟_未知",
            "smoking_status_formerly smoked": "吸烟_曾经",
            "smoking_status_never smoked": "吸烟_从不",
            "smoking_status_smokes": "吸烟_目前",
        }
        return mapping.get(name, name)

    feature_frame["特征中文"] = feature_frame["特征"].map(translate_feature)
    feature_frame = feature_frame.sort_values("重要性", ascending=False)
    feature_frame.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
    top = feature_frame.head(12).sort_values("重要性")
    plt.figure(figsize=(9.5, 6.2))
    ax = plt.gca()
    bars = ax.barh(top["特征中文"], top["重要性"], color=COLOR_PURPLE)
    ax.set_xlabel("特征重要性")
    ax.set_title("随机森林模型Top12特征重要性", fontsize=15, pad=12)
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, top["重要性"]):
        ax.text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    save_figure("18_feature_importance.png")


def write_text_summary(data: pd.DataFrame, metrics: pd.DataFrame) -> None:
    numeric = data.groupby("stroke")[NUMERIC_COLUMNS].mean()
    hypertension_rate = data.groupby("hypertension")["stroke"].mean() * 100
    heart_rate = data.groupby("heart_disease")["stroke"].mean() * 100
    main = metrics.loc[metrics["模型"] == "随机森林"].iloc[0]
    lines = [
        "脑卒中数据可视化与预测分析结果摘要",
        "=" * 36,
        f"有效样本数：{len(data)}",
        f"脑卒中样本：{int(data['stroke'].sum())}，占比 {data['stroke'].mean() * 100:.2f}%",
        f"未卒中组平均年龄：{numeric.loc[0, 'age']:.2f} 岁；卒中组平均年龄：{numeric.loc[1, 'age']:.2f} 岁",
        f"未卒中组平均血糖：{numeric.loc[0, 'avg_glucose_level']:.2f}；卒中组平均血糖：{numeric.loc[1, 'avg_glucose_level']:.2f}",
        f"无高血压组脑卒中率：{hypertension_rate.loc[0]:.2f}%；高血压组：{hypertension_rate.loc[1]:.2f}%",
        f"无心脏病组脑卒中率：{heart_rate.loc[0]:.2f}%；心脏病组：{heart_rate.loc[1]:.2f}%",
        "",
        "主模型：随机森林（使用类别权重处理类别不平衡）",
        f"阈值：{main['验证集最优阈值']:.3f}",
        f"准确率：{main['准确率']:.4f}",
        f"精确率：{main['精确率']:.4f}",
        f"召回率：{main['召回率']:.4f}",
        f"F1值：{main['F1值']:.4f}",
        f"ROC-AUC：{main['ROC-AUC']:.4f}",
        f"PR-AUC：{main['PR-AUC']:.4f}",
        "",
        "说明：本项目仅用于课程学习与数据分析演示，不能替代医学诊断。",
    ]
    (OUTPUT_DIR / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    data = add_derived_columns(load_data())
    data.to_csv(OUTPUT_DIR / "processed_brain_stroke.csv", index=False, encoding="utf-8-sig")
    make_overview_tables(data)

    plot_data_quality(data)
    plot_stroke_distribution(data)
    plot_numeric_distributions(data)
    plot_age_group_rate(data)
    plot_binary_rates(data)
    plot_categorical_rate(
        data,
        "smoking_status",
        "不同吸烟状态的脑卒中率",
        "09_stroke_rate_by_smoking_status.png",
        ["never smoked", "formerly smoked", "smokes", "Unknown"],
    )
    plot_categorical_rate(data, "work_type", "不同工作类型的脑卒中率", "10_stroke_rate_by_work_type.png")
    plot_categorical_rate(
        data,
        "ever_married",
        "不同婚姻状态的脑卒中率",
        "11_stroke_rate_by_marital_status.png",
        ["No", "Yes"],
    )
    plot_categorical_rate(
        data,
        "Residence_type",
        "不同居住类型的脑卒中率",
        "12_stroke_rate_by_residence.png",
        ["Rural", "Urban"],
    )
    plot_correlation_heatmap(data)

    metrics, fitted, probabilities, predictions, _, y_test, main_model_name = train_models(data)
    plot_model_comparison(metrics)
    plot_roc_curves(y_test, probabilities)
    plot_pr_curves(y_test, probabilities)
    plot_confusion_matrix(y_test, predictions[main_model_name], main_model_name)
    plot_feature_importance(fitted[main_model_name])
    write_text_summary(data, metrics)

    print("分析完成。输出目录：", OUTPUT_DIR)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
