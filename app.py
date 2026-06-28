# -*- coding: utf-8 -*-
"""基于 Streamlit 的脑卒中数据交互式可视化平台。"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sklearn

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "brain_stroke.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "stroke_random_forest.joblib"
METADATA_PATH = OUTPUT_DIR / "model_metadata.json"
METRICS_PATH = OUTPUT_DIR / "model_metrics.csv"
FEATURE_PATH = OUTPUT_DIR / "feature_importance.csv"

st.set_page_config(
    page_title="脑卒中风险可视化分析平台",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background-color: #f7f9fc;}
    .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        background: linear-gradient(120deg, #1f4e79, #4c78a8);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .4rem 0 0 0; opacity: .9;}
    .hint {
        background: #eef5fb;
        border-left: 4px solid #4c78a8;
        padding: .8rem 1rem;
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["卒中状态"] = data["stroke"].map({0: "未发生", 1: "发生"})
    data["年龄组"] = pd.cut(
        data["age"],
        bins=[-np.inf, 17, 34, 49, 64, 74, np.inf],
        labels=["0-17岁", "18-34岁", "35-49岁", "50-64岁", "65-74岁", "75岁及以上"],
    )
    return data


def rebuild_model_for_current_environment() -> None:
    """仅重新训练模型，解决不同 scikit-learn 版本导致的 joblib 不兼容。"""
    from analysis import add_derived_columns, load_data as load_analysis_data, train_models

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = add_derived_columns(load_analysis_data())
    train_models(data)


@st.cache_resource
def load_model():
    try:
        if not MODEL_PATH.exists() or not METADATA_PATH.exists():
            raise FileNotFoundError("模型文件缺失")
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if metadata.get("sklearn_version") != sklearn.__version__:
            raise RuntimeError("模型与当前 scikit-learn 版本不一致")
        # 进行一次最小预测，提前识别模型损坏。
        sample = pd.DataFrame([{
            "age": 45.0,
            "avg_glucose_level": 100.0,
            "bmi": 25.0,
            "hypertension": 0,
            "heart_disease": 0,
            "gender": "Female",
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "smoking_status": "never smoked",
        }])
        model.predict_proba(sample)
        return model, metadata
    except Exception:
        rebuild_model_for_current_environment()
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return model, metadata


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### 全局筛选")
    age_range = st.sidebar.slider(
        "年龄范围",
        min_value=int(data["age"].min()),
        max_value=int(np.ceil(data["age"].max())),
        value=(int(data["age"].min()), int(np.ceil(data["age"].max()))),
    )
    genders = st.sidebar.multiselect(
        "性别",
        options=sorted(data["gender"].unique()),
        default=sorted(data["gender"].unique()),
        format_func=lambda x: {"Female": "女性", "Male": "男性"}.get(x, x),
    )
    smoking = st.sidebar.multiselect(
        "吸烟状态",
        options=sorted(data["smoking_status"].unique()),
        default=sorted(data["smoking_status"].unique()),
        format_func=lambda x: {
            "never smoked": "从不吸烟",
            "formerly smoked": "曾经吸烟",
            "smokes": "目前吸烟",
            "Unknown": "未知",
        }.get(x, x),
    )
    result = data[
        data["age"].between(age_range[0], age_range[1])
        & data["gender"].isin(genders)
        & data["smoking_status"].isin(smoking)
    ].copy()
    st.sidebar.caption(f"当前筛选保留 {len(result):,} 条记录")
    return result


def show_header() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>脑卒中风险可视化分析与预测系统</h1>
          <p>课程项目：数据概览 · 因素分析 · 模型评估 · 个体风险演示</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(data: pd.DataFrame) -> None:
    st.subheader("数据总览")
    if data.empty:
        st.warning("当前筛选条件下没有数据，请调整侧边栏。")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("样本数量", f"{len(data):,}")
    c2.metric("脑卒中人数", f"{int(data['stroke'].sum()):,}")
    c3.metric("脑卒中率", f"{data['stroke'].mean() * 100:.2f}%")
    c4.metric("平均年龄", f"{data['age'].mean():.1f} 岁")

    left, right = st.columns(2)
    with left:
        counts = data["卒中状态"].value_counts().reset_index()
        counts.columns = ["卒中状态", "人数"]
        fig = px.pie(
            counts,
            names="卒中状态",
            values="人数",
            hole=0.45,
            title="脑卒中样本构成",
            color="卒中状态",
            color_discrete_map={"未发生": "#4C78A8", "发生": "#E45756"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        age_rate = data.groupby("年龄组", observed=False)["stroke"].mean().mul(100).reset_index()
        age_rate.columns = ["年龄组", "脑卒中率"]
        fig = px.bar(
            age_rate,
            x="年龄组",
            y="脑卒中率",
            text_auto=".2f",
            title="不同年龄组脑卒中率（%）",
        )
        fig.update_traces(marker_color="#8F63B8")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 数值变量分布")
    numeric_label = st.selectbox(
        "选择数值变量",
        ["age", "avg_glucose_level", "bmi"],
        format_func=lambda x: {"age": "年龄", "avg_glucose_level": "平均血糖水平", "bmi": "BMI"}[x],
    )
    fig = px.histogram(
        data,
        x=numeric_label,
        color="卒中状态",
        marginal="box",
        barmode="overlay",
        opacity=0.65,
        nbins=35,
        color_discrete_map={"未发生": "#4C78A8", "发生": "#E45756"},
        title="分布与箱线图联动展示",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("查看原始数据"):
        st.dataframe(data.head(500), use_container_width=True, hide_index=True)


def page_factors(data: pd.DataFrame) -> None:
    st.subheader("风险因素分析")
    if data.empty:
        st.warning("当前筛选条件下没有数据，请调整侧边栏。")
        return

    label_map = {
        "hypertension": "高血压",
        "heart_disease": "心脏病",
        "smoking_status": "吸烟状态",
        "work_type": "工作类型",
        "ever_married": "婚姻状况",
        "Residence_type": "居住类型",
        "gender": "性别",
    }
    selected = st.selectbox("选择分类因素", list(label_map), format_func=label_map.get)
    rate = data.groupby(selected, dropna=False)["stroke"].agg(["count", "sum", "mean"]).reset_index()
    rate["脑卒中率"] = rate["mean"] * 100
    rate["类别"] = rate[selected].astype(str)
    translate = {
        "0": "否",
        "1": "是",
        "Female": "女性",
        "Male": "男性",
        "No": "否",
        "Yes": "是",
        "Urban": "城市",
        "Rural": "农村",
        "Private": "私营企业",
        "Self-employed": "个体经营",
        "Govt_job": "政府工作",
        "children": "儿童",
        "never smoked": "从不吸烟",
        "formerly smoked": "曾经吸烟",
        "smokes": "目前吸烟",
        "Unknown": "未知",
    }
    rate["类别"] = rate["类别"].map(lambda x: translate.get(x, x))
    fig = px.bar(
        rate.sort_values("脑卒中率", ascending=False),
        x="类别",
        y="脑卒中率",
        text="脑卒中率",
        hover_data={"count": True, "sum": True},
        title=f"{label_map[selected]}分组脑卒中率",
    )
    fig.update_traces(marker_color="#59A14F", texttemplate="%{text:.2f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    l1, l2 = st.columns(2)
    with l1:
        fig = px.box(
            data,
            x="卒中状态",
            y="age",
            color="卒中状态",
            title="不同卒中状态的年龄箱线图",
            color_discrete_map={"未发生": "#4C78A8", "发生": "#E45756"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with l2:
        fig = px.scatter(
            data,
            x="age",
            y="avg_glucose_level",
            color="卒中状态",
            size="bmi",
            hover_data=["hypertension", "heart_disease", "smoking_status"],
            opacity=0.65,
            title="年龄-血糖-BMI多维关系",
            color_discrete_map={"未发生": "#4C78A8", "发生": "#E45756"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='hint'>分组脑卒中率属于描述性关联，不能直接解释为因果关系。年龄、慢性病和生活方式之间可能存在共同影响。</div>",
        unsafe_allow_html=True,
    )


def page_model() -> None:
    st.subheader("模型评估")
    if not METRICS_PATH.exists():
        st.error("未找到模型结果，请先运行 python analysis.py")
        return
    metrics = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    metric_long = metrics.melt(
        id_vars="模型",
        value_vars=["准确率", "精确率", "召回率", "F1值", "ROC-AUC", "PR-AUC"],
        var_name="评价指标",
        value_name="指标值",
    )
    fig = px.bar(
        metric_long,
        x="模型",
        y="指标值",
        color="评价指标",
        barmode="group",
        title="分类模型综合指标对比",
    )
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    main = metrics[metrics["模型"] == "随机森林"].iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("主模型召回率", f"{main['召回率']:.2%}")
    c2.metric("主模型ROC-AUC", f"{main['ROC-AUC']:.3f}")
    c3.metric("主模型PR-AUC", f"{main['PR-AUC']:.3f}")
    c4.metric("判断阈值", f"{main['验证集最优阈值']:.3f}")

    left, right = st.columns(2)
    with left:
        matrix = np.array([[main["TN"], main["FP"]], [main["FN"], main["TP"]]])
        fig = px.imshow(
            matrix,
            text_auto=True,
            x=["预测未发生", "预测发生"],
            y=["实际未发生", "实际发生"],
            color_continuous_scale="Blues",
            title="随机森林混淆矩阵",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if FEATURE_PATH.exists():
            feature = pd.read_csv(FEATURE_PATH).head(12).sort_values("重要性")
            fig = px.bar(
                feature,
                x="重要性",
                y="特征中文",
                orientation="h",
                title="随机森林Top12特征重要性",
            )
            fig.update_traces(marker_color="#8F63B8")
            st.plotly_chart(fig, use_container_width=True)

    st.info("由于正类仅占约5%，准确率不能单独反映模型质量。本项目重点观察召回率、F1、ROC-AUC和PR-AUC。")


def page_prediction() -> None:
    st.subheader("个体风险演示")
    try:
        model, metadata = load_model()
    except Exception as exc:
        st.error(str(exc))
        return

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("性别", ["Female", "Male"], format_func=lambda x: "女性" if x == "Female" else "男性")
            age = st.slider("年龄", 0.0, 82.0, 45.0, 1.0)
            ever_married = st.selectbox("是否结婚", ["No", "Yes"], format_func=lambda x: "否" if x == "No" else "是")
        with c2:
            hypertension = st.selectbox("是否有高血压", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
            heart_disease = st.selectbox("是否有心脏病", [0, 1], format_func=lambda x: "否" if x == 0 else "是")
            avg_glucose_level = st.number_input("平均血糖水平", min_value=40.0, max_value=300.0, value=100.0, step=1.0)
        with c3:
            bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            work_type = st.selectbox(
                "工作类型",
                ["Private", "Self-employed", "Govt_job", "children"],
                format_func=lambda x: {"Private": "私营企业", "Self-employed": "个体经营", "Govt_job": "政府工作", "children": "儿童"}[x],
            )
            residence = st.selectbox("居住类型", ["Urban", "Rural"], format_func=lambda x: "城市" if x == "Urban" else "农村")
            smoking = st.selectbox(
                "吸烟状态",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
                format_func=lambda x: {"never smoked": "从不吸烟", "formerly smoked": "曾经吸烟", "smokes": "目前吸烟", "Unknown": "未知"}[x],
            )
        submitted = st.form_submit_button("开始分析", use_container_width=True)

    if submitted:
        sample = pd.DataFrame(
            [
                {
                    "age": age,
                    "avg_glucose_level": avg_glucose_level,
                    "bmi": bmi,
                    "hypertension": hypertension,
                    "heart_disease": heart_disease,
                    "gender": gender,
                    "ever_married": ever_married,
                    "work_type": work_type,
                    "Residence_type": residence,
                    "smoking_status": smoking,
                }
            ]
        )
        probability = float(model.predict_proba(sample)[0, 1])
        threshold = float(metadata["threshold"])
        result = "高于模型筛查阈值" if probability >= threshold else "低于模型筛查阈值"
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": "模型输出概率"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#E45756" if probability >= threshold else "#4C78A8"},
                    "threshold": {
                        "line": {"color": "black", "width": 4},
                        "thickness": 0.8,
                        "value": threshold * 100,
                    },
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"### 结果：{result}")
        st.write(f"模型阈值为 {threshold:.3f}。该结果用于演示机器学习流程，不等同于临床诊断或真实发病概率。")

        factors = []
        if age >= 65:
            factors.append("年龄较高")
        if hypertension == 1:
            factors.append("存在高血压")
        if heart_disease == 1:
            factors.append("存在心脏病")
        if avg_glucose_level >= 140:
            factors.append("平均血糖水平偏高")
        if bmi >= 28:
            factors.append("BMI偏高")
        if smoking in {"formerly smoked", "smokes"}:
            factors.append("存在吸烟史")
        if factors:
            st.write("输入信息中需要关注的因素：" + "、".join(factors) + "。")
        else:
            st.write("输入信息中未出现本演示规则标记的明显高风险因素。")

    st.warning("重要提示：本页面仅用于数据可视化课程展示，不能用于自我诊断、治疗决策或替代医生建议。")


def main() -> None:
    show_header()
    data = load_data()
    page = st.sidebar.radio("功能导航", ["数据总览", "因素分析", "模型评估", "个体风险演示"])
    st.sidebar.markdown("---")
    st.sidebar.caption("数据集：Brain Stroke Dataset（4,981条记录）")

    if page in {"数据总览", "因素分析"}:
        filtered = filter_data(data)
    else:
        filtered = data

    if page == "数据总览":
        page_overview(filtered)
    elif page == "因素分析":
        page_factors(filtered)
    elif page == "模型评估":
        page_model()
    else:
        page_prediction()


if __name__ == "__main__":
    main()
