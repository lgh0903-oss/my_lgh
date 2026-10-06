import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="서울 연평균 기온 선형회귀",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 비교")

st.write(
    "1956~2005년과 1906~2005년의 데이터를 각각 학습하고 "
    "2006~2025년을 공통 테스트 데이터로 사용하여 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_URL, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")["평균기온"]
        .mean()
        .reset_index()
    )

    yearly.columns = ["연도", "연평균기온"]

    return yearly


df = load_data()

df = df[
    (df["연도"] >= 1906) &
    (df["연도"] <= 2025)
].copy()

train_50 = df[
    (df["연도"] >= 1956) &
    (df["연도"] <= 2005)
].copy()

train_100 = df[
    (df["연도"] >= 1906) &
    (df["연도"] <= 2005)
].copy()

test = df[
    (df["연도"] >= 2006) &
    (df["연도"] <= 2025)
].copy()

if len(train_50) == 0 or len(train_100) == 0 or len(test) == 0:
    st.error("분석에 필요한 연도별 데이터가 부족합니다.")
    st.stop()

model_50 = LinearRegression()
model_50.fit(
    train_50[["연도"]],
    train_50["연평균기온"]
)

model_100 = LinearRegression()
model_100.fit(
    train_100[["연도"]],
    train_100["연평균기온"]
)

X_test = test[["연도"]]
y_test = test["연평균기온"]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)

mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_


# --------------------------------------------
# 1. 데이터 구성
# --------------------------------------------

st.header("1. 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("50년 훈련 데이터", f"{len(train_50)}년")
    st.write("
