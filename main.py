```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================
# 페이지 설정
# ============================================

st.set_page_config(
    page_title="서울 연평균 기온 선형회귀",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 모델 비교")

st.write(
    "서울의 연평균 기온 데이터를 이용하여 "
    "최근 50년과 최근 100년의 선형회귀 모델을 학습하고, "
    "공통 테스트 데이터인 최근 20년의 기온을 얼마나 잘 예측하는지 비교합니다."
)


# ============================================
# 데이터 주소
# ============================================

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# ============================================
# 데이터 불러오기
# ============================================

@st.cache_data
def load_data():

    # UTF-8 계열로 먼저 읽고,
    # 혹시 문제가 있으면 다른 인코딩으로 다시 시도
    try:
        df = pd.read_csv(
            DATA_URL,
            encoding="utf-8-sig"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            DATA_URL,
            encoding="utf-8"
        )

    # 날짜 변환
    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온 계산
    yearly = (
        df.groupby("연도")["평균기온"]
        .mean()
        .reset_index()
    )

    yearly.columns = [
        "연도",
        "연평균기온"
    ]

    return yearly


# 데이터 불러오기
df = load_data()


# ============================================
# 1906~2025년 데이터만 사용
# ============================================

df = df[
    (df["연도"] >= 1906) &
    (df["연도"] <= 2025)
].copy()


# ============================================
# 데이터 나누기
# ============================================

# 최근 50년 훈련 데이터
train_50 = df[
    (df["연도"] >= 1956) &
    (df["연도"] <= 2005)
].copy()


# 최근 100년 훈련 데이터
train_100 = df[
    (df["연도"] >= 1906) &
    (df["연도"] <= 2005)
].copy()


# 공통 테스트 데이터
test = df[
    (df["연도"] >= 2006) &
    (df["연도"] <= 2025)
].copy()


# ============================================
# 데이터가 제대로 있는지 확인
# ============================================

if len(train_50) == 0:
    st.error("1956~2005년 훈련 데이터가 없습니다.")
    st.stop()

if len(train_100) == 0:
    st.error("1906~2005년 훈련 데이터가 없습니다.")
    st.stop()

if len(test) == 0:
    st.error("2006~2025년 테스트 데이터가 없습니다.")
    st.stop()


# ============================================
# 선형회귀 모델 만들기
# ============================================

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


# ============================================
# 테스트 데이터 예측
# ============================================

X_test = test[["연도"]]
y_test = test["연평균기온"]


pred_50 = model_50.predict(X_test)

pred_100 = model_100.predict(X_test)


# ============================================
# 평가 지표
# ============================================

mae_50 = mean_absolute_error(
    y_test,
    pred_50
)

mse_50 = mean_squared_error(
    y_test,
    pred_50
)

r2_50 = r2_score(
    y_test,
    pred_50
)


mae_100 = mean_absolute_error(
    y_test,
    pred_100
)

mse_100 = mean_squared_error(
    y_test,
    pred_100
)

r2_100 = r2_score(
    y_test,
    pred_100
)


# ============================================
# 회귀선 기울기와 절편
# ============================================

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_


# ============================================
# 1. 데이터 구성
# ============================================

st.header("1. 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 훈련 데이터",
        f"{len(train_50)}년"
    )
    st.write("1956~2005년")


with col2:
    st.metric(
        "최근 100년 훈련 데이터",
        f"{len(train_100)}년"
    )
    st.write("1906~2005년")


with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test)}년"
    )
    st.write("2006~2025년")


# ============================================
# 2. 회귀선 기울기 비교
# ============================================

st.header("2. 회귀선의 기울기 비교")

col1, col2 = st.columns
