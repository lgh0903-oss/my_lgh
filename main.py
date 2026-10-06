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
    "서울의 연평균 기온 데이터를 이용하여 "
    "50년 학습 모델과 100년 학습 모델을 비교합니다."
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(
        df["날짜"],
        errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    )

    df["연도"] = df["날짜"].dt.year

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


df = load_data()

df = df[
    (df["연도"] >= 1906)
    & (df["연도"] <= 2025)
].copy()


# -----------------------------
# 데이터 나누기
# -----------------------------

train_50 = df[
    (df["연도"] >= 1956)
    & (df["연도"] <= 2005)
].copy()

train_100 = df[
    (df["연도"] >= 1906)
    & (df["연도"] <= 2005)
].copy()

test = df[
    (df["연도"] >= 2006)
    & (df["연도"] <= 2025)
].copy()


if len(train_50) == 0:
    st.error("1956~2005년 데이터가 없습니다.")
    st.stop()

if len(train_100) == 0:
    st.error("1906~2005년 데이터가 없습니다.")
    st.stop()

if len(test) == 0:
    st.error("2006~2025년 데이터가 없습니다.")
    st.stop()


# -----------------------------
# 선형회귀 모델 만들기
# -----------------------------

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


# -----------------------------
# 테스트 데이터 예측
# -----------------------------

X_test = test[["연도"]]
y_test = test["연평균기온"]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# -----------------------------
# 평가 지표
# -----------------------------

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


# -----------------------------
# 기울기
# -----------------------------

slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_


# ==================================================
# 1. 데이터 구성
# ==================================================

st.header("1. 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "50년 훈련 데이터",
        f"{len(train_50)}년"
    )
    st.write("1956~2005년")

with col2:
    st.metric(
        "100년 훈련 데이터",
        f"{len(train_100)}년"
    )
    st.write("1906~2005년")

with col3:
    st.metric(
        "공통 테스트 데이터",
        f"{len(test)}년"
    )
    st.write("2006~2025년")


# ==================================================
# 2. 회귀선 기울기
# ==================================================

st.header("2. 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 최근 50년 모델")

    st.metric(
        "기울기",
        f"{slope_50:.4f} ℃/년"
    )

    st.write(
        f"회귀식: y = {slope_50:.4f} × 연도 + {intercept_50:.2f}"
    )


with col2:
    st.subheader("📈 최근 100년 모델")

    st.metric(
        "기울기",
        f"{slope_100:.4f} ℃/년"
    )

    st.write(
        f"회귀식: y = {slope_100:.4f} × 연도 + {intercept_100:.2f}"
    )


# ==================================================
# 3. 전체 기간 그래프
# ==================================================

st.header("3. 실제 연평균 기온과 회귀선")

years = np.arange(
    1906,
    2026
)

line_50 = model_50.predict(
    pd.DataFrame({"연도": years})
)

line_100 = model_100.predict(
    pd.DataFrame({"연도": years})
)


fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["연평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온"
    )
)


fig.add_trace(
    go.Scatter(
        x=years,
        y=line_50,
        mode="lines",
        name="1956~2005 회귀선"
    )
)


fig.add_trace(
    go.Scatter(
        x=years,
        y=line_100,
        mode="lines",
        name="1906~2005 회귀선"
    )
)


fig.add_vrect(
    x0=2006,
    x1=2025,
    opacity=0.15,
    line_width=0,
    annotation_text="테스트 기간"
)


fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=550
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ==================================================
# 4. 테스트 데이터 예측 그래프
# ==================================================

st.header("4. 2006~2025년 실제 기온과 예측값")

fig2 = go.Figure()


fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)


fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines+markers",
        name="50년 모델 예측"
    )
)


fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines+markers",
        name="100년 모델 예측"
    )
)


fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=500
)


st.plotly_chart(
    fig2,
    use_container_width=True
)


# ==================================================
# 5. 평가 결과
# ==================================================

st.header("5. 최근 20년 테스트 성능 비교")

result = pd.DataFrame(
    {
        "모델": [
            "최근 50년 학습",
            "최근 100년 학습"
        ],
        "훈련 기간": [
            "1956~2005",
            "1906~2005"
        ],
        "기울기": [
            slope_50,
            slope_100
        ],
        "MAE": [
            mae_50,
            mae_100
        ],
        "MSE": [
            mse_50,
            mse_100
        ],
        "R²": [
            r2_50,
            r2_100
        ]
    }
)


st.dataframe(
    result.style.format(
        {
            "기울기": "{:.4f}",
            "MAE": "{:.4f}",
            "MSE": "{:.4f}",
            "R²": "{:.4f}"
        }
    ),
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 6. 어떤 모델이 더 좋은가?
# ==================================================

st.header("6. 어떤 모델이 더 잘 예측했을까?")

if mae_50 < mae_100:
    mae_result = "50년 모델"
else:
    mae_result = "100년 모델"


if mse_50 < mse_100:
    mse_result = "50년 모델"
else:
    mse_result = "100년 모델"


if r2_50 > r2_100:
    r2_result = "50년 모델"
else:
    r2_result = "100년 모델"


col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "MAE가 더 좋은 모델",
        mae_result
    )


with col2:
    st.metric(
        "MSE가 더 좋은 모델",
        mse_result
    )


with col3:
    st.metric(
        "R²가 더 좋은 모델",
        r2_result
    )


# ==================================================
# 7. 기울기 차이
# ==================================================

st.header("7. 두 회귀선의 기울기 차이")

slope_difference = (
    slope_50 - slope_100
)

st.write(
    f"50년 모델 기울기: {slope_50:.4f} ℃/년"
)

st.write(
    f"100년 모델 기울기: {slope_100:.4f} ℃/년"
)

st.write(
    f"기울기 차이: {slope_difference:.4f} ℃/년"
)


if slope_50 > slope_100:

    st.info(
        "50년 모델의 기울기가 더 큽니다. "
        "1956~2005년을 사용한 모델에서 "
        "기온 상승 추세가 더 가파르게 나타납니다."
    )

elif slope_50 < slope_100:

    st.info(
        "100년 모델의 기울기가 더 큽니다. "
        "1906~2005년을 사용한 모델에서 "
        "기온 상승 추세가 더 가파르게 나타납니다."
    )

else:

    st.info(
        "두 모델의 기울기가 같습니다."
    )


# ==================================================
# 8. 평가 지표 설명
# ==================================================

st.header("8. 평가 지표 해석")

st.write(
    "MAE는 실제값과 예측값의 평균적인 차이를 나타냅니다. "
    "작을수록 예측이 정확합니다."
)

st.write(
    "MSE는 오차를 제곱하여 평균한 값입니다. "
    "큰 오차에 더 민감하며 작을수록 좋습니다."
)

st.write(
    "R²는 모델이 실제 기온의 변화를 얼마나 잘 설명하는지 "
    "나타냅니다. 일반적으로 1에 가까울수록 좋습니다."
)


# ==================================================
# 9. 최종 정리
# ==================================================

st.header("9. 최종 결과")

st.success(
    "두 모델 모두 동일한 2006~2025년 데이터를 "
    "테스트 데이터로 사용했습니다."
)

st.write(
    f"50년 모델의 MAE는 {mae_50:.4f} ℃입니다."
)

st.write(
    f"100년 모델의 MAE는 {mae_100:.4f} ℃입니다."
)

st.write(
    f"50년 모델의 R²는 {r2_50:.4f}입니다."
)

st.write(
    f"100년 모델의 R²는 {r2_100:.4f}입니다."
)


# ==================================================
# 10. 데이터 확인
# ==================================================

with st.expander("📋 연도별 연평균 기온 데이터 보기"):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
