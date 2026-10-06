import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 페이지 설정
# ==========================================
st.set_page_config(
    page_title="서울 연평균 기온 선형회귀",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 분석")
st.write(
    "과거 기온 데이터를 이용해 선형회귀 모델을 만들고 "
    "최근 20년의 기온을 얼마나 잘 예측하는지 비교합니다."
)

# ==========================================
# 데이터 불러오기
# ==========================================
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="cp949")

    # 날짜 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 결측값 제거
    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온 계산
    yearly = (
        df.groupby("연도")["평균기온"]
        .mean()
        .reset_index()
    )

    yearly.columns = ["연도", "연평균기온"]

    return yearly


df = load_data()

# ==========================================
# 분석에 사용할 기간 확인
# ==========================================
df = df[
    (df["연도"] >= 1906) &
    (df["연도"] <= 2025)
].copy()

# ==========================================
# 데이터 분리
# ==========================================
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

# ==========================================
# 선형회귀 함수
# ==========================================
def make_model(train_df):
    X = train_df[["연도"]]
    y = train_df["연평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


model_50 = make_model(train_50)
model_100 = make_model(train_100)

# ==========================================
# 테스트 데이터 예측
# ==========================================
X_test = test[["연도"]]
y_test = test["연평균기온"]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)

# ==========================================
# 평가 함수
# ==========================================
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, r2


mae_50, mse_50, r2_50 = evaluate_model(y_test, pred_50)
mae_100, mse_100, r2_100 = evaluate_model(y_test, pred_100)

# ==========================================
# 회귀식
# ==========================================
slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_

# ==========================================
# 데이터 개수 표시
# ==========================================
st.subheader("📊 데이터 구성")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 훈련",
        f"{len(train_50)}년",
        "1956~2005"
    )

with col2:
    st.metric(
        "최근 100년 훈련",
        f"{len(train_100)}년",
        "1906~2005"
    )

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}년",
        "2006~2025"
    )

# ==========================================
# 회귀선 기울기 비교
# ==========================================
st.subheader("📈 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "1956~2005년 회귀선 기울기",
        f"{slope_50:.4f} ℃/년"
    )
    st.write(
        f"회귀식: "
        f"**y = {slope_50:.4f} × 연도 + {intercept_50:.2f}**"
    )

with col2:
    st.metric(
        "1906~2005년 회귀선 기울기",
        f"{slope_100:.4f} ℃/년"
    )
    st.write(
        f"회귀식: "
        f"**y = {slope_100:.4f} × 연도 + {intercept_100:.2f}**"
    )

# ==========================================
# 전체 데이터 + 회귀선 그래프
# ==========================================
st.subheader("🌡️ 실제 연평균 기온과 두 회귀선")

fig = go.Figure()

# 실제 데이터
fig.add_trace(
    go.Scatter(
        x=df["연도"],
        y=df["연평균기온"],
        mode="lines+markers",
        name="실제 연평균 기온"
    )
)

# 50년 회귀선
years_all = np.arange(1906, 2026)

pred_line_50 = model_50.predict(
    pd.DataFrame({"연도": years_all})
)

fig.add_trace(
    go.Scatter(
        x=years_all,
        y=pred_line_50,
        mode="lines",
        name="1956~2005 회귀선"
    )
)

# 100년 회귀선
pred_line_100 = model_100.predict(
    pd.DataFrame({"연도": years_all})
)

fig.add_trace(
    go.Scatter(
        x=years_all,
        y=pred_line_100,
        mode="lines",
        name="1906~2005 회귀선"
    )
)

# 테스트 구간 표시
fig.add_vrect(
    x0=2006,
    x1=2025,
    fillcolor="gray",
    opacity=0.15,
    line_width=0,
    annotation_text="테스트 데이터 2006~2025",
    annotation_position="top left"
)

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 테스트 데이터 예측 그래프
# ==========================================
st.subheader("🔮 테스트 데이터(2006~2025) 예측 비교")

fig2 = go.Figure()

# 실제값
fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=test["연평균기온"],
        mode="lines+markers",
        name="실제 기온"
    )
)

# 50년 모델
fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_50,
        mode="lines",
        name="50년 학습 모델 예측"
    )
)

# 100년 모델
fig2.add_trace(
    go.Scatter(
        x=test["연도"],
        y=pred_100,
        mode="lines",
        name="100년 학습 모델 예측"
    )
)

fig2.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균 기온 (℃)",
    hovermode="x unified",
    height=500
)

st.plotly_chart(fig2, use_container_width=True)

# ==========================================
# 모델 성능 비교
# ==========================================
st.subheader("🎯 테스트 데이터 예측 성능 비교")

result = pd.DataFrame({
    "모델": [
        "1956~2005년 학습 (50년)",
        "1906~2005년 학습 (100년)"
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    result.style.format({
        "MAE (℃)": "{:.4f}",
        "MSE (℃²)": "{:.4f}",
        "R²": "{:.4f}"
    }),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# 가장 좋은 모델 자동 판단
# ==========================================
st.subheader("🏆 모델 성능 해석")

if mae_50 < mae_100:
    better_mae = "1956~2005년 50년 모델"
else:
    better_mae = "1906~2005년 100년 모델"

if mse_50 < mse_100:
    better_mse = "1956~2005년 50년 모델"
else:
    better_mse = "1906~2005년 100년 모델"

if r2_50 > r2_100:
    better_r2 = "1956~2005년 50년 모델"
else:
    better_r2 = "1906~2005년 100년 모델"

st.write(
    f"""
    - **MAE가 더 낮은 모델:** {better_mae}
    - **MSE가 더 낮은 모델:** {better_mse}
    - **R²가 더 높은 모델:** {better_r2}
    """
)

# ==========================================
# 회귀선 기울기 차이
# ==========================================
slope_difference = slope_50 - slope_100

st.write(
    f"""
    ### 기울기 해석

    - 50년 모델 기울기: **{slope_50:.4f} ℃/년**
    - 100년 모델 기울기: **{slope_100:.4f} ℃/년**
    - 기울기 차이: **{slope_difference:.4f} ℃/년**

    기울기가 양수라면 시간이 지날수록 연평균 기온이 상승하는
    경향이 있다는 뜻입니다.
    """
)

# ==========================================
# 데이터 표
# ==========================================
with st.expander("📋 연도별 연평균 기온 데이터 보기"):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )
