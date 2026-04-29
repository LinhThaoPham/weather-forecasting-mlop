# services/dashboard_ui/app.py
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import os

FORECAST_API_URL = os.environ.get("FORECAST_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Dự Báo Thời Tiết Việt Nam", layout="wide")
st.title("⛅ Dự Báo Thời Tiết Việt Nam")
st.markdown("**Hybrid Prophet + LSTM** | Real-time từ Open-Meteo")

# City selector
city = st.selectbox(
    "🏙️ Chọn thành phố",
    options=["hanoi", "hcm", "danang", "haiphong", "nhatrang", "dalat"],
    format_func=lambda x: {
        "hanoi": "Hà Nội",
        "hcm": "TP. Hồ Chí Minh",
        "danang": "Đà Nẵng",
        "haiphong": "Hải Phòng",
        "nhatrang": "Nha Trang",
        "dalat": "Đà Lạt"
    }.get(x, x)
)

# Tab giao diện
tab1, tab2 = st.tabs(["📍 Theo giờ hôm nay", "📅 Trung bình 3 ngày tới"])

# ================== TAB 1: THEO GIỜ HÔM NAY ==================
with tab1:
    st.subheader("Thời tiết theo giờ (72 giờ tới)")

    try:
        response = requests.post(
            f"{FORECAST_API_URL}/predict",
            json={"hours": 72, "mode": "hourly", "city": city}
        )
        data = response.json()["data"]
        df_hourly = pd.DataFrame(data)

        df_hourly['Giờ'] = pd.to_datetime(df_hourly['ds']).dt.strftime('%d/%m %H:%M')

        st.line_chart(df_hourly.set_index('Giờ')['final_pred'], use_container_width=True)

        st.dataframe(
            df_hourly[['Giờ', 'final_pred']].style.format({"final_pred": "{:.1f}°C"}),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Không kết nối được với forecast_api. Hãy chạy forecast_api trước.\nLỗi: {e}")

# ================== TAB 2: TRUNG BÌNH 3 NGÀY TỚI ==================
with tab2:
    st.subheader("Nhiệt độ trung bình 3 ngày tới")

    try:
        response = requests.post(
            f"{FORECAST_API_URL}/predict",
            json={"hours": 72, "mode": "daily", "city": city}
        )
        data = response.json()["data"]
        df_daily = pd.DataFrame(data)

        st.bar_chart(df_daily.set_index('ds')['final_pred'], use_container_width=True)

        st.dataframe(
            df_daily.style.format({"final_pred": "{:.1f}°C"}),
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Không kết nối được với forecast_api. Hãy chạy forecast_api trước.\nLỗi: {e}")

st.caption("Hệ thống dự báo thời tiết Việt Nam - Hybrid Prophet + LSTM")