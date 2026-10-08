from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from src.prediction import predict_risk
from src.data_analysis import load_data, feature_importance, create_feature_importance_chart
from src.weather_api import PROVINCES, fetch_weather

st.set_page_config(page_title="KKU Smart Flood AI", page_icon="🌊", layout="wide")
st.title("🌊 KKU Smart Flood AI")
st.subheader("ระบบ AI สำหรับประเมินความเสี่ยงน้ำท่วมจากข้อมูลสภาพแวดล้อม")
st.info("Prototype เพื่อการศึกษา ใช้ Synthetic Dataset สำหรับการฝึกโมเดล แต่ข้อมูลอากาศของจังหวัดที่เลือกดึงจาก API แบบอัตโนมัติ")

if not Path("models/flood_model.pkl").exists():
    st.error("ไม่พบโมเดล กรุณารัน generate_data.py และ train_model.py ก่อน")
    st.stop()

bundle = joblib.load("models/flood_model.pkl")
df = load_data()
if "history" not in st.session_state:
    st.session_state.history = []

st.header("1. เลือกจังหวัดและอัปเดตข้อมูล API")
province = st.selectbox(
    "จังหวัดของพื้นที่ที่ต้องการวิเคราะห์ (77 จังหวัด)",
    list(PROVINCES.keys()),
    index=list(PROVINCES.keys()).index("ขอนแก่น")
)

api_col, water_col = st.columns([2, 1])
with api_col:
    st.caption("แหล่งข้อมูลอากาศ: Open-Meteo API — ไม่ต้องใช้ API Key")
    refresh = st.button("🔄 อัปเดตข้อมูลจาก API", use_container_width=True)
with water_col:
    water = st.number_input(
        "ระดับน้ำ (m)", 0.05, 2.50, 0.80, 0.05,
        help="กรอกค่าจากสถานีตรวจวัดหรือแหล่งข้อมูลน้ำที่เชื่อถือได้ ระบบจะไม่เดาระดับน้ำจากข้อมูลอากาศ"
    )

if refresh or "weather" not in st.session_state or st.session_state.get("weather_province") != province:
    try:
        with st.spinner(f"กำลังดึงข้อมูลล่าสุดของ {province}..."):
            st.session_state.weather = fetch_weather(province)
            st.session_state.weather_province = province
            st.session_state.weather_error = None
    except Exception as exc:
        st.session_state.weather_error = str(exc)

if st.session_state.get("weather_error"):
    st.error("ดึงข้อมูล API ไม่สำเร็จ: " + st.session_state.weather_error)

weather = st.session_state.get("weather")
if weather:
    st.success(f"อัปเดตล่าสุด: {weather['updated_at']} | พิกัดตัวแทนจังหวัด {weather['latitude']:.4f}, {weather['longitude']:.4f}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ฝนสะสม 24 ชม.", f"{weather['rainfall_24h']:.2f} mm")
    c2.metric("อุณหภูมิ", f"{weather['temperature']:.1f} °C")
    c3.metric("ความชื้น", f"{weather['humidity']:.0f} %")
    c4.metric("ความเร็วลม", f"{weather['wind_speed']:.1f} km/h")

    rainfall = min(max(weather["rainfall_24h"], 0.0), 250.0)
    humidity = min(max(weather["humidity"], 35.0), 100.0)
    temperature = min(max(weather["temperature"], 18.0), 40.0)
else:
    st.warning("ยังไม่มีข้อมูลอากาศจาก API")
    rainfall, humidity, temperature = 0.0, 78.0, 29.0

st.caption("ข้อมูลฝน/อุณหภูมิ/ความชื้นมาจาก API ส่วนระดับน้ำต้องใช้ข้อมูลจากสถานีตรวจวัดหรือกรอกเอง")

if st.button("🔎 วิเคราะห์ความเสี่ยง", type="primary", use_container_width=True):
    prediction, probabilities = predict_risk(rainfall, water, humidity, temperature)
    st.session_state.prediction = prediction
    st.session_state.probabilities = probabilities
    st.session_state.history.insert(0, {
        "จังหวัด": province,
        "Rainfall 24h (mm)": rainfall,
        "Water Level (m)": water,
        "Humidity (%)": humidity,
        "Temperature (°C)": temperature,
        "Prediction": prediction,
        "Confidence (%)": round(probabilities[prediction] * 100, 2),
    })

if "prediction" in st.session_state:
    prediction = st.session_state.prediction
    probs = st.session_state.probabilities
    labels = {"LOW": "🟢 ความเสี่ยงต่ำ", "MEDIUM": "🟡 ความเสี่ยงปานกลาง", "HIGH": "🔴 ความเสี่ยงสูง"}
    st.header("2. ผลการวิเคราะห์ AI")
    st.success(f"ผลการประเมิน: {labels[prediction]}")
    a, b, c = st.columns(3)
    a.metric("LOW", f"{probs.get('LOW', 0) * 100:.2f}%")
    b.metric("MEDIUM", f"{probs.get('MEDIUM', 0) * 100:.2f}%")
    c.metric("HIGH", f"{probs.get('HIGH', 0) * 100:.2f}%")

st.header("3. AI Model Performance")
a, b, c = st.columns(3)
a.metric("Model Accuracy", f"{bundle['accuracy'] * 100:.2f}%")
b.metric("Decision Trees", bundle["model"].n_estimators)
c.metric("Training Dataset", len(df))
st.caption("Accuracy นี้มาจาก Synthetic Dataset จึงไม่ใช่ความแม่นยำของระบบพยากรณ์น้ำท่วมจริง")

st.header("4. AI ให้ความสำคัญกับปัจจัยใดมากที่สุด?")
imp = feature_importance()
st.dataframe(imp.style.format({"Importance": "{:.4f}"}), use_container_width=True, hide_index=True)
if not Path("images/feature_importance.png").exists():
    create_feature_importance_chart()
st.image("images/feature_importance.png", use_container_width=True)

st.header("5. วิเคราะห์ข้อมูล")
x, y = st.columns(2)
with x:
    st.subheader("Training Datasetแต่ละระดับ")
    st.bar_chart(df["Risk"].value_counts())
with y:
    st.subheader("ตัวอย่างข้อมูล")
    st.dataframe(df.head(10), use_container_width=True)

st.header("6. ประวัติการวิเคราะห์")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("ล้างประวัติ"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("ยังไม่มีประวัติ")

st.header("7. แหล่งข้อมูลและข้อจำกัด")
st.markdown("""
**Open-Meteo** ใช้สำหรับข้อมูลสภาพอากาศ เช่น อุณหภูมิ ความชื้น ฝน และลม โดยไม่ต้องใช้ API Key

**ThaiWater** มีมาตรฐาน API สำหรับข้อมูลฝน น้ำท่า แหล่งน้ำ และสถานีตรวจวัด ซึ่งสามารถนำมาต่อยอดเชื่อมข้อมูลระดับน้ำจริงในอนาคต

**ข้อจำกัดสำคัญ:** โมเดลปัจจุบันฝึกด้วย Synthetic Dataset ดังนั้นผลทำนายยังเป็น Prototype ไม่ใช่ระบบเตือนภัยน้ำท่วมจริง
""")
st.warning("⚠️ ระบบนี้ไม่ได้ใช้ผลทำนายเพื่อการตัดสินใจด้านความปลอดภัยในสถานการณ์จริง")
