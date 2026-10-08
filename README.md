# 🌊 KKU Smart Flood AI

AI Prototype สำหรับประเมินความเสี่ยงน้ำท่วมจาก Rainfall, Water Level, Humidity และ Temperature พร้อมเลือกจังหวัดไทย 77 จังหวัดและอัปเดตข้อมูลอากาศผ่าน API

## Live Weather API

ระบบเลือกพื้นที่ได้ครบ 77 จังหวัด โดยใช้พิกัดตัวแทนของจังหวัดเพื่อเรียก **Open-Meteo API** แบบไม่ต้องใช้ API Key ข้อมูลที่ดึงอัตโนมัติ ได้แก่ อุณหภูมิ ความชื้น ความเร็วลม และฝนสะสม 24 ชั่วโมง

ระดับน้ำไม่ถูกสร้างขึ้นจากการเดาข้อมูลอากาศ ระบบให้กรอกค่าจากสถานีตรวจวัดหรือแหล่งข้อมูลน้ำที่เชื่อถือได้ เพื่อไม่ทำให้ข้อมูลปลอมกลายเป็นข้อมูลจริง

ในขั้นต่อไปสามารถเชื่อม **ThaiWater API** เพื่อดึงข้อมูลฝน น้ำท่า และสถานีตรวจวัดภาครัฐได้ โดยมาตรฐาน ThaiWater ระบุ resource เช่น `/Rainfall`, `/Runoff` และ `/StationInfo`

## ⚠️ Disclaimer

โมเดล Machine Learning ใช้ **Synthetic Dataset** เพื่อการศึกษาและสาธิต ดังนั้น Accuracy ไม่ควรนำไปอ้างว่าเป็นความแม่นยำของระบบพยากรณ์น้ำท่วมจริง

## Features

- เลือกจังหวัดไทย 77 จังหวัด
- ดึงข้อมูลอากาศอัตโนมัติ
- ฝนสะสม 24 ชั่วโมง
- Random Forest Classification
- LOW / MEDIUM / HIGH risk
- Probability ของแต่ละระดับ
- Feature Importance
- Data Analysis
- Prediction History
- Streamlit Dashboard

## Run

```bash
pip install -r requirements.txt
python generate_data.py
python train_model.py
python src/data_analysis.py
streamlit run app.py
```

## Future Development

- เชื่อม ThaiWater/หน่วยงานภาครัฐสำหรับระดับน้ำจริง
- เพิ่มข้อมูลพยากรณ์ฝนล่วงหน้า
- เพิ่มแผนที่พื้นที่เสี่ยง
- ระบบแจ้งเตือน
- เปรียบเทียบโมเดลหลายชนิด

## Data Sources

- Open-Meteo: https://open-meteo.com/
- ThaiWater API Standard: https://standard.thaiwater.net/
