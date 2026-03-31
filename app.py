import streamlit as st
import cv2
from deepface import DeepFace
import numpy as np
from PIL import Image

# --- ฟังก์ชันตรวจหาจำนวนกล้องที่มีในเครื่อง ---
def get_available_cameras():
    index = 0
    arr = []
    # ตรวจสอบกล้องสูงสุด 5 ตัว (ปรับเพิ่มได้)
    while index < 5:
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            arr.append(index)
            cap.release()
        index += 1
    return arr

# การตั้งค่าหน้าเว็บ
st.set_page_config(page_title="AI Face & Emotion Detector", layout="wide")

st.title("🎭 AI Face & Emotion Detector")
st.markdown("---")

# --- Sidebar: ตั้งค่ากล้องและโมเดล ---
st.sidebar.header("🛠️ การตั้งค่าระบบ")

# ส่วนการเลือกกล้อง
available_cams = get_available_cameras()
if not available_cams:
    st.sidebar.error("❌ ไม่พบกล้องที่เชื่อมต่ออยู่")
    selected_cam_index = 0
else:
    selected_cam_index = st.sidebar.selectbox(
        "เลือกกล้อง (Camera Source)",
        available_cams,
        format_func=lambda x: f"Camera {x}"
    )

# ส่วนการเลือก Detector
detector_backend = st.sidebar.selectbox(
    "เลือกตัวตรวจจับใบหน้า (Detector)",
    ["opencv", "mediapipe", "retinaface", "mtcnn"],
    index=1 # mediapipe จะสมดุลที่สุดสำหรับ Live
)

st.sidebar.info("คำแนะนำ: 'mediapipe' จะทำงานได้ลื่นไหลที่สุดสำหรับกล้องสด")

# --- ฟังก์ชันประมวลผล ---
def analyze_frame(frame):
    try:
        results = DeepFace.analyze(
            frame, 
            actions=['emotion'], 
            detector_backend=detector_backend,
            enforce_detection=False
        )
        
        for res in results:
            x, y, w, h = res['region']['x'], res['region']['y'], res['region']['w'], res['region']['h']
            emotion = res['dominant_emotion']
            
            # วาดกรอบสี่เหลี่ยม (สีเขียว)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # ใส่ข้อความอารมณ์
            cv2.putText(frame, emotion.upper(), (x, y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
        return frame
    except:
        return frame

# --- ส่วนแสดงผลหน้าเว็บ ---
col1, col2 = st.columns([2, 1])

with col1:
    option = st.radio("โหมดการใช้งาน:", ("Live Webcam", "Upload Image"), horizontal=True)

    if option == "Live Webcam":
        run = st.checkbox('🔴 เริ่มการทำงานของกล้อง')
        FRAME_WINDOW = st.image([])
        
        if run:
            cam = cv2.VideoCapture(selected_cam_index)
            while run:
                ret, frame = cam.read()
                if not ret:
                    st.error("ไม่สามารถดึงภาพจากกล้องได้")
                    break
                
                # ปรับแต่งภาพเล็กน้อยก่อนแสดงผล
                frame = cv2.flip(frame, 1) # กลับด้านซ้ายขวา (Mirror)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # ประมวลผล
                processed_frame = analyze_frame(frame_rgb)
                
                # แสดงผลบนหน้าเว็บ
                FRAME_WINDOW.image(processed_frame)
            cam.release()
        else:
            st.write('กดที่ "เริ่มการทำงานของกล้อง" เพื่อดูภาพสด')

    else:
        uploaded_file = st.file_uploader("เลือกไฟล์ภาพ...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = np.array(Image.open(uploaded_file))
            with st.spinner('กำลังวิเคราะห์...'):
                processed_img = analyze_frame(image.copy())
                st.image(processed_img, use_column_width=True)

with col2:
    st.write("### ℹ️ คำอธิบาย")
    st.write("""
    - **Camera Source:** เลือก index ของกล้องที่ต้องการใช้ (0 คือกล้องหลัก)
    - **Detector:** - `opencv`: เร็วที่สุดแต่หลุดบ่อย
        - `mediapipe`: เร็วและแม่นยำ (แนะนำ)
        - `retinaface`: แม่นยำที่สุดแต่จะช้า (Lag)
    """)