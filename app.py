# .\momo\Scripts\activate
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import pandas as pd
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Configure TensorFlow to use CPU only to avoid conflicts
tf.config.set_visible_devices([], 'GPU')

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")
        cnn_model = load_model("model/nephron_classifier_model.h5")
        return yolo_model, cnn_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Email function
def send_email(recipient_email, image_path, report_path):
    load_dotenv()
    sender_email = os.getenv("SENDER_EMAIL")
    sender_password = os.getenv("SENDER_PASSWORD")

    if not sender_email or not sender_password:
        st.error("Email credentials not set. Please configure environment variables.")
        return False

    msg = EmailMessage()
    msg["Subject"] = "Your Nephron Detection Report"
    msg["From"] = sender_email
    msg["To"] = recipient_email
    msg.set_content("Please find your nephron detection image and report attached.")

    try:
        with open(image_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="image", subtype="png", filename="nephron_combined_output.png")

        with open(report_path, "rb") as f:
            msg.add_attachment(f.read(), maintype="text", subtype="csv", filename="nephron_report.csv")

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# Helper functions
def add_label(image, text):
    labeled = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 2
    color = (0, 0, 0)

    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 40), (255, 255, 255), -1)
    cv2.putText(labeled, text, (10, 30), font, font_scale, color, thickness, lineType=cv2.LINE_AA)
    return labeled

def is_nephron_image(image, cnn_model):
    try:
        # Preprocess image
        img_resized = cv2.resize(image, (638, 478))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_array = img_to_array(img_rgb) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Get prediction
        prediction = cnn_model.predict(img_array, verbose=0)
        confidence = float(prediction[0][0])
        return confidence >= 0.5, confidence
    except Exception as e:
        st.error(f"Error during CNN validation: {str(e)}")
        return False, 0.0

# Main Streamlit App
def main():
    st.set_page_config("Nephron Detection", layout="centered")
    st.title("🫁 Nephron Counting with CNN and YOLOv8")
    st.markdown("Upload an image and click **Quantify** to detect nephrons.")

    # Load models
    yolo_model, cnn_model = load_models()
    if yolo_model is None or cnn_model is None:
        st.error("Failed to load models. Please check if model files exist in the correct location.")
        return

    uploaded_file = st.file_uploader("\U0001F4E4 Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        try:
            bytes_data = uploaded_file.read()
            img_array = np.frombuffer(bytes_data, np.uint8)
            original_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

            st.image(original_rgb, caption="Uploaded Image", use_container_width=True)

            if st.button("\U0001F50D Quantify"):
                with st.spinner("Verifying nephron image..."):
                    is_nephron, confidence = is_nephron_image(original_bgr, cnn_model)
                    if not is_nephron:
                        st.error("⚠️ Uploaded image does NOT appear to be a nephron. Please upload a valid nephron image.")
                        return
                    st.success(f"✅ Image validated as nephron (Confidence: {confidence:.2%})")

                with st.spinner("Counting nephrons..."):
                    results = yolo_model(original_bgr)
                    filtered_boxes = [box for box, conf in zip(results[0].boxes.xyxy, results[0].boxes.conf) if conf >= 0.55]

                    annotated = original_bgr.copy()
                    for box in filtered_boxes:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    count = len(filtered_boxes)
                    st.success(f"\u2705 Nephrons Detected: {count}")

                    original_labeled = add_label(original_rgb, "Original Image")
                    annotated_labeled = add_label(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), "Annotated Image")
                    padding = np.ones((original_labeled.shape[0], 10, 3), dtype=np.uint8) * 255
                    combined = np.hstack((original_labeled, padding, annotated_labeled))

                    st.image(combined, caption="Original vs Annotated", use_container_width=True)

                    # Save outputs
                    img_out_path = "nephron_combined_output.png"
                    cv2.imwrite(img_out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

                    report_df = pd.DataFrame({
                        "Metric": ["Detected Nephrons", "CNN Validation Confidence"],
                        "Value": [count, f"{confidence:.2%}"]
                    })
                    report_out_path = "nephron_report.csv"
                    report_df.to_csv(report_out_path, index=False)

                    st.session_state["img_out"] = img_out_path
                    st.session_state["report_out"] = report_out_path

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            return

        if "img_out" in st.session_state and "report_out" in st.session_state:
            st.markdown("### \U0001F4E5 Download Your Results")
            col1, col2 = st.columns(2)
            with col1:
                with open(st.session_state["img_out"], "rb") as imgf:
                    st.download_button("Download Image", imgf, file_name="nephron_output.png", mime="image/png")
            with col2:
                with open(st.session_state["report_out"], "rb") as repf:
                    st.download_button("Download Report", repf, file_name="nephron_report.csv", mime="text/csv")

            st.markdown("### \U0001F4E7 Send via Email")
            email = st.text_input("Enter your email address")
            if st.button("\U0001F4E4 Send Email"):
                if email:
                    success = send_email(email, st.session_state["img_out"], st.session_state["report_out"])
                    if success:
                        st.success("\U0001F4EC Email sent successfully!")
                else:
                    st.warning("Please enter an email address.")

    st.markdown("---")
    st.markdown("<center>\u2692\ufe0f Developed by MSD</center>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
