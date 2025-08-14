# 🧬 Nephron_counting

**Nephron_counting** is an AI-powered biomedical image analysis tool that automates **nephron detection and counting** from kidney histology images.  
It combines a **CNN** for image validation and **YOLOv8** for object detection — all wrapped in a **Streamlit** web app for easy use by **researchers, clinicians, and educators**.

---

## ✨ Features

- 📊 **Streamlit Web App** – Upload kidney histology images for instant analysis.
- 🔍 **CNN Validation** – Ensures uploaded images are nephron tissue samples.
- 🎯 **YOLOv8 Detection** – Detects, annotates, and counts nephrons automatically.
- 📥 **Download & Email Results** – Get annotated images and CSV reports instantly.
- 🐍 **Python-Based** – Uses major deep learning & image processing libraries.

---

## 🧪 Workflow

1. **Upload an Image** – JPG, PNG, or JPEG format.
2. **Validation** – CNN verifies it’s a nephron sample.
3. **Detection & Counting** – YOLOv8 processes, annotates, and counts.
4. **Results** –  
   - 📷 `nephron_combined_output.png` – Annotated output image  
   - 📄 `nephron_report.csv` – Nephron count & metrics  
   - 📧 Email your results directly

---

## ⚙️ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/rpdeeksh/Nephron_counting.git
cd Nephron_counting

# 2️⃣ Install dependencies
pip install -r requirements.txt
