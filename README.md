# 🌿 AgroScribe AI

**AI-Powered Plant Disease Detection for Farmers**

AgroScribe AI is a Flask-based web application that uses a trained Convolutional Neural Network (CNN) to detect plant diseases from leaf images. It provides instant diagnosis, severity and urgency indicators, organic and chemical treatment recommendations, pesticide suggestions, live weather updates with 7-day forecasts, district-wise pesticide shop listings across West Bengal, and built-in farmer support.

---

## ✨ Features

- **🔬 AI Disease Detection** — Upload a leaf photo and get an instant diagnosis using a trained TensorFlow/Keras CNN model (15 disease classes across Pepper, Potato, and Tomato with **92% classification accuracy**).
- **📊 Confidence Score & Severity** — Visual confidence bar, severity rating (`Low` / `Medium` / `High`), and urgency-based color coding (`Low` / `Medium` / `High` / `Critical`).
- **💊 Treatment Recommendations** — Detailed immediate action, treatment, prevention, organic alternatives, and estimated yield loss for each disease.
- **🧪 Pesticide Suggestions** — Recommended pesticide products with brand, company, type, and indicative price mapped to the predicted disease.
- **🌤️ Live Weather** — Real-time weather based on the user's geolocation (via OpenWeatherMap API), with an agricultural advisory tip.
- **📅 7-Day Forecast** — Daily high/low temperatures and rain probability for the upcoming week.
- **🏪 Pesticide Shop Directory** — District-wise list of pesticide/agro shops across 20 districts of West Bengal.
- **📖 Disease Library** — Searchable guide covering pathogen, symptoms, causes, treatment, and organic options for major crop diseases.
- **🗓️ Crop Sowing Calendar** — Seasonal sowing windows and disease-risk indicators for major West Bengal crops.
- **🆘 Help & Support Widget** — Floating support button with helpline, WhatsApp chat, email, agri-expert contact (via KVK), and FAQs.
- **🎨 Modern UI** — Glassmorphism design, animated gradients, and responsive layout for mobile and desktop.

---

## 🗂️ Project Structure

```text
AgroScribe-AI/
├── flask_app.py            # Main Flask backend & prediction API
├── models/
│   └── agro_cnn.h5         # Trained CNN model (excluded from repo — see below)
├── static/
│   └── outputs/            # Auto-created folder for uploaded leaf images
├── templates/
│   └── index.html          # Main frontend (UI, weather, shops, disease guide)
├── .gitignore              # Git exclusion rules
├── LICENSE                 # MIT License
└── README.md               # Project documentation
