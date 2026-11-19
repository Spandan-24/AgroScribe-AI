# 🌿 AgroScribe AI

**AI-Powered Plant Disease Detection for West Bengal Farmers**

AgroScribe AI is a Flask-based web application that uses a trained CNN model to detect plant diseases from leaf images. It provides instant diagnosis, treatment recommendations, pesticide suggestions, live weather updates with 7-day forecasts, district-wise pesticide shop listings across West Bengal, and built-in customer support.

---

## ✨ Features

- **🔬 AI Disease Detection** — Upload a leaf photo and get an instant diagnosis using a trained TensorFlow/Keras CNN model (15 disease classes across Pepper, Potato, and Tomato).
- **📊 Confidence Score & Severity** — Visual confidence bar, severity rating (Low/Medium/High), and urgency-based color coding.
- **💊 Treatment Recommendations** — Detailed immediate action, treatment, prevention, organic alternatives, and estimated yield loss for each disease.
- **🧪 Pesticide Suggestions** — Recommended pesticide products with brand, company, type, and indicative price, mapped to the predicted disease.
- **🌤️ Live Weather** — Real-time weather based on the user's geolocation (via OpenWeatherMap API), with an agricultural advisory tip.
- **📅 7-Day Forecast** — Daily high/low temperatures and rain probability for the upcoming week.
- **🏪 Pesticide Shop Directory** — District-wise list of pesticide/agro shops across 20 districts of West Bengal.
- **📖 Disease Library** — Searchable guide covering pathogen, symptoms, causes, treatment, and organic options for major crop diseases.
- **🗓️ Crop Sowing Calendar** — Seasonal sowing windows and disease-risk indicators for major West Bengal crops.
- **🆘 Help & Support Widget** — Floating support button with helpline, WhatsApp chat, email, agri-expert contact, and FAQs.
- **🎨 Modern UI** — Glassmorphism design, animated gradients, responsive layout for mobile and desktop.

---

## 🗂️ Project Structure

```
agroscribe/
├── flask_app.py                  # Flask backend & prediction API
├── models/
│   └── agro_cnn.h5         # Trained CNN model (not included — see below)
├── static/
│   └── outputs/            # Auto-created folder for uploaded leaf images
├── templates/
│   └── index.html          # Main frontend (UI, weather, shops, disease guide)
└── README.md
```

---

## 🧠 Model Details

The CNN model (`agro_cnn.h5`) is trained to classify the following **15 classes**:

| Crop    | Classes |
|---------|---------|
| Pepper  | Bacterial Spot, Healthy |
| Potato  | Early Blight, Late Blight, Healthy |
| Tomato  | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-Spotted), Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

- Input size: **224 × 224** pixels
- Preprocessing: pixel normalization (`/ 255.0`)
- Output: softmax probabilities → class with highest confidence is returned

> ⚠️ The trained model file (`agro_cnn.h5`) is **not included** in this repository due to size. Place your trained model inside the `models/` folder before running the app, or train your own using a dataset such as the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease).

---

## ⚙️ Setup & Installation

### 1. Clone / Download the project
```bash
cd d:\agroscribe
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install flask tensorflow numpy pillow
```

> 💡 If `tensorflow` installation fails on low-spec machines, use:
> ```bash
> pip install tensorflow-cpu
> ```

### 5. Add your trained model
Place your trained model file at:
```
models/agro_cnn.h5
```

### 6. Run the app
```bash
python app.py
```

### 7. Open in browser
Visit:
```
http://127.0.0.1:5000
```

---

## 🔑 API Keys

The backend uses the **OpenWeatherMap API** for live weather and forecasts.

Set the key as a server environment variable instead of placing it in `index.html`:

```powershell
$env:OPENWEATHERMAP_API_KEY="your-new-key"
python flask_app.py
```

The browser calls same-origin `/api/weather` and `/api/forecast` endpoints. Never
commit the real key to source control, and rotate a key that was previously exposed.

---

## 🔌 API Endpoints

### `GET /`
Renders the main frontend (`index.html`).

### `POST /predict`
Accepts a leaf image and returns a disease diagnosis.

**Request:** `multipart/form-data` with field `file` (image)

**Response (JSON):**
```json
{
  "prediction": "Tomato_Late_blight",
  "confidence": 94.32,
  "severity": "High",
  "urgency": "Critical",
  "image": "/static/uploads/leaf.jpg"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Predicted disease class label |
| `confidence` | number | Confidence score (0–100) |
| `severity` | string | `Low` / `Medium` / `High` / `None` |
| `urgency` | string | `Low` / `Medium` / `High` / `Critical` |
| `image` | string | Path to the saved uploaded image |

---

## 🏪 Pesticide Shop Directory

Covers **20 districts** of West Bengal including:

Kolkata, Howrah, North & South 24 Parganas, Hooghly, Nadia, Murshidabad, Bardhaman (East & West), Birbhum, Bankura, Purulia, Midnapore (East & West), Jalpaiguri, Darjeeling, Cooch Behar, Malda, and Dinajpur (North & South).

Each shop entry includes name, address, and product tags (pesticides, seeds, fertilizers, organic inputs).

---

## 🌾 Pesticide Recommendations

Based on the predicted disease, the app suggests relevant products from companies including:

Syngenta, Bayer, UPL, Indofil Industries, Rallis India, FMC Corp, DuPont, Sumitomo Chemical, Dhanuka, NACL Industries, and AgriLife Biotech — covering fungicides, bactericides, miticides, and organic/bio-pesticide alternatives.

---

## 🆘 Support Channels

Built into the floating help widget:

- 📞 Helpline call
- 💬 WhatsApp chat
- 📧 Email support
- 👨‍🌾 Agri-expert consultation (via KVK)
- ❓ FAQ section

> Update contact details (`tel:`, `wa.me`, `mailto:`) in `index.html` with your actual support numbers/emails before deployment.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask (Python) |
| Model | TensorFlow / Keras CNN |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| UI Framework | Bootstrap 5.3.3, Font Awesome 6 |
| Weather API | OpenWeatherMap |
| Fonts | Google Fonts (Inter, Playfair Display) |

---

## 📌 Notes & Disclaimers

- Pesticide prices and product availability are **indicative** and may vary by region and season.
- Always confirm dosage and safety guidelines on product labels.
- For severe or unclear cases, consult your local **Krishi Vigyan Kendra (KVK)** or agricultural extension officer.
- Uploaded images are stored temporarily in `static/uploads/` for prediction purposes.

---

## 📄 License

This project is provided for educational and agricultural extension purposes. Customize and deploy as needed for your region.

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) for disease classification training data
- [OpenWeatherMap](https://openweathermap.org) for weather data
- [Font Awesome](https://fontawesome.com) & [Bootstrap](https://getbootstrap.com) for UI components