# Bambu Lab A1 Smart Cloud Optimizer

A production-ready SaaS web application and automation engine tailored specifically for the **Bambu Lab A1 3D Printer**.

Automatically packages STL / OBJ / 3MF files into official **Bambu Studio / OrcaSlicer native 3MF project archives** with calibrated slicing settings for devotional idols, miniatures, and high-precision prints.

---

## ⚡ Key Features

- **1-Click 3MF Generation:** Injects official Bambu Studio package namespaces (`xmlns:BambuStudio="http://schemas.bambulab.com/package/2021"`), eliminating the *"Not from Bambu Lab"* warning.
- **Arachne Engine Tuning:** Ultra-crisp thin walls, idol face details, jewelry, and sharp corners.
- **Auto Tree Slim Supports:** Optimized branch/tip diameters with calibrated Z-distance (0.16mm for ≤50mm, 0.20mm for larger) for clean separation.
- **Warp Prevention:** Preloaded 65°C Textured PEI plate temperature and 220°C nozzle profiles.
- **Built-in Private Access Control:** Guarded by credentials (`blogo` / `Blogo@12`) configurable via environment variables.
- **Automatic Resource Cleanup:** Background tasks delete temporary uploaded models and packaged 3MFs to preserve server disk space.

---

## 🚀 Quick Start (Local)

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the server:**
   ```bash
   python web_app.py
   ```
   Open `http://localhost:8000` in your browser.

---

## ☁️ Deployment (Render / Railway / Docker)

### Render.com (Free Tier)
1. Push this repository to GitHub (Private).
2. Create a new **Web Service** on Render connected to the repository.
3. Settings:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn web_app:app --host 0.0.0.0 --port $PORT`
4. Set optional Environment Variables in the dashboard:
   - `APP_AUTH_USER`: `blogo`
   - `APP_AUTH_PASS`: `Blogo@12`

### Docker
```bash
docker build -t bambu-optimizer .
docker run -p 8000:8000 bambu-optimizer
```

---

## 📂 Project Structure

```
├── web_app.py                # FastAPI Cloud SaaS Backend & Frontend UI
├── bambu_base_template.json  # Complete 502-key Bambu Studio configuration template
├── requirements.txt          # Python dependencies
├── Dockerfile                # Production container definition
├── Procfile                  # Process definition for web hosting
├── .env.example              # Environment variables template
├── .gitignore                # Git exclusions (ignores large STL/3MF test files)
└── README.md                 # Project documentation
```
