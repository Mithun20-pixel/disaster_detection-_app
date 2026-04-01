 # 🛰️ Disaster Damage Detection using Satellite Images

AI-powered satellite image analysis for detecting flood and earthquake damage. Upload a satellite image → AI detects damage → Visualize affected areas on a map.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🏷️ **Binary Classification** — Damaged vs. Not-Damaged (EfficientNet transfer learning)
- 🗺️ **Pixel-level Segmentation** — Precise damage masks (U-Net with ResNet-34 encoder)
- 🔥 **Heatmap Visualization** — Damage probability overlays
- 📦 **Bounding Boxes** — Detected damage regions highlighted
- 📊 **Severity Gauge** — Visual damage percentage scoring
- 🌍 **Interactive Map** — Results plotted on Folium/Leaflet map
- 🔄 **Before/After** — Compare pre- and post-disaster images
- 📥 **Downloadable Reports** — Export heatmaps and analysis reports

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the web app
streamlit run app.py

# 3. (Optional) Train on your own data
python train.py --mode classification --epochs 10 --data_dir data/train

# 4. (Optional) CLI prediction
python predict.py --image path/to/satellite.png --mode both
```

## 📁 Project Structure

```
├── app.py                # Streamlit web app
├── train.py              # Training script
├── predict.py            # CLI prediction
├── config.py             # Configuration
├── models/
│   ├── cnn_model.py      # EfficientNet classifier
│   └── unet_model.py     # U-Net segmentation
├── utils/
│   ├── preprocessing.py  # Image transforms
│   ├── prediction.py     # Inference
│   └── visualization.py  # Heatmaps & overlays
├── data/sample/          # Sample images
└── .streamlit/config.toml
```

## 🧠 Models

| Model | Use Case | Training Time | Accuracy |
|-------|----------|---------------|----------|
| EfficientNet-B0 | Binary classification | ~5 min (GPU) | ~92% |
| U-Net (ResNet-34) | Pixel segmentation | ~20 min (GPU) | IoU ~0.75 |

## 📊 Datasets

- [xBD Dataset](https://xview2.org/dataset) — 22K images, 19 disaster events
- [Kaggle Hurricane Damage](https://www.kaggle.com/datasets/kmader/satellite-images-of-hurricane-damage)
- [Copernicus EMS](https://emergency.copernicus.eu)
- [Maxar Open Data](https://www.maxar.com/open-data)

## ☁️ Deployment

- **Streamlit Cloud:** Push to GitHub → [share.streamlit.io](https://share.streamlit.io)
- **HuggingFace Spaces:** Create Space → Upload files
- **Render:** Add `render.yaml` → Connect GitHub

## 📄 License

MIT License — Free for hackathons, research, and humanitarian use.
