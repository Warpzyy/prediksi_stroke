# Sistem Prediksi Risiko Stroke (Regresi Logistik)

Prototipe aplikasi web untuk memprediksi risiko stroke berdasarkan faktor kesehatan dan gaya hidup.
Dataset: Stroke Prediction Dataset (Kaggle, fedesoriano).

## Struktur
- `analisis_stroke.ipynb` : EDA, preprocessing, SMOTE, training, evaluasi, penyimpanan model
- `app.py`, `templates/index.html` : aplikasi Flask
- `model/` : hasil training (dibuat oleh notebook): `preprocessor.joblib`, `model_logreg.joblib`, `config.json`
- `data/` : dataset CSV, `gambar/` : grafik hasil notebook

## Cara menjalankan (Windows / VS Code)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
```
1. Buka `analisis_stroke.ipynb` di VS Code, pilih kernel `venv`, lalu **Run All**.
2. Jalankan aplikasi: `python app.py` lalu buka http://127.0.0.1:5000

## Deploy (Render)
- Build command : `pip install -r requirements.txt`
- Start command : `gunicorn app:app`
- Pin versi scikit-learn di `requirements.txt` sama dengan versi lokal (`pip show scikit-learn`),
  karena file .joblib sensitif terhadap versi.
