# FTIR Spectra Analysis and Denoising for Microplastics

**Developed by:** Jarawee Saengkaewsuk  
*Senior Project – Digital Engineering*  
School of Information, Computer and Communication Technology, Faculty of Digital Engineering  
Sirindhorn International Institute of Technology, Thammasat University

---

## 📌 Project Overview

This web application provides an interactive platform for analyzing and denoising Fourier Transform Infrared (FTIR) spectral data, specifically for microplastic analysis.

**Tech Stack:**
- **Frontend:** React.js – for an intuitive and responsive user interface
- **Backend:** FastAPI + Python – for handling data processing and deep learning model execution

### 🎯 Objective

The goal is to improve the quality of noisy FTIR spectra using deep learning models, enhancing their readability and preserving key spectral features for accurate microplastic classification.

---

## 🧠 Denoising Models

### ✅ Proposed Deep Learning Models
- `CNNAE-MobileNet`
- `CNNAE-InceptionV3`
- `CNNAE-Xception`
- `CNNAE-InceptionResNet`
- `CNNAE-ResNet50`

### 🔬 Baseline Models for Comparison
- `Autoencoder`
- `U-Net`
- `CNN Autoencoder`

---

## ⚙️ Installation Guide

Follow these steps to run the application locally:

### 1. Clone the Repository

```bash
git clone https://github.com/Jarawee222/SL1.git
```

### 2. Set Up the Backend

```bash
cd backend
pip install -r requirements.txt
```

### 3. Set Up the Frontend

```bash
cd frontend
npm install
```

### 4. Start the Backend Server

```bash
cd backend
uvicorn main:app --reload
```

### 5. Start the Frontend Application

```bash
cd frontend
npm start
```

### 🌐 Access the Application

Once both the backend and frontend are running, open your browser and go to:  
[http://localhost:3000](http://localhost:3000)

---

## 🧪 Project Usage 

1. **Upload FTIR CSV File:**  
    Use the interface to upload a noisy spectrum file in `.csv` format.
2. **Select the preprocessing options:**  
    Choose one of the available preprocessing options (e.g., baseline correction) and click the "Apply" button to preprocess the spectrum and fine-tune it.
3. **Denoise the spectrum by using models:**  
    Choose one of the available deep learning models (e.g., CNNAE-Xception) and click the "Denoise" button to process the preprocessed file.
4. **Compare results:**  
    View and compare the originally cleaned vs. denoised spectrum through overlaid graph display.

---

## 📁 Project Structure

```
SL1/
├── backend/
│   ├── main.py
│   ├── model_files/
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   └── package.json
├── README.md
└── .gitignore
```

---

## 📷 Screenshots



---

## 📊 Performance Metrics

| Model Name                | Pearson Correlation | Accuracy Score |
|---------------------------|--------------------|---------------|
| CNNAE-Xception            | 0.9471625          | 0.959567      |
| CNNAE-InceptionResNet     | 0.9043131          | 0.929233      |
| CNNAE-Mobilenet           | 0.9038519          | 0.925633      |
| CNNAE-ResNet50            | 0.9088153          | 0.937433      |
| CNNAE-InceptionV3         | 0.8280775          | 0.900900      |


---

## 🛠 Tech Stack Summary

| Component   | Technology          |
|-------------|---------------------|
| Frontend    | React.js            |
| Backend     | FastAPI, Python     |
| DL Framework| TensorFlow / Keras  |
| Styling     | CSS, Bootstrap      |

---

## 📬 Contact

For questions or feedback, feel free to contact:  
[jarawee.saeng@email.com](mailto:jarawee.saeng@email.com)  
[linkedin/in/jarawee](https://linkedin.com/in/jarawee)