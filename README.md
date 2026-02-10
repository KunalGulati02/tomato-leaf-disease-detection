
# Tomato Leaf Disease Detection (CNN + KNN Hybrid Model)

This project detects diseases in tomato leaves using a hybrid **Convolutional Neural Network (CNN)** for feature extraction and **K-Nearest Neighbors (KNN)** for classification.  
The work was presented at the **IC-AMSI-2024 International Conference**.

---

## Overview
Tomato crops suffer from multiple bacterial, fungal, and viral diseases. Early detection helps prevent losses.  
This project uses:
- CNN for deep feature extraction  
- KNN for lightweight classification  
- PlantVillage dataset  
- Jupyter Notebooks & Python scripts  
- Research-backed methodology  

---

## Model Architecture

### **1️. CNN (Convolutional Neural Network)**
Extracts high-level image features such as:
- Texture  
- Lesion patterns  
- Color distortions  

### **2️. KNN (K-Nearest Neighbors)**
Uses CNN features for classification:
- Fast inference  
- Low complexity  
- Interpretable results  

---

## 📂 Project Structure
```
tomato-leaf-disease-detection/
├── notebooks/
│   └── model.ipynb
├── src/
│   └── model.py
├── model.json
├── paper/
│   └── IC-AMSI-2024.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset
This project uses the **PlantVillage Tomato Leaf Dataset**, which includes categories like:

- Early Blight  
- Late Blight  
- Leaf Mold  
- Septoria Leaf Spot  
- Spider Mite Damage  
- Bacterial Spot  
- Healthy Leaves  

Dataset download link:  
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset  

(The dataset is not included in this repo due to size.)

---

## Results
The hybrid model achieved:
- **High accuracy** across classes  
- **Lower inference time** than CNN-only models  
- **Stable performance** on noisy samples  

More details are covered in the research paper located under `paper/Final Paper.pdf`.

---

## Technologies Used

### **Languages**
- Python

### **Libraries**
- TensorFlow / Keras  
- scikit-learn  
- NumPy  
- pandas  
- Matplotlib  
- OpenCV  

---

##  How to Run

### **1️. Install dependencies**
```
pip install -r requirements.txt
```

### **2️. Open the training notebook**
```
jupyter notebook notebooks/model.ipynb
```

### **3️. Run the Python script (optional)**
```
python src/model.py
```

---

## Research Publication
This project was presented at:

### IC-AMSI-2024 — International Conference on Advances in Multidisciplinary Sciences and Innovations

Read the paper in `paper/Final Paper.pdf`.


