# Adnan Thange - AI, ML & Mlops Portfolio

_A curated collection of my Machine Learning, Deep Learning, Generative AI, and MLOps projects._

---

## 🔹 Projects & Research

### **Selective Data Clarity Model (SDC-AF) – Research Paper**
- Developing a novel ML preprocessing framework based on the Selective Data Clarity Principle  
- Combines automated outlier removal with task-aware feature selection  
- Outperforms PCA on benchmark datasets (e.g., Iris: 100% vs. 95% accuracy), enhancing classification and regression performance  

### **PolarGraphFormer (PGF) – Research Paper (In Progress)**
- Hybrid transformer–polar CNN–graph neural network  
- Designed with cross-attention fusion for global and rotation-invariant features  
- Achieved 99.6% test accuracy on MNIST, exceeding CNN and ViT baselines  

### **AI Chatbot** – Python | Google Generative AI SDK
- Built an interactive chatbot using Google Gemini LLM  
- Implemented real-time streaming responses and secure API integration  

### **Heart Disease Prediction (Accuracy: 85%)**
- ML pipeline with PCA, Logistic Regression, SVM, Random Forest  
- Applied hyperparameter tuning with RandomizedSearchCV  
- Best model selected based on cross-validated scores  

### **Movie Recommendation System**
- Personalized recommendation engine using user-based collaborative filtering  
- Used cosine similarity on normalized user-item matrix to suggest top-rated unseen movies  

### **Market Basket Analysis using Apriori**
- Analyzed 14,963 grocery transactions to find frequent itemsets and association rules  
- Applied mlxtend Apriori with min support 0.005 and lift > 0.3  

### **Customer Churn Prediction**
- Built real-time object tracking system using Python, OpenCV, and NLP-based processing  
- Optimized accuracy and speed through algorithm improvements  

### **Brain Tumor Detection**
- MRI image preprocessing and CNN for automated tumor classification  
- Improved detection accuracy  

### **CVTrackr – Object Tracking System**
- Real-time object tracking with Python, OpenCV, and NLP integration  
- Optimized for accuracy and speed  

### **Sentiment Analysis**
- ML-based text classification to analyze and categorize user-generated content  

### **Hand Gesture Recognition System**
- Real-time hand gesture recognition using MediaPipe and OpenCV  
- Integrated gesture-based controls to launch apps and adjust volume for touchless PC interaction  

### **IMDB Sentiment Analysis with MLflow & DagsHub**
- Built an LSTM-based NLP pipeline for IMDB movie reviews  
- Tracked experiments, metrics, and models using **MLflow**  
- Integrated versioning and collaborative workflow with **DagsHub**  
- Optimized model performance through hyperparameter tuning and preprocessing  

# CI Workflow for IMDB LSTM Training

This repository contains the Continuous Integration (CI) workflow for the IMDB LSTM training project. The workflow automatically runs on `push` or `pull request` to the `main` branch and executes the training script.

---

## 🛠 Workflow Steps

1. **Checkout Code**  
   The workflow first checks out the repository code.

2. **Set Up Python**  
   Python 3.10 is installed.

3. **Install Dependencies**  
   Required Python packages from `requirements.txt` are installed.

4. **Run Training Script**  
   Executes `train_imdb.py` located in:  
   `Ml and Dl Projects with Mlops/dl_projects_lstm_with_mlflow/train_imdb.py`

5. **Upload Artifacts (Optional)**  
   Any generated files (like logs, predictions, or screenshots) can be uploaded as workflow artifacts.

---

## 📸 CI Run Screenshot

The following screenshot shows a successful CI run for the training workflow:  

![CI Run Screenshot](Ml and Dl Projects with Mlops/dl_projects_lstm_with_mlflow/CI AND CD/ci_run_screenshot.png)

---

## ⚡ Notes

- This workflow is **CI only**. No deployment (CD) is configured yet.
- Screenshots are stored in the `Ml and Dl Projects with Mlops/dl_projects_lstm_with_mlflow/CI AND CD` folder.
- Ensure your Python scripts and dependencies are correctly configured to avoid workflow failures.


---

## 📫 Contact

- **Email**: thangeadnan31@gmail.com  
- **GitHub**: [github.com/AdnanThange](https://github.com/AdnanThange)  
- **LinkedIn**: [linkedin.com/in/adnan-thange](https://linkedin.com/in/adnan-thange)

