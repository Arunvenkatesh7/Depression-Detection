# 🧠 Multi-Modal Depression Detection System

[![Google Drive](https://img.shields.io/badge/Google%20Drive-Download%20Project-blue)](https://drive.google.com/file/d/1qNRY9nSeS2c3TijeUPwOzoNKi_-vOUn4/view?usp=drive_link)
[![Research Paper](https://img.shields.io/badge/Research-Academic%20Paper-green)](https://your-paper-link)

A comprehensive system for detecting depression and mental health disorders using multiple data modalities: Quantitative EEG (QEEG), EEG-EDF signals, and Audio recordings. This project implements advanced machine learning and deep learning techniques to provide accurate classification and diagnosis support.

## 📊 Project Overview

This depression detection platform consists of three primary modules, each utilizing different data modalities:

1. **🔢 Quantitative EEG Channel Data** - Classification of 12 mental disorders from EEG features
2. **📈 EEG-EDF Based Depression Detection** - Binary classification of Major Depressive Disorder (MDD) vs. Healthy
3. **🎤 Audio Based Depression Detection** - Identification of depression markers from speech patterns

## 📋 Module Details

### 🔢 Quantitative EEG Channel Data Module

This module processes EEG data containing 1140 features across different brain waves (alpha, beta, gamma, delta, and theta) from 19 EEG channels, along with demographic and categorical features.

**Technical Implementation:**
- **Data Preprocessing:**
  - Missing value imputation using column mean/median
  - Ordinal encoding of categorical features
  - Normalization of continuous variables using StandardScaler
- **Feature Selection:**
  - ElasticNet regularization for identifying significant features
  - Retention of top 50 most important EEG features
- **Dataset Balancing:**
  - SMOTE (Synthetic Minority Over-sampling Technique) for class balance
- **Model Selection & Training:**
  - XGBoost classifier (outperformed LGBM, Bi-LSTM, RNN, Gradient Boosting, AdaBoost, KNN)

**Performance Metrics:**
- High classification accuracy across 12 mental disorder categories

### 📈 EEG-EDF Based Depression Detection Module

This module analyzes EEG data in European Data Format (EDF) from 30 healthy and 34 MDD participants across three recording states: Eyes Opened (EO), Eyes Closed (EC), and Task.

**Technical Implementation:**
- **Data Preprocessing:**
  - Missing value handling
  - Bandpass filtering (0.5-45 Hz)
  - Signal conversion from Volts to Microvolts
  - EEG signal visualization and image creation
- **Model Selection & Training:**
  - ResNet18 deep learning model (outperformed DenseNet, EfficientNet, Swin Transformer)
  - Binary classification for MDD vs. Healthy
  - 6-epoch training with CrossEntropyLoss and Adam optimizer

**Performance Metrics:**
- Superior train and test accuracy with low loss metrics

### 🎤 Audio Based Depression Detection Module

This module processes audio recordings from 189 interview sessions with an animated virtual interviewer (Ellie), with durations ranging from 7 to 33 minutes (16 minutes average).

**Technical Implementation:**
- **Audio Preprocessing:**
  - 16 kHz sampling rate standardization
  - Mono channel conversion and amplitude normalization
  - Silence removal and segmentation
  - Audio augmentation (noise injection, speed/pitch modification, time stretching)
- **Feature Extraction:**
  - Chroma STFT (pitch class profile)
  - RMS Energy (signal loudness)
  - Spectral Centroid (sound brightness)
  - Spectral Bandwidth (sound spectrum distribution)
  - Zero Crossing Rate (frequency content)
  - 20 Mel-Frequency Cepstral Coefficients (MFCCs)
- **Data Preparation:**
  - SMOTE for class balancing
  - StandardScaler for feature normalization
- **Model Development:**
  - Bi-LSTM architecture (outperformed ConvLSTM2D, LSTM, Dense sequential, XGBoost, KNN, SVC, Bagging Classifier, Decision Tree, Logistic Regression)
  - Binary Cross-Entropy loss function with Adam optimizer

## 🛠️ Technologies Used

- **Programming Languages:** Python
- **Machine Learning Frameworks:** 
  - Scikit-learn for preprocessing and traditional ML models
  - TensorFlow/Keras for deep learning models
  - XGBoost for gradient boosting
- **Signal Processing:**
  - MNE-Python for EEG processing
  - Librosa for audio feature extraction
- **Data Visualization:**
  - Matplotlib and Seaborn for data visualization
  - Plotly for interactive visualizations
- **Deep Learning Models:**
  - ResNet18 for image-based classification
  - Bi-LSTM for sequence data processing

## 📦 Installation

1. **Access the project**
   
   **Option 1: GitHub Repository**
   ```bash
   git clone https://github.com/your-username/depression-detection.git
   cd depression-detection
   ```
   
   **Option 2: Google Drive**
   The complete project is also available on Google Drive:
   [Download Depression Detection System](https://drive.google.com/file/d/1L3NFVBg6SqqoRrZfCVlknxoJZRbOJOCY/view?usp=drive_link)

2. **Environment Setup**
   ```bash
   # Create and activate a virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Data Requirements**
   - QEEG .csv files with 1140 features
   - EEG-EDF files from EC, EO, and Task states
   - Audio recordings in compatible format (.wav recommended)

4. **Run the modules**
   ```bash
   
   # For Audio analysis
   python audio_analysis.py
   ```

## 🧪 Challenges & Solutions

### QEEG Module Challenges:
- **Missing Data:** Handled through careful imputation techniques
- **Feature Selection:** Optimized ElasticNet with cross-validation
- **Class Imbalance:** Addressed with carefully tuned SMOTE
- **Model Tuning:** XGBoost hyperparameters required meticulous adjustment

### EEG-EDF Module Challenges:
- **Signal Noise:** Resolved with bandpass filtering
- **Small Dataset:** Augmented data and applied balancing techniques
- **Model Selection:** Required extensive computational resources for testing
- **Multi-state Data:** Handled each state (EO, EC, Task) with appropriate validation

### Audio Module Challenges:
- **Recording Variability:** Standardized through preprocessing
- **Feature Extraction:** Carefully selected acoustic features
- **Complex Model Training:** Applied regularization and cross-validation
- **Deployment:** Implemented robust serialization and inference pipeline

## 📚 Research Foundation

This project integrates evidence-based approaches to depression detection:
- Analysis of brain wave patterns associated with mental disorders
- EEG markers specific to Major Depressive Disorder
- Speech and acoustic features linked to depressive states

## 🧪 Future Enhancements

- Integration of additional modalities (facial expressions, text analysis)
- Real-time analysis capabilities
- Mobile application deployment
- Enhanced explainability through feature importance visualization
- Cloud-based deployment for clinical usage

## 🤝 Contribution

Researchers, data scientists, and mental health professionals are welcome to contribute to this project. Please feel free to fork this repository and submit pull requests.

## 📄 License

MIT License – see LICENSE file for details.

## 🙏 Acknowledgements

- Special thanks to all researchers who contributed to the datasets
- Gratitude to the academic advisors guiding this research
- Recognition of the open-source community providing tools and libraries
