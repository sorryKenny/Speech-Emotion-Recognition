# Speech Emotion Recognition Using Emotion2Vec and Deep Learning

**Undergraduate Final Year Project | B.Sc. in Data Science | BNU-HKBU United International College**

**Author:** Anyi Chen (Kenny)  
**Thesis:** November 2024

## Overview

This project investigates speech emotion recognition (SER) using deep learning and pretrained speech representations. The goal was to classify emotions from spoken audio while comparing traditional acoustic features with representations learned by pretrained speech models.

The study integrated four emotional speech datasets, evaluated multiple neural network architectures, and examined how model performance changes between controlled benchmark recordings and independently recorded speech.

**Best result:** 78.5% test accuracy using pretrained Emotion2Vec features with a deep neural network (DNN) classifier.

## Datasets

Four publicly available speech emotion datasets were combined:

- **CREMA-D:** Crowd-sourced emotional speech recordings.
- **RAVDESS:** Emotional speech recordings by professional actors.
- **SAVEE:** Emotional speech recordings from British English speakers.
- **TESS:** Emotional speech recordings from two female speakers.

The combined dataset covered eight emotion categories: anger, calm, disgust, fear, happiness, neutral, sadness, and surprise.

Data preprocessing included label encoding, feature standardization, and a stratified training/validation/test split of 70%/15%/15%.

To address class imbalance and improve training diversity, the project explored noise injection, time stretching, pitch shifting, and SMOTE-based oversampling.

## Methodology

### Feature Extraction

Three approaches were compared:

1. **Traditional Acoustic Features:** Zero-crossing rate (ZCR), MFCCs, chroma, RMS energy, and Mel spectrograms.
2. **Wav2Vec 2.0:** Pretrained speech representations for downstream emotion classification.
3. **Emotion2Vec:** Pretrained emotion-oriented speech embeddings designed to capture emotion-related information.

### Model Architectures

The experiments compared multiple neural network architectures:

- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM)
- CNN-LSTM
- WaveNet
- Deep Neural Network (DNN)

The best-performing configuration used **Emotion2Vec features with a DNN classifier**.

The DNN used fully connected layers with ReLU activations, dropout regularization, and a softmax output layer for eight-class classification. Training used categorical cross-entropy loss and the Adam optimizer.

### Technologies

Python, TensorFlow/Keras, scikit-learn, Librosa, imbalanced-learn (SMOTE), NumPy, and pretrained speech representation models.

## Experimental Results

The following results were reported in the undergraduate thesis:

| Feature Extraction / Model | Validation Accuracy | Test Accuracy |
|---|---:|---:|
| Traditional Features + CNN | 58.2% | 56.8% |
| Traditional Features + LSTM | 55.0% | 55.0% |
| Traditional Features + BiLSTM | 57.6% | 57.0% |
| Traditional Features + CNN-LSTM | 60.4% | 56.8% |
| Traditional Features + WaveNet | 58.4% | 56.3% |
| Traditional Features + DNN | 63.5% | 62.3% |
| Wav2Vec 2.0 + DNN | 72.3% | 71.7% |
| **Emotion2Vec + DNN** | **80.5%** | **78.5%** |

The Emotion2Vec + DNN configuration achieved the highest test accuracy among the evaluated approaches, with an overall reported F1-score of approximately 0.79.

### Generalization to Self-Recorded Speech

To evaluate performance outside the benchmark datasets, the final model was additionally tested on **69 self-recorded audio samples** from non-professional speakers.

- **Benchmark test accuracy:** 78.5%
- **Self-recorded speech accuracy:** 44.93%

The substantial performance drop highlights the difficulty of generalizing emotion recognition models from controlled, acted speech datasets to less controlled real-world recordings.

This finding suggests that broader speaker diversity, more natural speech data, and stronger robustness evaluations would be important next steps.

## Limitations and Future Work

- The benchmark evaluation used a stratified split of the combined datasets; it should not be interpreted as an independent cross-dataset or speaker-disjoint evaluation.
- Performance decreased substantially on self-recorded audio.
- The study relied on pretrained Emotion2Vec and Wav2Vec 2.0 representations rather than training these speech foundation models from scratch.
- Future work could investigate stronger cross-speaker and cross-dataset validation, more diverse spontaneous speech recordings, and domain adaptation techniques.

## Thesis

The full undergraduate thesis is available in this repository:

**[Speech Emotion Recognition — Undergraduate Thesis](Speech_Emotion_Recognition_Thesis.pdf)**

The thesis contains the methodology, experimental comparisons, training curves, confusion matrices, and a discussion of the model's limitations.

## Code Availability

The experiments were originally developed in Kaggle Notebooks. The original Kaggle account is currently inaccessible, so the source notebooks and trained model weights are not included in this repository.

This repository currently serves as a **research documentation and results portfolio**, rather than a fully reproducible software release.

## Author

**Anyi Chen (Kenny)**  
B.Sc. Data Science, BNU-HKBU United International College  
[GitHub Profile](https://github.com/sorryKenny)
