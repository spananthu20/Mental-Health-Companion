# 🧠 Mental Health Companion

### Voice Emotion Detection Using Python and Librosa

## 📌 Project Overview

Mental Health Companion is a machine learning project that explores voice-based emotion recognition using audio processing techniques and Python.

The project analyzes audio recordings, extracts relevant audio features, and uses a trained machine learning model to predict emotion labels from voice recordings.

The goal is to explore how speech signals and machine learning can support emotion-aware applications.

**Important:** Voice emotion recognition is experimental. It cannot reliably determine a person's mental health condition or replace professional mental health assessment.

## 🎯 Objectives

* Process audio recordings using Python.
* Explore voice signals and their characteristics.
* Extract meaningful audio features for emotion recognition.
* Train and use a machine learning model.
* Understand how audio processing and machine learning can be combined in a practical application.

## 🛠️ Technologies Used

* **Python** — Programming language
* **Librosa** — Audio processing and feature extraction
* **NumPy** — Numerical computing
* **Pandas** — Dataset handling
* **Scikit-learn** — Machine learning tools, if used in the implementation
* **Jupyter Notebook** — Development and experimentation
* **Pickle** — Model serialization and loading

*The final list of dependencies should be adjusted to match the libraries actually used in the notebook.*

## 📂 Repository Structure

```text
Mental-Health-Companion/
│
├── Mental_Health_Companion.ipynb
├── README.md
├── emotion_dataset.csv
├── emotion_model.pkl
└── voice_waveform.png
```

## ⚙️ Project Workflow

### 1. Audio Data Collection

The project works with audio recordings in WAV format.

### 2. Audio Processing

Librosa is used to load audio recordings and represent them as numerical signals for further analysis.

### 3. Feature Extraction

Audio characteristics are extracted to provide numerical inputs for emotion classification.

The exact features used should be documented after verifying the implementation in the notebook.

### 4. Model Training

A machine learning model is trained using emotion-related data.

### 5. Emotion Prediction

The saved model can be used to predict emotion labels from audio features, provided the input features match the model's expected format.

### 6. Evaluation

Model performance should be evaluated using appropriate classification metrics and a separate test dataset.

## 🚀 Getting Started

### Prerequisites

Install Python and Jupyter Notebook.

### Installation

Install the required libraries:

```bash
pip install numpy pandas librosa scikit-learn matplotlib seaborn jupyter
```

Additional libraries may be required depending on the notebook implementation.

### Run the Project

1. Clone or download this repository.
2. Install the required dependencies.
3. Obtain the audio dataset and place it in the appropriate directory.
4. Update the dataset path in the notebook.
5. Open `Mental_Health_Companion.ipynb` in Jupyter Notebook or Google Colab.
6. Run the cells in order.

**Note:** The notebook currently contains a local dataset path. Update it before running the project on another computer.

## 📊 Model Evaluation

The model's performance should be reported using appropriate metrics, such as:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

Actual evaluation results should be added after running the model on a suitable test dataset.

## 🔒 Ethical Considerations and Limitations

* Emotion predictions from voice can be inaccurate.
* Speech patterns may vary with language, accent, recording quality, and individual differences.
* Predicted emotions should not be treated as a diagnosis of depression, anxiety, or another mental health condition.
* Audio recordings and datasets should be handled with appropriate privacy protections.
* The system should be used as an experimental learning project rather than a clinical decision-making tool.

## 🔮 Future Improvements

* Build an interactive user interface.
* Improve audio preprocessing and feature extraction.
* Evaluate multiple machine learning algorithms.
* Add comprehensive model evaluation and error analysis.
* Improve reproducibility through configuration files and dependency management.
* Explore responsible ways to integrate emotion recognition into supportive applications.

## 👨‍💻 Author

**Ananthu S P**

M.Sc. Data Science Student

Vellore Institute of Technology (VIT), Vellore

---

*This project is intended for educational and research purposes.*
