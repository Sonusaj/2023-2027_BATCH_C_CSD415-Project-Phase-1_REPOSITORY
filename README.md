# 🩺 Diabetic Retinopathy Detection Using Deep Learning

A deep learning-based project for **automated detection and classification of Diabetic Retinopathy (DR)** from retinal fundus images. The system uses image preprocessing and a convolutional neural network with a **Multi-Scale Attention Gate (MSAG)** module to improve feature extraction and classification performance.

## 📌 Project Overview

**Diabetic Retinopathy** is a diabetes-related eye disease that can cause vision loss if not detected and treated at an early stage.

This project aims to develop an automated system that analyzes retinal fundus images and classifies them according to the severity of diabetic retinopathy.

The project pipeline consists of:

**Input Retinal Image → Preprocessing → Feature Extraction → MSAG Attention → Classification → DR Prediction**

## 🎯 Objectives

- Detect diabetic retinopathy from retinal fundus images.
- Classify retinal images based on the severity of the disease.
- Improve important feature extraction using attention mechanisms.
- Reduce the effect of image noise and variations through preprocessing.
- Provide an automated tool that can assist in preliminary DR screening.

## 🧠 Model Architecture

The project uses a deep learning architecture incorporating **Multi-Scale Attention Gates (MSAG)**.

The major components include:

- Image preprocessing
- Convolutional feature extraction
- Multi-scale feature processing
- Attention-based feature refinement
- Classification layers
- Final diabetic retinopathy prediction

### Workflow

```text
Retinal Fundus Image
        ↓
Image Preprocessing
        ↓
Feature Extraction
        ↓
Multi-Scale Attention Gate (MSAG)
        ↓
Deep Feature Representation
        ↓
Classification Layer
        ↓
DR Severity Prediction
```

## 📂 Project Structure

```text
Diabetic-Retinopathy/
│
├── data/
│   └── dataset/
│
├── src/
│   ├── preprocessing.py
│   ├── msag.py
│   ├── dataset.py
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── models/
│   └── model files
│
├── results/
│   ├── confusion_matrix/
│   ├── plots/
│   └── predictions/
│
├── requirements.txt
├── README.md
└── .gitignore
```

> The exact file structure may vary depending on the final implementation.

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Pandas**

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/diabetic-retinopathy.git
cd diabetic-retinopathy
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment.

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 📊 Dataset

The model requires a dataset containing **retinal fundus images** labeled according to diabetic retinopathy severity.

A commonly used dataset for this task is the **APTOS 2019 Blindness Detection Dataset**.

The dataset should be organized according to the structure expected by `dataset.py` and the configuration specified in `config.py`.

> Dataset files are not included in this repository because of their size and licensing restrictions.

## 🔄 Image Preprocessing

The preprocessing pipeline prepares retinal images before they are provided to the model.

Typical preprocessing operations include:

- Image resizing
- Color normalization
- Noise reduction
- Contrast enhancement
- Pixel normalization
- Conversion into the required input format

The preprocessing implementation is available in:

```text
src/preprocessing.py
```

## 🚀 Training the Model

After configuring the dataset path and training parameters, run:

```bash
python train.py
```

The training process generates the trained model and training statistics.

Model-related configuration can be modified in:

```text
config.py
```

## 🧪 Model Evaluation

To evaluate the trained model on the test dataset:

```bash
python evaluate.py
```

The evaluation can generate metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

## 🔍 Making Predictions

To predict the diabetic retinopathy class of an individual retinal image:

```bash
python predict.py --image path/to/image.jpg
```

The system preprocesses the input image and passes it through the trained model to obtain the predicted class.

## 📈 Results

The performance of the model can be evaluated using standard classification metrics.

Example:

```text
Accuracy  : XX.XX%
Precision : XX.XX%
Recall    : XX.XX%
F1-Score  : XX.XX%
```

> Replace these values with the actual results obtained from your trained model.

## 🩻 DR Severity Classes

Depending on the dataset configuration, the model can classify retinal images into the following stages:

| Class | Description |
|---|---|
| 0 | No Diabetic Retinopathy |
| 1 | Mild Diabetic Retinopathy |
| 2 | Moderate Diabetic Retinopathy |
| 3 | Severe Diabetic Retinopathy |
| 4 | Proliferative Diabetic Retinopathy |

## 🔬 Attention Mechanism

The **Multi-Scale Attention Gate (MSAG)** helps the network focus on informative regions and features within retinal images.

This is particularly useful for diabetic retinopathy because important pathological signs can appear at different scales, including:

- Microaneurysms
- Hemorrhages
- Exudates
- Cotton-wool spots
- Abnormal blood vessels

The attention mechanism allows the network to emphasize relevant visual features while reducing the influence of less informative regions.

## 📌 Applications

This project can be used as a foundation for:

- Automated diabetic retinopathy screening
- Computer-aided ophthalmic diagnosis
- Retinal image analysis
- Medical image classification research
- Deep learning research in ophthalmology

## ⚠️ Disclaimer

This project is intended **for educational and research purposes only**.

It is not a certified medical diagnostic system and should not be used as a substitute for examination or diagnosis by a qualified healthcare professional.

## 🔮 Future Improvements

Possible future enhancements include:

- Improving model accuracy and generalization.
- Using larger and more diverse retinal datasets.
- Implementing advanced data augmentation techniques.
- Adding explainable AI techniques such as Grad-CAM.
- Developing a web-based prediction interface.
- Deploying the model for real-time screening assistance.
- Comparing MSAG with other attention mechanisms.
- Optimizing the model for mobile or edge-device deployment.

## 👨‍💻 Project

**Project:** Diabetic Retinopathy Detection Using Deep Learning  
**Domain:** Deep Learning / Computer Vision / Medical Image Analysis  
**Framework:** TensorFlow & Keras  
**Language:** Python

## 📜 License

This project is intended for educational and research purposes. Add an appropriate open-source license if you plan to distribute the code publicly.

---

⭐ **If you find this project useful, consider giving the repository a star!**
