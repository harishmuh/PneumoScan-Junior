<img width="507" height="633" alt="image" src="https://github.com/user-attachments/assets/8ee55a71-60f1-4392-b9dc-2fe51a099fac" /># 🧠 Model Development

This directory documents the development process of the deep learning model used in the **PneumoScan Junior App**.

Rather than selecting a single architecture from the outset, multiple convolutional neural network (CNN) models were developed, trained, evaluated, and compared using the same pediatric chest X-ray dataset. The objective was to identify a model that achieved strong classification performance while also providing meaningful explainability through Gradient-weighted Class Activation Mapping (Grad-CAM).

> **Note**
>
> The notebook contained in this directory represents the original training pipeline used to develop the final model. Because deep learning training involves stochastic processes (random initialization, GPU nondeterminism, library versions, and data augmentation), rerunning the notebook may produce slightly different numerical results while maintaining similar overall performance trends.

---

## 📂 Dataset

The model was trained using a publicly available pediatric chest X-ray dataset containing two classes:

- Normal
- Pneumonia

The dataset consists of pediatric anterior–posterior chest radiographs collected for research purposes and is widely used for benchmarking deep learning models in pneumonia classification.

The dataset was divided into:

- Training set
- Validation set
- Test set

The test set remained completely unseen during model development and was used only for final evaluation.

---

## ⚙️ Preprocessing

Prior to training, the following preprocessing pipeline was applied:

- Image resizing
- Pixel normalization
- Data augmentation (selected experiments)
- Batch preparation using TensorFlow/Keras generators
- Binary class encoding (Normal vs Pneumonia)

Several experiments explored different preprocessing strategies to improve model generalization while reducing overfitting.

---

## 🏗️ Five Candidate Models

Five CNN architectures were investigated throughout the development process.

| Model | Description |
|--------|-------------|
| CNN Baseline | A custom convolutional neural network used as the initial benchmark. |
| CNN + Data Augmentation | Baseline CNN combined with image augmentation techniques to improve robustness. |
| CNN + Batch Normalization | Enhanced CNN architecture incorporating Batch Normalization layers for improved optimization and training stability. |
| VGG16 Transfer Learning | Transfer learning using the pretrained VGG16 architecture. |
| Xception Transfer Learning | Transfer learning using the pretrained Xception architecture, which ultimately demonstrated the best overall performance. |

Each model was trained and evaluated independently using identical evaluation metrics.




---

## 📊 Performance Comparison

Model performance was assessed using multiple evaluation metrics rather than classification accuracy alone.

Evaluation included:

- Accuracy
- Precision
- Recall (Sensitivity)
- Specificity
- F1-score
- ROC-AUC
- Confusion Matrix
- Grad-CAM explainability

The comparison of multiple architectures helped ensure that the final model selection was based on overall predictive performance rather than a single metric.

> *(Optional: Insert your comparison table or performance figure here.)*

---

## ✅ Why Xception Was Selected

Among the evaluated architectures, **Xception** demonstrated the strongest overall performance.

The final model was selected because it provided:

- High classification accuracy
- Strong ROC-AUC performance
- Balanced sensitivity and specificity
- Stable validation performance
- Better feature representation through transfer learning
- Clear and interpretable Grad-CAM visualizations

These characteristics made Xception the most suitable architecture for deployment within the PneumoScan Junior application.

---

## Model Architecture

The final model is based on the pretrained Xception architecture with a lightweight custom classification head.

Input (224×224×3)
        │
        ▼
Pretrained Xception
(Feature Extraction)
        │
        ▼
Flatten
        │
        ▼
Dense (198)
        │
        ▼
Dense (128)
        │
        ▼
Dropout
        │
        ▼
Sigmoid Output


## Training curves

Figure Training curves

The training history indicates rapid convergence within the first few epochs.

Training and validation accuracy increased consistently, while the corresponding losses decreased and stabilized without evidence of severe overfitting.



## ROC Curve

Figure ROC curve

---

## 🔥 Explainability with Grad-CAM

To improve model transparency, Gradient-weighted Class Activation Mapping (Grad-CAM) was incorporated into the development pipeline.

Grad-CAM generates visual heatmaps highlighting image regions that most strongly influenced the model's prediction.

This provides qualitative insight into model behavior and helps users better understand the decision-making process.

Figure Grad CAM

> Grad-CAM illustrates model attention rather than providing a definitive localization of disease and should not be interpreted as a clinical diagnosis.

---

# 📓 Training Notebook

The complete development workflow is documented in:

```
notebooks/Pneumonia_image_classification.ipynb
```

The notebook includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Five CNN experiments
- Hyperparameter configuration
- Model training
- Performance evaluation
- Confusion matrices
- ROC analysis
- Grad-CAM implementation
- Final model selection

The notebook is preserved as an archival record of the original experiments that produced the deployed Xception model used by **PneumoScan Junior**.
