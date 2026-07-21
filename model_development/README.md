# 🧠 Model Development (under construction)

This directory documents the research and development process of the deep learning model used in the **PneumoScan Junior App**.

Five convolutional neural network (CNN) architectures were developed and evaluated for automated pediatric pneumonia detection from chest X-ray images. Rather than selecting an architecture from the outset, each candidate model was trained using a consistent pipeline and compared using multiple evaluation metrics. The objective was to identify a model that achieved strong classification performance while also providing meaningful explainability through Gradient-weighted Class Activation Mapping (Grad-CAM).

Because the dataset is imbalanced, Balanced Accuracy and ROC-AUC were considered the primary model selection criteria, while Accuracy, Precision, Sensitivity, and Specificity were used as supporting metrics.

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

Before training, the following preprocessing pipeline was applied:

- Image resizing
- Pixel normalization
- Data augmentation (selected experiments)
- Batch preparation using TensorFlow/Keras generators
- Binary class encoding (Normal vs Pneumonia)

---

## 🏗️ Candidate Models

Five CNN architectures were investigated throughout the development process.

| Model | Description |
|--------|-------------|
| CNN Baseline | A custom convolutional neural network used as the initial benchmark. |
| CNN + Data Augmentation | Baseline CNN combined with image augmentation techniques to improve robustness. |
| CNN + Batch Normalization | Enhanced CNN architecture incorporating Batch Normalization layers for improved optimization and training stability. |
| VGG16 Transfer Learning | Transfer learning using the pretrained VGG16 architecture. |
| Xception Transfer Learning | Transfer learning using the pretrained Xception architecture, which ultimately demonstrated the best overall performance. |

All models were trained using the same preprocessing pipeline and evaluated independently using identical evaluation metrics.





---

## Model Architecture

The final model is based on the pretrained Xception architecture with a lightweight custom classification head.



```text
                     Input Image
                  (224 × 224 × 3)
                           │
                           ▼
               Pretrained Xception CNN
              (Image Feature Extraction)
                           │
                           ▼
                      Flatten Layer
                           │
                           ▼
                  Fully Connected (198)
                           │
                           ▼
                  Fully Connected (128)
                           │
                           ▼
                         Dropout
                           │
                           ▼
                 Output Layer (Sigmoid)
                           │
                           ▼
          Normal        or       Pneumonia
```



<p align="center">
<img src="figures/https://github.com/user-attachments/assets/d2e3cecd-786e-47b9-82af-373c92fd2469" width="650">
</p>

<img width="424" height="216" alt="image" src="https://github.com/user-attachments/assets/4310d4b7-5b36-415d-90ff-bf9e4de2e5c7" />


## 📈 Training Process


<p align="center">
    <img src="figures/training_curve.png" width="850">
</p>

<p align="center">
<i>Figure 2. Training and validation loss and accuracy of the final Xception model.</i>
</p>

Figure 2 illustrates the learning behavior of the final Xception model. Training and validation accuracy increased consistently while the corresponding losses decreased rapidly during the early epochs before stabilizing. These curves indicate successful optimization under the selected training configuration.





## 📈 Model Evaluation

Receiver Operating Characteristic (ROC) analysis was performed to evaluate the model's ability to distinguish between Normal and Pneumonia chest X-rays across different decision thresholds.

<p align="center">
    <img src="figures/roc_curve.png" width="850">
</p>

<p align="center">
<i>Figure 3. Training and validation loss and accuracy of the final Xception model.</i>
</p>

Figure 3 compares the predictive performance of the five candidate architectures. Since the dataset is imbalanced, Balanced Accuracy and ROC-AUC were prioritized during model selection.

Xception achieved the highest Accuracy (83.49%), Precision (79.96%), Specificity (58.97%), and Balanced Accuracy (78.59%). Although VGG16 produced the highest ROC-AUC (96.08%), its substantially lower specificity resulted in inferior Balanced Accuracy. Consequently, Xception was selected as the final deployment model.


---

## 🔥 Explainability with Grad-CAM

To improve model transparency, Gradient-weighted Class Activation Mapping (Grad-CAM) was incorporated into the development pipeline.

Grad-CAM generates visual heatmaps highlighting image regions that most strongly influenced the model's prediction.

This provides qualitative insight into model behavior and helps users better understand the decision-making process.

### True Positive Example

<p align="center">
    <img src="figures/pneumonia_gradcam.png" width="1000">
</p>

<p align="center">
<i>Figure 4. Grad-CAM comparison for a correctly classified pneumonia case.</i>
</p>

### True Negative Example

<p align="center">
    <img src="figures/normal_gradcam.png" width="1000">
</p>

<p align="center">
<i>Figure 5. Grad-CAM comparison for a correctly classified normal chest X-ray.</i>
</p>

Figure Grad-CAM

> Grad-CAM illustrates model attention rather than providing a definitive localization of disease and should not be interpreted as a clinical diagnosis.
>
To complement the quantitative evaluation, Grad-CAM was used to visualize the image regions contributing most strongly to each prediction. Compared with the other candidate architectures, Xception generally produced more localized and anatomically plausible activation patterns for both pneumonia and normal chest radiographs.

---

# 📓 Training Notebook

The complete development workflow, including preprocessing, model training, evaluation, and Grad-CAM implementation, is documented in the accompanying Jupyter notebook. The notebook represents the original experiments used to produce the final deployed Xception model.

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

> **Note**
>
> The notebook contained in this directory represents the original training pipeline used to develop the final model. Because deep learning training involves stochastic processes (random initialization, GPU nondeterminism, library versions, and data augmentation), rerunning the notebook may produce slightly different numerical results while maintaining similar overall performance trends.

