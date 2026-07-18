<p align="center">
  <img src="assets/banner.png" width="90%">
</p>


### **PneumoScan-Junior**
AI-powered web application for automatic pediatric chest X-ray classification using a deep learning model based on **Xception CNN** with **Grad-CAM explainability**, developed for educational and research purposes.



<p align="center">

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.49-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</p>




---
### 📖 Project Overview

Pediatric pneumonia remains one of the leading causes of illness and mortality among children worldwide. Chest X-ray imaging is one of the primary diagnostic tools used to identify pneumonia; however, interpreting radiographs requires clinical expertise and may be challenging in resource-limited settings.

PneumoScan Junior is an end-to-end deep learning application that demonstrates how artificial intelligence can assist in the automated classification of pediatric chest X-ray images. The application integrates a pretrained **Xception Convolutional Neural Network (CNN)** with an interactive Streamlit interface, enabling users to upload chest radiographs and obtain real-time predictions.

Beyond image classification, the application incorporates **Gradient-weighted Class Activation Mapping (Grad-CAM)** to visualize the image regions that most strongly influence the model's prediction. This explainability component provides greater transparency into the decision-making process and helps users better understand how the model reaches its conclusions.


### ✨ Features

- Upload pediatric chest X-ray images
- Automatic pneumonia prediction
- Prediction confidence score
- Explainable AI using Grad-CAM
- Sample image gallery
- Streamlit web interface

### 📸 Application Preview

**🏠 Home Interface**
> Upload a pediatric chest X-ray or select one of the provided sample images.
<p align="center">
  <img src="assets/v3_home.PNG" width="60%">
</p>

**🤖 AI Prediction**
> The Xception CNN predicts Normal or Pneumonia and reports the associated confidence score.
<p align="center">
  <img src="assets/v3_PneumoScanJ_prediction_demo_pneumonia_case.PNG" width="60%">
</p>

**🔥 Explainable AI (Grad-CAM)**
> Grad-CAM highlights the image regions that most influenced the model's prediction.
<p align="center">
  <img src="assets/v3_PneumoScanJ_gradcam_demo_pneumonia_case.PNG" width="60%">
</p>



### 🌐 Live Demo

**Try the application here:**

**🔗 https://pneumoscan-junior.streamlit.app**

---

>**⚠️ Disclaimer**
>
>PneumoScan Junior is intended **for educational and research purposes only**.
>It is **not** a medical device and **must not** be used for clinical diagnosis or treatment decisions.


