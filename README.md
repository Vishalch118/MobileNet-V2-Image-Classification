# 🐶🐱 MobileNest V2 – Cat vs Dog Image Classification

This project demonstrates how to perform **image classification** using **MobileNetV2**, a state-of-the-art **Convolutional Neural Network (CNN)** architecture pre-trained on ImageNet.  
By leveraging **transfer learning**, this notebook efficiently distinguishes between **cat and dog** images with high accuracy and reduced training time.

---

## 🧩 Introduction

**MobileNest V2** applies **deep learning** techniques to classify images from the famous **Dogs vs. Cats** dataset.  
It showcases the full workflow — from data extraction and preprocessing to model training, evaluation, and visualization — using **TensorFlow** and **Keras**.

The notebook walks through the complete process:

- Downloading and extracting the dataset from Kaggle  
- Visualizing the dataset and image distribution  
- Preprocessing and resizing all images to 224×224 px  
- Implementing **MobileNetV2** with transfer learning  
- Training and evaluating the model  
- Visualizing accuracy and loss curves

---

## 📊 Dataset

The project uses the **Dogs vs. Cats dataset** from **Kaggle**, which includes:
- **25,000+ labeled images** (cats and dogs)
- Divided into training and validation sets
- Each image resized to **224×224 pixels** for compatibility with MobileNetV2

**Dataset Source:**  
🔗 [Dogs vs. Cats – Kaggle Competition](https://www.kaggle.com/competitions/dogs-vs-cats)

---

## ⚙️ Features

- ✅ **Transfer Learning** with MobileNetV2 for faster training and better generalization  
- 📈 **Training Visualization** using Matplotlib  
- 🧹 **Automated Data Preprocessing** (resizing, normalization, batching)  
- 🔍 **Evaluation Metrics:** Accuracy, Loss, and Confusion Matrix  
- 🧠 **Fine-tuned Layers** for improved classification performance  

---

## 💻 Installation

To run this project locally:

```bash
# Clone the repository
git clone https://github.com/Vishalch118/MobileNet-V2-Image-Classification

# Navigate into the project directory
cd MobileNet-V2-Image-Classification
```

## Dependencies

Ensure the following Python libraries are installed:
```bash
pip install tensorflow keras numpy matplotlib scikit-learn pillow kaggle
```

## How to Run

Download the dataset using the Kaggle API.

Extract the files and organize them into training and validation folders.

Open the notebook in Jupyter or Google Colab.

Run all cells sequentially to train and evaluate the model.


## 📈 Results

After training, the notebook produces:

Model Accuracy Plot and Loss Curve across epochs

Confusion Matrix showing classification performance

Predicted vs. Actual Visualization for sample images


## 👥 Contributors

Vishal Ch — GitHub Profile

Contributions are welcome! Feel free to open issues or submit pull requests to improve this project.


## 🪪 License

This project is licensed under the MIT License.
See the LICENSE file for details.


## 🔗 References

TensorFlow Documentation: https://www.tensorflow.org/api_docs

Kaggle Dataset: Dogs vs. Cats

MobileNetV2 Paper: https://arxiv.org/abs/1801.04381

Matplotlib Documentation: https://matplotlib.org/

⭐ If you found this project helpful, consider starring the repository!

