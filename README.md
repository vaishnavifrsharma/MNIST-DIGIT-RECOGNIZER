# MNIST KNN Digit Recognizer ✍️🔢

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-KNN-orange)
![Gradio](https://img.shields.io/badge/Gradio-Web%20App-ff7c00)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-purple)
![Status](https://img.shields.io/badge/Status-Deployed-brightgreen)

An interactive handwritten digit recognizer built with **K-Nearest Neighbors**, **Scikit-learn**, **OpenCV**, and **Gradio**.

Draw a digit directly on the web canvas. The app cleans the drawing, converts it into a **28 × 28 MNIST-style image**, and predicts the digit using a trained KNN classifier.

---

## Live Demo

Try the deployed app here:

👉 [MNIST KNN Digit Recognizer on Hugging Face Spaces](https://huggingface.co/spaces/vaishnavifr/mnist-knn-digit-recognizer)

---

## Project Overview

This project takes the classic MNIST digit recognition task beyond a notebook and turns it into a working web app.

**Pipeline:**

Draw digit → preprocess image → convert to 28 × 28 MNIST format → predict with KNN → show confidence summary

The main learning goal was not only to train a model, but to understand how a machine learning model can be saved, loaded, deployed, and used through an interactive interface.

---

## Features

- Draw digits directly on a web canvas
- Predict handwritten digits from **0 to 9**
- Use a trained KNN model
- Convert drawings into MNIST-style **28 × 28** inputs
- Show the processed image that the model actually sees
- Display confidence scores for all digit classes
- Use a clean custom Gradio interface
- Deploy the app on Hugging Face Spaces

---

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn |
| Model | K-Nearest Neighbors |
| Dataset | MNIST via OpenML |
| Image Processing | OpenCV, NumPy, Pillow |
| Web App | Gradio |
| Model Saving | Joblib |
| Deployment | Hugging Face Spaces |

---

## Project Structure

| File / Folder | Purpose |
|---|---|
| `app.py` | Gradio web app |
| `main.py` | Trains and saves the KNN model |
| `knn_mnist.joblib` | Saved trained KNN model |
| `requirements.txt` | Project dependencies |
| `MNIST_pynb.ipynb` | Notebook experiments |
| `assets/` | Screenshots and charts |
| `LICENSE` | MIT license |

---

## Dataset

The project uses the MNIST dataset from OpenML.

| Property | Value |
|---|---|
| Dataset | MNIST |
| Source | `fetch_openml("mnist_784", version=1)` |
| Total Samples | 70,000 |
| Image Size | 28 × 28 |
| Features | 784 pixel values |
| Classes | Digits 0 to 9 |

Each image is flattened into a 784-dimensional vector, where each value represents one grayscale pixel.

---

## Model Used

The deployed app uses a **K-Nearest Neighbors classifier**.

| Parameter | Value |
|---|---|
| Model | `KNeighborsClassifier` |
| Neighbors | `k = 3` |
| Weighting | Distance-weighted voting |
| Distance Metric | Euclidean distance |
| Training Samples | 30,000 |
| Test Samples | 8,000 |
| Saved File | `knn_mnist.joblib` |

KNN predicts a digit by comparing the user drawing with stored MNIST examples and choosing the class of the nearest examples.

In simple terms:

> “This drawing looks closest to these saved digits, so I will predict the most similar class.”

---

## Preprocessing Pipeline

The web drawing is not automatically in MNIST format, so it must be cleaned before prediction.

| Step | Operation |
|---|---|
| 1 | Take user drawing from canvas |
| 2 | Convert to grayscale |
| 3 | Invert colors to match MNIST |
| 4 | Detect ink pixels |
| 5 | Crop around the digit |
| 6 | Resize while preserving shape |
| 7 | Center using image moments |
| 8 | Convert to 28 × 28 image |
| 9 | Flatten into 784 features |
| 10 | Predict with KNN |

This step is important because KNN is distance-based. If the drawing is too small, off-center, or poorly cropped, the distance comparison becomes less reliable.

---

## Notebook Model Comparison

The notebook compares multiple classical ML models on MNIST.

| Rank | Model | Test Setup | Accuracy |
|---|---|---|---:|
| 1 | KNN, k=3 | 10,000 train samples, 2,000 test samples | 94.90% |
| 2 | Logistic Regression | Full test set of 14,000 images | 92.15% |
| 3 | Decision Tree | Full test set of 14,000 images | 87.69% |

KNN performed best in the comparison, so it was selected for the deployed interactive app.

---

## KNN Hyperparameter Experiment

| K Value | Accuracy |
|---|---:|
| 1 | 95.60% |
| 3 | 94.90% |
| 5 | 94.80% |
| 7 | 94.05% |

Although `k = 1` gave the highest score in one experiment, `k = 3` was chosen because it is less sensitive to one noisy nearest neighbor.

---

## How to Run Locally

### 1. Clone the repository

`git clone https://github.com/vaishnavifrsharma/MNIST-DIGIT-RECOGNIZER.git`

`cd MNIST-DIGIT-RECOGNIZER`

### 2. Install dependencies

`pip install -r requirements.txt`

### 3. Train the model

`python main.py`

This creates the saved model file:

`knn_mnist.joblib`

### 4. Start the app

`python app.py`

Then open the Gradio link, draw a digit, and press **Predict Digit**.

---

## How to Use the App

For better predictions:

- Draw one digit only
- Draw large and centered
- Use clear strokes
- Avoid very tiny digits
- Keep the digit similar to MNIST-style handwriting

The app shows a processed **28 × 28** preview. If that preview looks clean, the prediction is more likely to be correct.

---

## Limitations

This project intentionally uses classical machine learning, so it has some limitations:

- KNN can become slow with larger training sets
- Unusual handwriting may confuse the model
- Very small or off-center drawings reduce prediction quality
- Raw pixel distance is less robust than CNN-based feature learning
- The saved KNN model is relatively large because KNN stores training examples

These limitations are part of the learning story and show the trade-offs between simplicity, interpretability, speed, and accuracy.

---

## Future Improvements

- Add example screenshots to the README
- Add a gallery of correct and wrong predictions
- Improve preprocessing for thin or unusual strokes
- Add a toggle to hide/show the processed 28 × 28 image
- Compare KNN with a CNN-based web model
- Add more explanation inside the app for beginner users

---

## Key Learnings

- MNIST images can be represented as 784-dimensional vectors
- Normalization improves model behavior
- KNN is simple, explainable, and effective for MNIST
- Image preprocessing is essential when using user-drawn inputs
- Model deployment requires more than notebook accuracy
- A saved ML model can be turned into an interactive web app using Gradio
- Hugging Face Spaces can host small ML demos cleanly

---

## Final Takeaway

This project starts with a classic dataset and turns it into a complete mini ML product.

**Raw pixels → model training → saved model → web app → live prediction**

It demonstrates the full workflow from experimentation to deployment, while keeping the model understandable and interactive.

---

## Author

**Vaishnavi Sharma**

Built as part of a hands-on machine learning learning path using MNIST, classical ML models, Gradio, and Hugging Face Spaces.
