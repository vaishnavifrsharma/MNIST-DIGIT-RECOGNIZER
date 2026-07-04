# MNIST KNN Digit Recognizer ✍️🔢

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-KNN-orange)
![Gradio](https://img.shields.io/badge/Gradio-Web%20App-ff7c00)
![Dataset](https://img.shields.io/badge/Dataset-MNIST-purple)
![Status](https://img.shields.io/badge/Status-Interactive%20Demo-brightgreen)

A handwritten digit recognizer built with **classical machine learning** and wrapped inside a clean, interactive **draw-on-web app**.

Instead of uploading an image, you draw a digit directly on the canvas. The app cleans the drawing, converts it into a proper **28 × 28 MNIST-style input**, and predicts the digit using a **K-Nearest Neighbors classifier**.

It is tiny, visual, explainable, and mildly obsessed with handwritten numbers.

---

## Demo Preview

> Add your final app screenshot here after deployment.

```md
![App Screenshot](assets/app_screenshot.png)
```

---

## What This Project Does

- Lets users draw a digit from **0 to 9** on a web canvas
- Converts the drawing into MNIST format using image preprocessing
- Predicts the digit using a trained **KNN model**
- Shows a large predicted digit for quick feedback
- Displays a detailed confidence summary for all digit classes
- Keeps the UI editable through a single CSS block inside `app.py`

---

## Why This Project Is Interesting

MNIST is often used as a beginner machine learning dataset, but this project takes it one step further.

Instead of stopping at notebook accuracy, it turns the model into something interactive:

```text
hand-drawn digit → preprocessing pipeline → KNN prediction → confidence explanation
```

The important learning is not just “train a model.”  
The real challenge is making a model trained on neat MNIST images understand messy human drawings from a web canvas.

That means handling:

- off-center drawings
- thick and thin strokes
- extra whitespace
- inverted colors
- noisy mouse strokes
- shape distortion during resizing

This is where the project becomes more than a dataset exercise. It becomes a tiny end-to-end ML product.

---

## Tech Stack

| Area | Tools Used |
|---|---|
| Language | Python |
| Machine Learning | Scikit-learn |
| Model | K-Nearest Neighbors |
| Dataset | MNIST from OpenML |
| Image Processing | OpenCV, NumPy, Pillow |
| Web App | Gradio |
| Model Saving | Joblib |

---

## Project Structure

```text
MNIST-DIGIT-RECOGNIZER/
│
├── main.py              # trains and saves the KNN model
├── app.py               # Gradio drawing app
├── requirements.txt     # project dependencies
├── knn_mnist.joblib     # saved model generated after training
├── README.md            # project documentation
└── assets/              # screenshots and charts
```

---

## Dataset

| Property | Value |
|---|---|
| Dataset | MNIST |
| Source | `fetch_openml("mnist_784", version=1)` |
| Samples | 70,000 |
| Image Size | 28 × 28 |
| Features | 784 pixel values |
| Classes | Digits 0 to 9 |

Each image is flattened into a 784-dimensional vector, where each value represents one pixel.

---

## Model Used in the Web App

The deployed app uses **K-Nearest Neighbors**.

Current training setup in `main.py`:

| Parameter | Value |
|---|---|
| Model | `KNeighborsClassifier` |
| Neighbors | `k = 3` |
| Weighting | Distance-weighted voting |
| Distance Metric | Euclidean distance |
| Training Samples | 30,000 |
| Test Samples | 8,000 |
| Saved Model | `knn_mnist.joblib` |

KNN does not learn weights like a neural network. It stores training examples and predicts by checking which stored digits are closest to the new drawing.

A simple way to think about it:

```text
“This drawing looks closest to these 3 saved digits. Most of them are 7, so I predict 7.”
```

---

## Preprocessing Pipeline

The web canvas drawing is not naturally in MNIST format, so the app cleans it before prediction.

```text
User drawing
   ↓
Convert to grayscale
   ↓
Invert colors
   ↓
Detect ink pixels
   ↓
Crop around the digit
   ↓
Make the crop square-safe
   ↓
Resize to fit MNIST scale
   ↓
Center using image moments
   ↓
Flatten into 784 features
   ↓
Predict with KNN
```

Why this matters: KNN is distance-based, so even a shifted or stretched digit can confuse it. A clean centered input gives the model a much better chance of making the correct prediction.

---

## Model Comparison From Notebook Experiments

| Rank | Model | Test Setup | Accuracy |
|---|---|---|---:|
| 1 | KNN, k=3 | 10,000 train samples, 2,000 test samples | 94.90% |
| 2 | Logistic Regression | Full test set of 14,000 images | 92.15% |
| 3 | Decision Tree | Full test set of 14,000 images | 87.69% |

KNN performed best in the comparison, which is why it was selected for the interactive drawing app.

---

## KNN Hyperparameter Experiment

| K Value | Accuracy |
|---|---:|
| 1 | 95.60% |
| 3 | 94.90% |
| 5 | 94.80% |
| 7 | 94.05% |

Although `k = 1` gave the highest experimental score, `k = 3` was chosen for the app because it is less sensitive to one noisy nearest neighbor.

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/MNIST-DIGIT-RECOGNIZER.git
cd MNIST-DIGIT-RECOGNIZER
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python main.py
```

This creates:

```text
knn_mnist.joblib
```

### 4. Start the web app

```bash
python app.py
```

Open the local Gradio link, draw a digit, and press **Predict Digit**.

---

## How the App Should Be Used

For best predictions:

- Draw one digit only
- Draw large
- Keep the digit centered
- Avoid fancy handwriting
- Use clear strokes

The app works best when the drawing looks reasonably similar to MNIST-style handwritten digits.

---

## Confidence Summary

The app does not only show the predicted digit. It also shows a confidence breakdown.

Example:

```text
Predicted digit: 8
Top confidence: 86.4%
Second guess: 3
Confidence gap: 31.2%
```

This helps show when the model is certain and when it is confused between visually similar digits.

---

## Key Learnings

- MNIST images are represented as 784-dimensional vectors
- Normalization improves model behavior
- KNN is simple but powerful for image similarity tasks
- KNN becomes slower as training data increases
- Web drawings need careful preprocessing before prediction
- Centering, cropping, and resizing matter a lot for distance-based models
- Confidence scores make predictions easier to interpret
- Turning a notebook model into a web app teaches real ML deployment workflow

---

## Limitations

This project intentionally uses classical ML, so it has some limits:

- KNN prediction slows down with larger training sets
- Unusual handwriting can confuse the model
- Very tiny or off-center drawings may reduce accuracy
- The model compares raw pixel distances, not high-level digit features
- It is less robust than a CNN for real-world handwriting variation

These limitations are not failures. They are part of the learning story.

---

## Future Improvements

- Deploy the app on Hugging Face Spaces
- Add example screenshots to the README
- Add a small gallery of correct and wrong predictions
- Add a toggle to show/hide the 28 × 28 processed input
- Improve preprocessing for thin strokes and unusual handwriting
- Add a short model explanation section inside the app
- Compare app predictions with notebook test-set results

---

## Final Takeaway

This project starts with a classic dataset, but turns it into a hands-on ML demo.

It shows the full path from:

```text
raw pixels → model training → saved model → web interface → live prediction
```

The result is a compact, explainable digit recognizer that makes machine learning feel less like a black box and more like a tiny number detective with a magnifying glass.

---

## Author

**Vaishnavi Sharma**

Built as part of a hands-on machine learning learning path using MNIST, classical ML models, and interactive web deployment.
