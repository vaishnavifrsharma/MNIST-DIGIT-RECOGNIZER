# MNIST Digit Recognizer

This project compares classical machine learning models for handwritten digit recognition using the MNIST dataset.

## Models Implemented

- Logistic Regression
- K-Nearest Neighbors

## Current Results

| Model | Test Setup | Accuracy |
|---|---|---:|
| Logistic Regression | Full test set of 14,000 images | 92.15% |
| KNN, k=3 | 10,000 train samples, 2,000 test samples | 94.90% |

## KNN Hyperparameter Results

| K Value | Accuracy |
|---:|---:|
| 1 | 95.60% |
| 3 | 94.90% |
| 5 | 94.80% |
| 7 | 94.05% |

## Key Learnings

- MNIST images are represented as 784 pixel values.
- Normalization improves model training and distance-based comparison.
- Logistic Regression is a strong baseline for multiclass classification.
- KNN performs better on this dataset but has slower prediction time.

## Next Steps

- Add Decision Tree model
- Compare all three models
- Add confusion matrices and error analysis
- Test custom handwritten digit images
