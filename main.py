import joblib
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


MODEL_PATH = "knn_mnist.joblib"

# we can increase it later to 60000 for better accuracy but slower prediction.
TRAIN_LIMIT = 30000
TEST_LIMIT = 8000


def main():
    print("Downloading MNIST dataset...")

    mnist = fetch_openml("mnist_784", version=1, as_frame=False)

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    print("Dataset loaded:", X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=TRAIN_LIMIT,
        test_size=TEST_LIMIT,
        random_state=42,
        stratify=y
    )

    print("Training KNN model...")

    knn = KNeighborsClassifier(
        n_neighbors=3,
        weights="distance",
        metric="euclidean",
        n_jobs=-1
    )

    knn.fit(X_train, y_train)

    print("Testing model...")
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    model_package = {
        "model": knn,
        "accuracy": accuracy,
        "train_limit": TRAIN_LIMIT,
        "pixel_scale": "0_to_1",
        "input_shape": "28x28 flattened"
    }

    joblib.dump(model_package, MODEL_PATH)

    print(f"Model saved as {MODEL_PATH}")


if __name__ == "__main__":
    main()
