import joblib
import numpy as np
import gradio as gr
import cv2

from PIL import Image


MODEL_PATH = "knn_mnist.joblib"


# ----------------------------
# Load trained KNN model
# ----------------------------
model_package = joblib.load(MODEL_PATH)
knn = model_package["model"]


# ----------------------------
# Editable design section
# Change colors/fonts here
# ----------------------------
APP_CSS = """
:root {
    --bg-dark: #09090f;
    --card: rgba(255, 255, 255, 0.08);
    --card-border: rgba(255, 255, 255, 0.18);
    --accent: #a78bfa;
    --accent-2: #22d3ee;
    --text-main: #f8fafc;
    --text-soft: #cbd5e1;
}

.gradio-container {
    background:
        radial-gradient(circle at top left, rgba(167, 139, 250, 0.22), transparent 32%),
        radial-gradient(circle at bottom right, rgba(34, 211, 238, 0.18), transparent 30%),
        var(--bg-dark) !important;
    color: var(--text-main) !important;
    font-family: Inter, system-ui, sans-serif !important;
}

#hero {
    text-align: center;
    padding: 24px 16px 10px 16px;
}

#hero h1 {
    font-size: 48px;
    font-weight: 900;
    margin-bottom: 6px;
    letter-spacing: -1px;
}

#hero p {
    color: var(--text-soft);
    font-size: 17px;
}

#main-card {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 28px;
    padding: 22px;
    box-shadow: 0 20px 70px rgba(0, 0, 0, 0.35);
}

#draw-box {
    border-radius: 24px;
    overflow: hidden;
}

#predict-btn {
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: #050505 !important;
    font-weight: 800 !important;
    border-radius: 16px !important;
    border: none !important;
}

#clear-btn {
    border-radius: 16px !important;
}

#digit-output textarea {
    font-size: 72px !important;
    font-weight: 900 !important;
    text-align: center !important;
}

#footer-note {
    text-align: center;
    color: var(--text-soft);
    font-size: 14px;
}
"""


# ----------------------------
# White canvas for drawing
# ----------------------------
def blank_canvas():
    canvas = np.ones((280, 280, 3), dtype=np.uint8) * 255

    return {
        "background": canvas,
        "layers": [],
        "composite": canvas
    }


# ----------------------------
# Extract actual canvas image from Gradio
# ----------------------------
def extract_canvas(editor_value):
    if editor_value is None:
        return None

    if isinstance(editor_value, dict):
        image_array = editor_value.get("composite")
    else:
        image_array = editor_value

    if image_array is None:
        return None

    image_array = image_array.astype("uint8")

    # Remove alpha channel if present
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    return image_array


# ----------------------------
# Center digit using center of mass
# ----------------------------
def center_digit(image_28):
    """
    image_28 is a 28x28 grayscale image.
    Digit should be white on black.
    This function shifts the digit so its center of mass is near the center.
    """

    moments = cv2.moments(image_28)

    if moments["m00"] == 0:
        return image_28

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    shift_x = 14 - cx
    shift_y = 14 - cy

    transform_matrix = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    centered = cv2.warpAffine(
        image_28,
        transform_matrix,
        (28, 28),
        borderValue=0
    )

    return centered


# ----------------------------
# Convert drawing into proper MNIST format
# ----------------------------
def preprocess_digit(editor_value):
    """
    Final output:
    - white digit on black background
    - centered
    - square-safe
    - 28x28
    - flattened to 784 pixels
    """

    canvas = extract_canvas(editor_value)

    if canvas is None:
        return None, None

    # Convert RGB canvas to grayscale
    gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)

    # User draws black on white.
    # MNIST is white digit on black.
    inverted = 255 - gray

    # Smooth slightly so mouse strokes look less jagged
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)

    # Convert to clean binary ink mask
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Remove tiny noise
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Make strokes slightly fuller
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Find connected ink shapes
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None, None

    # Use the largest drawn component as the digit
    largest_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest_contour) < 20:
        return None, None

    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the digit from the original inverted grayscale image
    digit_crop = inverted[y:y + h, x:x + w]

    # Also crop the binary mask and remove background noise
    mask_crop = binary[y:y + h, x:x + w]
    digit_crop = cv2.bitwise_and(digit_crop, digit_crop, mask=mask_crop)

    # Resize while keeping aspect ratio
    crop_h, crop_w = digit_crop.shape

    scale = 20.0 / max(crop_w, crop_h)
    new_w = int(crop_w * scale)
    new_h = int(crop_h * scale)

    resized = cv2.resize(
        digit_crop,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Place resized digit into 28x28 black canvas
    mnist_image = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    mnist_image[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Shift by center of mass
    mnist_image = center_digit(mnist_image)

    # Normalize to match training data
    model_input = mnist_image.astype(np.float32) / 255.0
    model_input = model_input.reshape(1, 784)

    preview = Image.fromarray(mnist_image)

    return model_input, preview


# ----------------------------
# Prediction function
# ----------------------------
def predict_digit(editor_value):
    model_input, processed_image = preprocess_digit(editor_value)

    if model_input is None:
        return "Draw first", {}, None

    prediction = int(knn.predict(model_input)[0])

    probabilities = knn.predict_proba(model_input)[0]
    class_labels = knn.classes_

    confidence_dict = {
        str(label): float(prob)
        for label, prob in zip(class_labels, probabilities)
    }

    return str(prediction), confidence_dict, processed_image


# ----------------------------
# Clear function
# ----------------------------
def clear_canvas():
    return blank_canvas(), "", {}, None


# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(css=APP_CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ✍️ MNIST KNN Digit Recognizer
        Draw a digit from **0 to 9** and let KNN compare it with stored MNIST examples.
        """,
        elem_id="hero"
    )

    with gr.Group(elem_id="main-card"):
        with gr.Row():
            with gr.Column(scale=1):
                sketchpad = gr.ImageEditor(
                    value=blank_canvas,
                    type="numpy",
                    image_mode="RGB",
                    sources=(),
                    canvas_size=(280, 280),
                    fixed_canvas=True,
                    brush=gr.Brush(
                        default_size=18,
                        colors=["#000000"],
                        default_color="#000000",
                        color_mode="fixed"
                    ),
                    eraser=gr.Eraser(default_size=24),
                    layers=False,
                    transforms=[],
                    label="Draw here",
                    elem_id="draw-box"
                )

                with gr.Row():
                    predict_button = gr.Button("Predict digit", elem_id="predict-btn")
                    clear_button = gr.Button("Clear", elem_id="clear-btn")

            with gr.Column(scale=1):
                digit_output = gr.Textbox(
                    label="Predicted digit",
                    interactive=False,
                    elem_id="digit-output"
                )

                confidence_output = gr.Label(
                    label="Confidence scores"
                )

                processed_preview = gr.Image(
                    label="What the model actually sees: centered 28x28 MNIST image",
                    image_mode="L"
                )

    gr.Markdown(
        """
        Check the preview. It should look like a clean white digit centered on a black square.
        If the preview looks cursed, the prediction will also be cursed.
        """,
        elem_id="footer-note"
    )

    predict_button.click(
        fn=predict_digit,
        inputs=sketchpad,
        outputs=[digit_output, confidence_output, processed_preview]
    )

    clear_button.click(
        fn=clear_canvas,
        inputs=None,
        outputs=[sketchpad, digit_output, confidence_output, processed_preview]
    )


if __name__ == "__main__":
    demo.launch(share=True)
