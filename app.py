"""Local Gradio interface for the three CIFAKE classifiers."""
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import gradio as gr
import joblib
import torch
from PIL import ImageOps
from torchvision.models import resnet18

from preprocessing.RandomForest_preprocessing import preprocess_single_image
from preprocessing.ResNet18_preprocessing import preprocess_image
from preprocessing.svm_feature_extraction import extract_features

MODELS = Path(__file__).resolve().parent / "models"
FILES = {
    "Random Forest": "rf_e9.joblib",
    "SVM": "hog_lbp_svm.joblib",
    "CNN": "rai_resnet18_cifake_final.pth",
}


@lru_cache(maxsize=3)
def load_model(name):
    path = MODELS / FILES[name]
    if not path.is_file():
        raise gr.Error(f"Place {path.name} in models/ first. See README.md.")
    if name != "CNN":
        return joblib.load(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, 2)
    net.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    return net.eval()


def analyse_image(image, model_name):
    if image is None:
        return "Please upload an image first.", None, ""
    image = ImageOps.exif_transpose(image).convert("RGB")
    info = f"{image.width} × {image.height} pixels, RGB"
    model = load_model(model_name)
    if model_name == "CNN":
        with torch.inference_mode():
            probs = model(preprocess_image(image)).softmax(dim=1)[0].tolist()
        scores = dict(zip(["FAKE", "REAL"], probs))
    else:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "input.png"
            image.save(path)
            if model_name == "Random Forest":
                features = preprocess_single_image(path)
                if features.shape != (1, 970) or model.n_features_in_ != 970:
                    raise gr.Error("The E9 model and its 970-feature extractor must match.")
                scores = dict(zip(model.classes_, map(float, model.predict_proba(features)[0])))
            else:
                features = extract_features(path).reshape(1, -1)
                pipeline = model["model"]
                labels = model["classes"]
                if not hasattr(pipeline, "predict_proba"):
                    label = labels[int(pipeline.predict(features)[0])]
                    return f"Prediction (SVM): {label}", None, info
                scores = {labels[int(i)]: float(p) for i, p in
                          zip(pipeline.classes_, pipeline.predict_proba(features)[0])}
    return f"Prediction ({model_name}): {max(scores, key=scores.get)}", scores, info


with gr.Blocks(title="CIFAKE Image Detector") as demo:
    gr.Markdown(
        """
        # Real or AI-Generated Image?
        Upload or paste an image, select a classifier, and click **Analyse Image**.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Upload or Paste Image",
                type="pil",
                image_mode="RGB",
                sources=["upload", "clipboard"],
            )

            model_selector = gr.Dropdown(
                choices=["Random Forest", "SVM", "CNN"],
                value="CNN",
                label="Select Model",
            )

            with gr.Row():
                analyse_btn = gr.Button(
                    "Analyse Image",
                    variant="primary",
                )
                clear_btn = gr.ClearButton()

        with gr.Column(scale=1):
            result_text = gr.Textbox(
                label="Analysis Result",
                interactive=False,

            )

            probability_output = gr.Label(
                label="Model scores",
                num_top_classes=2,
            )

            image_info = gr.Textbox(
                label="Image Information",
                lines=3,
                interactive=False,
            )



    # Connect buttons directly inside the same block
    analyse_btn.click(
        fn=analyse_image,
        inputs=[input_image, model_selector],
        outputs=[
            result_text,
            probability_output,
            image_info,
        ],
    )

    clear_btn.add([
        input_image,
        result_text,
        probability_output,
        image_info,
    ])

if __name__ == "__main__":
    demo.launch()
