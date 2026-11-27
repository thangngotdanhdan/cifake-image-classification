# CIFAKE image classification

A UTS group project comparing Random Forest, a HOG/LBP support vector machine, and ResNet-18 on real and synthetic CIFAKE images. The Gradio app accepts an uploaded image and runs the selected classifier. Use the model selector to compare predictions on the same image.

## Run locally

From the repository root, create a Python virtual environment and activate it. Install the inference dependencies and launch the app:

    python -m pip install -r requirements.txt
    python app.py

Download the trained files from the group's Drive models folder into the local models/ directory first. Run only trusted model files. Training data is not needed for inference. Each model loads on its first use.

## Models

| Model | Input representation | Model file |
| --- | --- | --- |
| Random Forest E9 | 32×32 image; DCT (100), HOG (324), LBP (10), GLCM (16), wavelet statistics (8), BGR histogram (512), in that order | rf_e9.joblib |
| SVM | Grayscale HOG and spatial LBP | hog_lbp_svm.joblib |
| ResNet-18 | 224×224 RGB, ImageNet normalization | rai_resnet18_cifake_final.pth |

The SVM shows a class label when probability estimates are unavailable. Scores describe the selected model's output on this input. The project was developed on CIFAKE; reliability on other generators and photographic domains has not been established here.

## Project files and contributions

- app.py adapts the group's Gradio notebook for local inference.
- preprocessing/ contains the model-specific feature extractors.
- notebooks/ contains the group's original training workflows with saved cell outputs removed. These notebooks use their original Colab or Kaggle dataset paths; update the paths for a different training environment.
- Ngoc Hieu Truong contributed the Gradio interface and model integration. The training notebooks and classifiers are part of the UTS team project.

### Trained weights (Drive access required)

- [Random Forest E9](https://drive.google.com/file/d/1CuF1uho1gNP5er4pV7X00P30mv6ZSATu/view)
- [HOG/LBP SVM](https://drive.google.com/file/d/1qQTeeati8X0ZZ07wZCHtDFDXA5ID6ikB/view)
- [ResNet-18](https://drive.google.com/file/d/12PXunw48Gso03sdqU08BSZWOyWfBcSpn/view)
