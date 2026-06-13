# DeepGuard: Video Deepfake Detection System

DeepGuard is a deepfake detection prototype that analyzes uploaded videos and
predicts whether the video is likely real or fake. The app uses a hybrid deep
learning architecture: EfficientNet-B0 extracts spatial features from detected
face frames, and a GRU models temporal patterns across the video sequence.

Live demo: https://b-a-r-a-p-deepguard.hf.space/

## Overview

DeepGuard combines a Gradio web interface with a PyTorch video classification
model. A user uploads a video, chooses a processing mode, and receives a
REAL/FAKE confidence score. The project also includes preprocessing and training
scripts that document the broader research workflow used to explore spatial,
frequency, and gradient artifacts in deepfake videos.

This is a research and portfolio project, not a production-grade verification
system.

## Live Demo

Try the hosted Gradio app here:

https://b-a-r-a-p-deepguard.hf.space/

The hosted demo is the easiest way to test the app quickly because local setup
requires PyTorch, OpenCV, dlib, and a trained model checkpoint.

## Why I Built This

Deepfake generation tools are becoming more accessible and realistic, which
makes media authenticity harder to evaluate. This project explores how a video
detector can combine frame-level spatial features with temporal sequence
modeling to identify visual inconsistencies that may indicate manipulation.

The goal is to demonstrate an end-to-end AI/security research prototype: dataset
preprocessing, model architecture, evaluation, and a usable Gradio inference
interface.

## Features

- Gradio web app for video upload and prediction
- PyTorch model using EfficientNet-B0 and GRU
- Face detection with dlib and a faster OpenCV Haar cascade option
- Video frame sampling and face crop preprocessing
- Multiple inference modes: Super Fast, Normal, and Deep Think
- Optional visualization of detected face regions and processed face crops
- Example videos included in the app UI
- Hugging Face Space demo for quick testing

## How It Works

Inference pipeline:

```text
Video upload
-> frame sampling
-> face detection
-> face crop preprocessing
-> EfficientNet-B0 feature extraction
-> GRU temporal modeling
-> binary classifier
-> REAL/FAKE score
```

`app.py` reads the uploaded video with OpenCV, samples frames from one or more
chunks, detects faces, crops and normalizes the face regions, stacks them into a
sequence tensor, and sends the tensor through the model. The final score is the
average prediction across the processed chunks and optional test-time
augmentation passes.

## Model Architecture

The model combines:

- `EfficientNet-B0`: extracts per-frame spatial features from face crops
- `GRU`: models temporal patterns across the video frame sequence
- `MLP classifier`: converts the final temporal representation into one binary
  logit

The reusable model definition is in `model.py`. The Gradio app currently keeps a
matching model definition inside `app.py` for deployment compatibility.

## Research Methodology

The project paper describes a broader research and training pipeline:

```text
video dataset
-> dlib face tracking
-> facial bounding boxes
-> 32 extracted frames per video
-> RGB/spatial stream
-> DCT frequency stream
-> Sobel gradient stream
-> model training and evaluation
```

The research explored three feature families:

- spatial RGB face crops
- frequency features using Discrete Cosine Transform
- gradient features using the Sobel operator

Training used AdamW, a cosine annealing learning rate scheduler, automatic
mixed precision, gradient accumulation, and an NVIDIA RTX 3060 12GB GPU.

## Reported Results

The project paper reports evaluation on the challenging Celeb-DF-v2 dataset:

- 87% test accuracy
- 0.907 ROC AUC

These are reported evaluation results from the project paper and should not be
interpreted as production performance across all deepfake generation methods.
The paper also discusses a generalization gap between validation and test
performance, which is an important limitation for deepfake detection systems.

## Processing Modes

- `Super Fast`: uses fewer frames, a smaller image size, one video chunk, and
  the faster OpenCV face detector.
- `Normal`: uses the default frame count and image size with dlib face
  detection.
- `Deep Think`: uses more chunks and horizontal flip test-time augmentation for
  a slower but more thorough pass.

## Visualization Mode

The app can run a prediction with visualization. This mode shows:

- detected face regions
- processed face crops
- frame and chunk analysis
- per-chunk prediction scores
- final REAL/FAKE confidence score

Screenshots can be added here:

- `assets/demo-ui.png`
- `assets/prediction-output.png`
- `assets/visualization-output.png`

## Local Setup

Python 3.10+ or 3.11 is recommended.

```bash
git clone https://github.com/Barap1/Deepfake-Detection.git
cd Deepfake-Detection
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

dlib can be difficult to install because it may require CMake and native build
tools. If local installation fails, use the hosted Hugging Face demo for quick
testing.

Model weights may need to be provided separately depending on how you run the
project. Do not commit new model weights or uploaded videos to GitHub.

## Usage

Run the Gradio app with a trained model checkpoint:

```bash
python app.py --model_path path/to/model.pth
```

Then open the local Gradio URL printed in the terminal and upload a video.

## Testing

Run the lightweight model-shape test suite:

```bash
python -m unittest discover tests
```

The tests do not require a trained model checkpoint, video file, dlib detector,
or Gradio launch.

## Project Structure

```text
.
|-- app.py                         # Main Gradio app entrypoint
|-- model.py                       # Reusable EfficientNet-B0 + GRU model
|-- dataset.py                     # PyTorch dataset and transforms
|-- train.py                       # Training/evaluation script
|-- sampleDFDC.py                  # DFDC sampling utility
|-- SpatialFeaturePreprocessing.py # dlib face tracking and bbox extraction
|-- AdvancedFeatureExtraction.py   # RGB/DCT/Sobel feature extraction
|-- Example1.mp4                   # Example app video
|-- Example2.mp4                   # Example app video
|-- Example3.mp4                   # Example app video
|-- Example4.mp4                   # Example app video
|-- best_model.pth                 # Tracked model checkpoint
|-- GHP 62 Deepfake detection Paper.pdf
|-- tests/test_model_shape.py       # Lightweight model forward-pass test
|-- .github/workflows/tests.yml     # CI test workflow
|-- requirements.txt
|-- requirements-dev.txt
|-- README.md
```

## Limitations

- This is a prototype/research project.
- It is not production-grade.
- It is not suitable for high-stakes decisions.
- Predictions depend on video quality, compression, lighting, and face
  visibility.
- The model can fail if no face is detected.
- No complete portable training pipeline is currently included in the repo.
- Reported results are dataset-dependent.
- Deepfake detection models can become outdated as generation methods improve.

## Ethical Use

This project is for educational and research purposes. It should not be used to
harass, accuse, or make high-stakes judgments about people. Deepfake detection
outputs should be treated as probabilistic signals, not proof.

## Future Improvements

- document the full training pipeline
- add dataset and evaluation scripts
- add Grad-CAM or saliency visualizations
- add batch inference
- split preprocessing, inference, and visualization into modules
- add a Dockerfile
- add screenshots or a demo GIF
- add tests for preprocessing utilities
- consolidate the duplicate model definitions in `app.py` and `model.py` after
  deployment compatibility is verified

## Related Writeup / Paper

- [Improving Deep Fake Detection: Integrating Spatial, Frequency, and Gradient Analyses](https://github.com/user-attachments/files/21153386/GHP.62.Deepfake.detection.Paper.pdf)
- Local tracked copy: `GHP 62 Deepfake detection Paper.pdf`
