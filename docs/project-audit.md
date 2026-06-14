# Project Audit and Improvement Plan

## Current Repository Snapshot

The repository currently contains a Gradio deepfake detection app, model code,
dataset and preprocessing scripts, a sparse README, a tracked paper PDF, example
videos, and a tracked model checkpoint.

Tracked files at the start of this audit:

- `.gitignore`
- `AdvancedFeatureExtraction.py`
- `SpatialFeaturePreprocessing.py`
- `app.py`
- `best_model.pth`
- `dataset.py`
- `Example1.mp4`
- `Example2.mp4`
- `Example3.mp4`
- `Example4.mp4`
- `GHP 62 Deepfake detection Paper.pdf`
- `model.py`
- `README.md`
- `requirements.txt`
- `sampleDFDC.py`
- `train.py`
- `__pycache__/model.cpython-312.pyc`

The tracked model checkpoint, example videos, paper PDF, and Python cache file
were already present before this polish pass. This audit does not add duplicate
large binaries.

## What `app.py` Does

`app.py` is the main runnable Gradio entrypoint. It accepts a required model
path:

```bash
python app.py --model_path path/to/model.pth
```

At launch, it:

- selects CUDA if available, otherwise CPU
- initializes a dlib frontal face detector
- initializes an OpenCV Haar cascade detector for faster face detection
- defines and loads a local `DeepfakeDetector` model class
- loads model weights from `--model_path`
- builds a Gradio UI with video upload, processing mode selection, prediction
  output, example videos, and optional visualization plots

The app exposes three processing modes:

- `Super Fast`: shorter sequence length, smaller image size, one chunk, and the
  OpenCV Haar detector
- `Normal`: default sequence length and image size, one chunk, and dlib face
  detection
- `Deep Think`: multiple chunks with dlib detection and horizontal flip
  test-time augmentation

The inference path is:

1. read uploaded video with OpenCV
2. sample frames from one or more chunks
3. detect a face in each sampled frame
4. crop the detected face
5. resize and normalize crops with Albumentations
6. stack face crops into a tensor sequence
7. run EfficientNet-B0 feature extraction
8. run GRU temporal modeling
9. classify the sequence as real or fake

The visualization path shows detected face boxes, processed face crops, per-chunk
predictions, and a final confidence summary.

## What `model.py` Does

`model.py` defines a reusable `DeepfakeDetector` class. It combines:

- EfficientNet-B0 as the per-frame CNN feature extractor
- a GRU for temporal sequence modeling
- a small MLP classifier with two linear layers, ReLU, dropout, and one output
  logit

The model expects an input tensor shaped like:

```text
[batch_size, sequence_length, channels, height, width]
```

It returns one logit per video sequence with shape:

```text
[batch_size, 1]
```

## Supporting Scripts

- `sampleDFDC.py` samples real and fake videos from a local DFDC-style dataset
  into a smaller training set.
- `SpatialFeaturePreprocessing.py` uses dlib to detect and track faces, then
  writes frame-level bounding boxes to JSON files.
- `AdvancedFeatureExtraction.py` reads bounding-box JSON files and source
  videos, then extracts RGB, DCT frequency, and Sobel gradient streams into
  compressed `.npz` files.
- `dataset.py` defines a PyTorch dataset for `.npz` feature files plus training
  and validation transforms.
- `train.py` trains the model with PyTorch, AdamW, gradient accumulation, AMP,
  and validation/test metrics. It contains local absolute dataset paths and is
  not currently a portable end-to-end training pipeline.

## Dependencies Detected from Imports

Runtime and app dependencies:

- `torch`
- `torchvision`
- `opencv-python` or an OpenCV package with `cv2`
- `dlib`
- `albumentations`
- `gradio`
- `tqdm`
- `pillow`
- `matplotlib`
- `numpy`

Training and feature extraction dependencies:

- `scipy`
- `scikit-learn`

The app currently depends on dlib, which can be difficult to install on some
systems because it may require CMake and native build tools.

## Current Gradio Entrypoint Behavior

`app.py` remains the compatibility-critical Gradio entrypoint. The command-line
contract is:

```bash
python app.py --model_path path/to/model.pth
```

The Hugging Face Space deployment likely depends on this file staying runnable
as-is. Refactoring the UI, model loading, or callback signatures would carry
risk unless tested directly in the deployment environment.

## What the Research Paper Contributes

The tracked paper, "Improving Deep Fake Detection: Integrating Spatial,
Frequency, and Gradient Analyses," provides important project context that is
not represented in the original README:

- the work targets the generalization problem in deepfake detection
- the architecture uses EfficientNet-B0 plus GRU sequence modeling
- evaluation was performed on the challenging Celeb-DF-v2 dataset
- reported test accuracy is 87%
- reported ROC AUC is 0.907
- training used 20,000 real and fake videos
- dlib face tracking saved facial bounding boxes
- 32 frames per video were extracted
- RGB/spatial, DCT frequency, and Sobel gradient feature streams were explored
- training used AdamW, a cosine annealing scheduler, AMP, gradient accumulation,
  and an NVIDIA RTX 3060 12GB
- the discussion notes a validation-to-test generalization gap

These results should be described as reported project evaluation results, not
as universal production performance.

## Safe Improvements

The safest high-impact improvements are documentation and lightweight checks:

- rewrite the README as an honest portfolio-quality project page
- preserve the Hugging Face demo link
- preserve the existing paper link
- add a research summary based on the tracked paper
- add setup notes for local usage and dlib installation friction
- add a `.gitignore` that prevents new model weights, videos, logs, outputs, and
  virtual environments from being committed
- add lightweight model-shape tests that import only `model.py`
- add CI that does not install dlib, launch Gradio, require videos, or require
  model weights
- document the Gradio app architecture without refactoring `app.py`

## Deferred Work

The following changes should be deferred until they can be verified more deeply:

- consolidating the duplicate model definitions in `app.py` and `model.py`
- moving preprocessing, inference, and visualization out of `app.py`
- changing Gradio callback signatures or app launch behavior
- making `train.py` a fully portable training pipeline
- adding complete dataset download, preprocessing, and evaluation scripts
- removing or replacing tracked large files unless the repository owner confirms
  that history and hosting constraints should be addressed
- adding production claims or broad real-world accuracy statements

