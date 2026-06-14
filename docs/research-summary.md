# Research Summary

## Title

Improving Deep Fake Detection: Integrating Spatial, Frequency, and Gradient
Analyses

## Motivation

Deepfake generation has become easier to access and increasingly realistic. The
paper frames this as a risk to information integrity and public trust because
synthetic videos can be used for disinformation and impersonation.

The central technical motivation is generalization. Many deepfake detectors
perform well on the datasets they were trained on but degrade when evaluated on
unseen or in-the-wild videos. The paper argues that effective detection should
focus on features that are more fundamental to the synthesis process rather than
artifacts tied only to one training dataset or generation method.

## Problem

Deepfake detectors often fail to generalize to new forgery techniques or unseen
video distributions. A model can learn dataset-specific artifacts instead of
robust manipulation signals, which creates a gap between validation performance
and real test performance.

## Dataset and Preprocessing

The paper reports a training dataset of 20,000 real and fake videos.

Preprocessing used a two-stage approach:

1. dlib face tracking detected faces and saved facial bounding boxes for each
   video.
2. 32 frames per video were extracted and processed into feature streams.

The repository reflects this workflow through:

- `SpatialFeaturePreprocessing.py`, which detects/tracks faces and writes
  bounding boxes to JSON files
- `AdvancedFeatureExtraction.py`, which uses those boxes to extract per-video
  feature arrays

## Feature Streams Explored

The research explored three feature streams:

- Spatial/RGB: resized face crops from selected video frames
- Frequency/DCT: frequency-domain features generated with Discrete Cosine
  Transform
- Gradient/Sobel: edge and gradient information generated with the Sobel
  operator

These streams were saved as compressed `.npz` files for training and
evaluation.

## Model Architecture

The reported architecture is a hybrid spatio-temporal model:

- EfficientNet-B0 extracts discriminative spatial features from individual video
  frames.
- A Gated Recurrent Unit (GRU) models temporal coherence across the frame
  sequence.
- A small MLP classifier produces the final binary output.

This architecture is implemented in `model.py` and mirrored in `app.py` for the
Gradio inference workflow.

## Training Procedure

The paper reports the following training setup:

- 10 epochs
- NVIDIA RTX 3060 12GB GPU
- AdamW optimizer
- cosine annealing learning rate scheduler
- automatic mixed precision
- gradient accumulation to work within hardware constraints

The repository includes `train.py`, which demonstrates a PyTorch training loop
with AdamW, mixed precision, gradient accumulation, validation metrics, and
checkpoint saving. It currently contains local dataset paths and should be
treated as research code rather than a fully portable training pipeline.

## Evaluation

The paper reports evaluation on Celeb-DF-v2, a challenging deepfake forensics
dataset.

Reported results:

- 87% test accuracy
- 0.907 ROC AUC

These results should be understood as reported project evaluation results on
that dataset. They should not be treated as guaranteed real-world or
production-grade performance across all video sources and deepfake generation
methods.

## Discussion

The reported accuracy and ROC AUC indicate that the EfficientNet-B0 plus GRU
approach learned useful spatial and temporal signals. At the same time, the
paper explicitly notes a generalization gap between validation and test
performance.

That gap matters because deepfake generation methods evolve quickly. Future work
should focus on more fundamental manipulation artifacts, broader evaluation
coverage, and stronger generalization to unseen datasets and wild videos.

## Relationship to the Current App

The current Gradio app demonstrates the deployed inference side of the project.
It lets a user upload a video, runs face detection and face-crop preprocessing,
passes sampled frame sequences through the EfficientNet-B0 plus GRU model, and
returns a REAL/FAKE score.

This research summary documents the broader training and evaluation approach
behind the project: dataset construction, dlib face tracking, multi-stream
feature extraction, training setup, and reported Celeb-DF-v2 results.

