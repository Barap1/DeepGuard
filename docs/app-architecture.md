# App Architecture

## Entrypoint

`app.py` remains the main Gradio entrypoint for local usage and Hugging Face
Space compatibility.

Run it with:

```bash
python app.py --model_path path/to/model.pth
```

The app requires a model checkpoint path at startup. It loads the checkpoint,
builds the Gradio interface, and launches the app from the same file.

## Config Class

`Config` stores the core inference defaults:

- `SEQ_LENGTH = 20`
- `IMG_SIZE = 224`
- `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

The processing modes override some of these values at prediction time.

## Model Loading

Inside `main(args)`, the app:

1. creates `DeepfakeDetector(pretrained=False)`
2. loads weights from `args.model_path`
3. moves the model to `Config.DEVICE`
4. sets the model to evaluation mode

If the model file is missing or cannot be loaded, the app prints an error and
returns before launching Gradio.

## Face Detectors

The app initializes two face detectors:

- dlib frontal face detector: used by Normal and Deep Think modes
- OpenCV Haar cascade detector: used by Super Fast mode

The dlib detector is slower but generally more robust for the default path. The
OpenCV detector is used when the app needs a faster pass with fewer frames.

## Preprocessing Flow

### `preprocess_video()`

`preprocess_video()` reads the uploaded video with OpenCV, seeks to one or more
chunk start positions, detects a face in sampled frames, crops the detected face,
pads the sequence when needed, applies transforms, and returns tensor chunks for
model inference.

Important parameters:

- `seq_length`: number of face frames to collect per chunk
- `img_size`: resize target for face crops
- `num_chunks`: number of video chunks to sample
- `use_fast_detector`: switches between OpenCV and dlib detection
- `tta_flip`: adds a horizontally flipped test-time augmentation pass

### `get_inference_transforms()`

Builds the standard Albumentations inference transform:

- resize to the configured image size
- ImageNet normalization
- tensor conversion

### `get_inference_transforms_tta()`

Builds the Deep Think test-time augmentation transform:

- resize to the configured image size
- horizontal flip
- ImageNet normalization
- tensor conversion

## Visualization Flow

### `visualize_detection_process()`

This function runs a readable analysis pass over the video and returns:

- a face detection plot
- a processed face crop plot
- detailed analysis text

The visualization includes video metadata, face detection counts by chunk,
detected face bounding boxes, processed face crops, feature extraction context,
and test-time augmentation status.

## Prediction Flow

### `predict()`

`predict()` is the fast prediction callback for the Predict Only button. It:

1. maps the selected processing mode to preprocessing parameters
2. calls `preprocess_video()`
3. runs each tensor chunk through the loaded model
4. averages chunk predictions
5. returns a Gradio label dictionary with FAKE and REAL probabilities

### `predict_with_visualization()`

`predict_with_visualization()` follows the same prediction path but first calls
`visualize_detection_process()`. It returns the prediction label, face detection
plot, processed face plot, and detailed analysis text.

## Processing Modes

- `Super Fast`: sequence length 8, image size 160, one chunk, OpenCV face
  detector, no test-time augmentation
- `Normal`: sequence length 20, image size 224, one chunk, dlib detector, no
  test-time augmentation
- `Deep Think`: sequence length 20, image size 224, three chunks, dlib detector,
  horizontal flip test-time augmentation

## Gradio UI Components

The Gradio app uses:

- video input
- processing mode radio
- Predict Only button
- Predict with Visualization button
- label output
- face detection plot
- processed faces plot
- detailed analysis textbox
- example videos

The visualization row is hidden by default and shown after the visualization
callback runs.

## Duplicate Model Definitions

The repo currently has a model architecture in both `app.py` and `model.py`.
Future cleanup should consolidate this into one shared definition, but that
should not be changed casually. `app.py` likely supports the current
Gradio/Hugging Face deployment, so the duplicate definition is intentionally
left in place until deployment compatibility can be fully verified.

