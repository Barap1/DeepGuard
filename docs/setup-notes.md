# Setup Notes

## Recommended Python Version

Use Python 3.10+ or 3.11 for local development.

## Fastest Way to Try the Project

The hosted Hugging Face demo is the easiest path for quick testing:

https://b-a-r-a-p-deepguard.hf.space/

Local setup requires PyTorch, OpenCV, dlib, Gradio, and access to a trained model
checkpoint.

## Local Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## dlib Installation Notes

dlib may require native build tools and CMake. If `pip install dlib` fails,
install the platform build prerequisites first, then rerun the requirements
installation.

## Running the Gradio App

Run the app with a trained model checkpoint:

```bash
python app.py --model_path path/to/model.pth
```

`app.py` remains the main entrypoint for local Gradio usage and Hugging Face
Space compatibility.

## Hardware Notes

GPU is optional for inference, but CUDA can improve speed when a compatible GPU
and PyTorch CUDA build are available. CPU inference should still work, although
larger videos and Deep Think mode may be slower.



