<div align="center">

# Precise Aircraft Recognition

### Real-time aircraft detection & classification from video using deep learning

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://docs.ultralytics.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)

---

**Upload a video. Detect aircraft. Classify the model. Get results in seconds.**

</div>

<br>

## Overview

Precise Aircraft Recognition is an end-to-end deep learning pipeline that detects and classifies aircraft in video footage. It combines **YOLOv8** for real-time object detection with a fine-tuned **ResNet-18** classifier to identify specific aircraft types --- from commercial jets to stealth bombers.

A **Streamlit web app** provides an intuitive interface powered by **Google Gemini** for additional AI-driven analysis and annotation.

<br>

## Supported Aircraft

The model classifies **9 aircraft types** across commercial and military categories:

| Commercial | Military |
|:---:|:---:|
| Airbus | B-2 Bomber |
| ATR | B-52 Bomber |
| Boeing | F-16 |
| Boeing Defence | F-22 |
| | F-35 |

<br>

## Architecture

```
                    Input Video
                        |
                        v
               +------------------+
               |   YOLOv8 (nano)  |      Frame-by-frame object detection
               |   Detection      |      Locates aircraft bounding boxes
               +--------+---------+
                        |
                  Cropped Regions
                        |
                        v
               +------------------+
               | ResNet-18        |      Fine-tuned classifier
               | Classification   |      Identifies aircraft type
               +--------+---------+
                        |
                        v
               +------------------+
               |  Annotated Video |      Bounding boxes + labels
               |  Output          |      overlaid on every frame
               +------------------+
```

<br>

## Model Performance

The ResNet-18 classifier was trained for **20 epochs** on a curated dataset of aircraft images:

```
Epoch  1/20  --  Loss: 0.7550  |  Accuracy: 73.82%
Epoch  5/20  --  Loss: 0.0508  |  Accuracy: 98.63%
Epoch 10/20  --  Loss: 0.0209  |  Accuracy: 99.48%
Epoch 15/20  --  Loss: 0.0274  |  Accuracy: 99.20%
Epoch 20/20  --  Loss: 0.0392  |  Accuracy: 98.82%
```

> **Peak accuracy: 99.62%** (Epoch 19)

<br>

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Precise-aircraft-recognition.git
cd Precise-aircraft-recognition
```

### 2. Install dependencies

```bash
pip install torch torchvision ultralytics opencv-python streamlit google-genai python-dotenv
```

### 3. Set up your Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

### 4. Launch the Streamlit app

```bash
streamlit run app_streamlit.py
```

Then open **http://localhost:8501** in your browser, upload an MP4 video, and watch the magic happen.

<br>

## Project Structure

```
Precise-aircraft-recognition/
|
|-- app_streamlit.py              # Streamlit web app (Gemini-powered analysis)
|-- Flight_almost_final.ipynb     # Training notebook (YOLOv8 + ResNet-18 pipeline)
|-- README.md
```

| File | Description |
|---|---|
| `app_streamlit.py` | Web interface for uploading videos and viewing AI-annotated results. Uses Google Gemini for airline & model identification with text overlay on processed frames. |
| `Flight_almost_final.ipynb` | Full training pipeline --- dataset extraction, ResNet-18 fine-tuning, YOLOv8 detection integration, and video inference with bounding box annotations. |

<br>

## How It Works

### Training Pipeline (Notebook)

1. **Dataset** --- Aircraft images organized by class in an ImageFolder structure
2. **Preprocessing** --- Resize to 224x224, random horizontal flips, ImageNet normalization
3. **Model** --- ResNet-18 pretrained on ImageNet, final FC layer replaced for 9-class output
4. **Training** --- 20 epochs with Adam optimizer (lr=0.0001) and CrossEntropy loss
5. **Inference** --- YOLOv8 detects aircraft in video frames, crops are classified by ResNet-18

### Streamlit App

1. **Upload** --- User uploads an MP4 video through the web UI
2. **Analyze** --- Video is sent to Google Gemini for airline and aircraft model identification
3. **Annotate** --- Analysis results are overlaid on every frame using OpenCV
4. **Download** --- Processed video with annotations is available for playback and download

<br>

## Tech Stack

| Component | Technology |
|---|---|
| Object Detection | YOLOv8 (Ultralytics) |
| Classification | ResNet-18 (PyTorch) |
| Video Processing | OpenCV |
| Web Interface | Streamlit |
| AI Analysis | Google Gemini 2.5 Flash |
| Training Environment | Google Colab (CUDA) |

<br>

## License

This project is open source and available under the [MIT License](LICENSE).

<br>

---

<div align="center">

**Built with PyTorch, YOLOv8 & Google Gemini**

</div>
