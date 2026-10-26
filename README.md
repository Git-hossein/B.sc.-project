# Automated Video-to-Audio RAG Generation Pipeline

> **Bachelor's Thesis Project**  
> **University of Tübingen**  

## Overview
This repository contains the complete implementation of an automated **Video-to-Audio Retrieval-Augmented Generation (RAG)** pipeline. Designed specifically for **10-second, 
audio-less video clips**, the system automatically generates suitable audio by leveraging a 
slightly modfied (which is provided) database of the widely used **VGGSound** dataset. This work is meant for generating environmental sounds and noises, and so videos where the focus is on human speach are not suitable candiates. 

The pipeline covers everything from raw dataset preparation and multi-modal feature extraction to fast batched retrieval, temporal re-ranking, and audio generation using state-of-the-art models like **AudioGen** and **AudioLDM2**.

## Key Pipeline Stages

### 1. Data Preparation & Ingestion
* **Downloading & Trimming**: Automatically downloads videos in batches from YouTube using `yt-dlp`, trimming them down to uniform 10-second segments. I have used the same csv file the official **VGGSound** repository provides, but have taken out
  some of the duplicated items in the csv. My version of the dataset is provided under "vggsound/vggsound.csv"
* **Integritiy verification**: Detect bad frames, or short audio files etc. to automatically remove the videos which are not successfully downloaded and or trimmed
* **Extraction**: Extracts synchronized high-quality audio tracks (`.wav`) and video frames ('.jpg')for downstream processing.

### 2. Multi-Modal Embedding Extraction
* **Video Features**: Extracts visual representations using OpenAI's **CLIP (ViT)** vision transformer.
* **Audio Features**: Extracts audio representations using **Wav2Vec**.
* **Advanced Representations**: Integrates **CAV MAE** and **CAV MAE SYNC** for more powerful joint audio-video and spatio-temporal representations.

### 3. Retrieval & Temporal Re-Ranking (The RAG Core)
* **Coarse Retrieval**: Batched, vectorized cosine similarity search over normalized embedding spaces coupled with temperature-scaled softmax scoring.
* **Temporal Re-Ranking**: Uses **CavMae Sync** to perform frame-level temporal alignment, combining coarse semantic similarity and fine-grained temporal scores.

### 4. Audio Generation (RAG-Augmented & Pure Text)
Supports parallel execution across multiple GPUs (via Slurm job arrays on the TCML server) for two primary generation backends:
* **AudioGen (Continuation)**: Uses the top-retrieved audio match as a temporal reference base for audio continuation.
* **AudioLDM2 (Style Transfer)**: Uses the top-retrieved audio match as a style-transfer base for audio-to-audio generation.
* **Pure Text-to-Audio**: Fallback or baseline generation using default text prompts derived from VGGSound metadata.

## Repository Structure
```text
├── vggsound/             # Dataset handling: downloading, trimming 10s clips, and file integrity verification
├── experiment/           # Automated experiment runner for batch retrieval, generation, and result logging
├── audio_inference/      # Retrieval, evaluation, inference, and temporal re-ranking modules
├── audio_generation/     # Generation pipelines (features audio-to-audio & text-to-audio runners for TCML, plus reference folders for AudioLDM2 and AudioGen)
├── video/                # Video processing module for frame extraction and CLIP embedding extraction
└── audio/                # Audio processing module for raw audio extraction and embedding extraction
└── README.md
```

Additionally there are two more repositories which contain relevant code for this project. One is a dedicated repository for the **CAVMAE** and **CAVMAE SYNC** models which feature audio and video embedding extractions using these models: 
https://github.com/Git-hossein/CAV_MAE

And another dedicated repository for the purpose of audiogeneration using the two modesl **AudioGEN** and **AudioLDM2**: https://github.com/Git-hossein/B.sc.-Audiocraft-module
