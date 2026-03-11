# Emergency Vehicle Audio Classification System (EVACS)

A machine learning system for detecting **emergency vehicle audio** from environmental recordings.

The system processes `.wav` audio clips, extracts  **log-mel spectrogram features** , and classifies them using a  **Convolutional Neural Network (CNN)** .

---

# Overview

EVACS identifies whether an audio clip contains:

* 🚑 Ambulance
* 🚒 Firetruck
* 🚗 Traffic

The system is designed to be:

* **Accurate**
* **Fast**
* **Deployable on CPU**

---

# Example Pipeline

<pre class="overflow-visible! px-0!" data-start="722" data-end="968"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Audio File (.wav)</span><br/><span>        │</span><br/><span>        ▼</span><br/><span>Audio Preprocessing</span><br/><span>(mono, normalize, resample, trim)</span><br/><span>        │</span><br/><span>        ▼</span><br/><span>Log-Mel Feature Extraction</span><br/><span>        │</span><br/><span>        ▼</span><br/><span>CNN Classifier</span><br/><span>        │</span><br/><span>        ▼</span><br/><span>Prediction</span><br/><span>(ambulance / firetruck / traffic)</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Dataset

The dataset contains **596 labeled audio clips** across three classes:

| Class     | Samples |
| --------- | ------- |
| Ambulance | 196     |
| Firetruck | 200     |
| Traffic   | 200     |

Directory structure:

<pre class="overflow-visible! px-0!" data-start="1173" data-end="1229"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>data/</span><br/><span>    ambulance/</span><br/><span>    firetruck/</span><br/><span>    traffic/</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Dataset Splits

To ensure reproducible evaluation, the dataset is divided into  **frozen splits** :

| Split      | Size |
| ---------- | ---- |
| Train      | 417  |
| Validation | 89   |
| Test       | 90   |

The  **test set is never used during training** .

---

# Model Architecture

A lightweight CNN is trained on  **log-mel spectrograms** .

Architecture:

<pre class="overflow-visible! px-0!" data-start="1569" data-end="1789"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Input: Log-mel spectrogram</span><br/><span>        │</span><br/><span>Conv2D + ReLU</span><br/><span>        │</span><br/><span>Max Pool</span><br/><span>        │</span><br/><span>Conv2D + ReLU</span><br/><span>        │</span><br/><span>Max Pool</span><br/><span>        │</span><br/><span>Conv2D + ReLU</span><br/><span>        │</span><br/><span>Global Avg Pool</span><br/><span>        │</span><br/><span>Fully Connected Layer</span><br/><span>        │</span><br/><span>Softmax</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

The model is exported using **TorchScript** for fast inference.

---

# Results

Evaluation on the  **frozen test split** :

**Test Accuracy:**

<pre class="overflow-visible! px-0!" data-start="1935" data-end="1948"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>91.1%</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Confusion Matrix:

| True \ Pred | Ambulance | Firetruck | Traffic |
| ----------- | --------- | --------- | ------- |
| Ambulance   | 22        | 7         | 1       |
| Firetruck   | 0         | 30        | 0       |
| Traffic     | 0         | 0         | 30      |

Most errors occur between  **ambulance and firetruck** , which have similar siren frequency patterns.

---

# Performance

Inference latency on CPU:

| Metric       | Latency |
| ------------ | ------- |
| Median (p50) | 12.1 ms |
| p95          | 14.5 ms |

This allows  **~80 predictions per second** .

---

# Project Structure

<pre class="overflow-visible! px-0!" data-start="2450" data-end="2958"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>evacs/</span><br/><span>│</span><br/><span>├── evacs/</span><br/><span>│   ├── dataset.py</span><br/><span>│   ├── preprocess.py</span><br/><span>│   ├── features.py</span><br/><span>│   ├── model.py</span><br/><span>│   ├── torch_cnn.py</span><br/><span>│   └── pipeline.py</span><br/><span>│</span><br/><span>├── scripts/</span><br/><span>│   ├── train.py</span><br/><span>│   ├── train_cnn.py</span><br/><span>│   ├── evaluate.py</span><br/><span>│   └── make_splits.py</span><br/><span>│</span><br/><span>├── configs/</span><br/><span>│   └── baseline.yaml</span><br/><span>│</span><br/><span>├── models/</span><br/><span>│   ├── cnn_logmel.pt</span><br/><span>│   └── cnn_model.json</span><br/><span>│</span><br/><span>├── splits/</span><br/><span>│   ├── train.csv</span><br/><span>│   ├── val.csv</span><br/><span>│   └── test.csv</span><br/><span>│</span><br/><span>├── docs/</span><br/><span>│   ├── SRS.docx</span><br/><span>│   ├── SAD.docx</span><br/><span>│   ├── LLD.docx</span><br/><span>│   └── Final_Report.docx</span><br/><span>│</span><br/><span>└── README.md</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Installation

Create a virtual environment:

<pre class="overflow-visible! px-0!" data-start="3012" data-end="3070"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python </span><span class="ͼu">-m</span><span> venv .venv</span><br/><span class="ͼs">source</span><span> .venv/bin/activate</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Install dependencies:

<pre class="overflow-visible! px-0!" data-start="3095" data-end="3142"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>pip install numpy torch python-docx</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Training

Train the CNN:

<pre class="overflow-visible! px-0!" data-start="3177" data-end="3327"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python scripts/train_cnn.py \</span><br/><span></span><span class="ͼu">--data</span><span> /path/to/data \</span><br/><span></span><span class="ͼu">--splits_dir</span><span> splits \</span><br/><span></span><span class="ͼu">--epochs</span><span></span><span class="ͼq">60</span><span> \</span><br/><span></span><span class="ͼu">--patience</span><span></span><span class="ͼq">7</span><span> \</span><br/><span></span><span class="ͼu">--batch</span><span></span><span class="ͼq">32</span><span> \</span><br/><span></span><span class="ͼu">--lr</span><span> 1e-3</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Evaluation

Evaluate the model on the frozen test split:

<pre class="overflow-visible! px-0!" data-start="3394" data-end="3507"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>python scripts/evaluate.py \</span><br/><span></span><span class="ͼu">--model</span><span> models/cnn_model.json \</span><br/><span></span><span class="ͼu">--splits_dir</span><span> splits \</span><br/><span></span><span class="ͼu">--split</span><span> test</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Example output:

<pre class="overflow-visible! px-0!" data-start="3526" data-end="3592"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>accuracy: 0.9111</span><br/><span>latency p50: 12.1 ms</span><br/><span>latency p95: 14.5 ms</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Inference Example

Classify a single audio file:

<pre class="overflow-visible! px-0!" data-start="3651" data-end="3695"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute inset-x-4 top-12 bottom-4"><div class="pointer-events-none sticky z-40 shrink-0 z-1!"><div class="sticky bg-token-border-light"></div></div></div><div class=""><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>evacs classify path/to/audio.wav</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

Output:

<pre class="overflow-visible! px-0!" data-start="3706" data-end="3752"><div class="relative w-full mt-4 mb-1"><div class=""><div class="relative"><div class="h-full min-h-0 min-w-0"><div class="h-full min-h-0 min-w-0"><div class="border border-token-border-light border-radius-3xl corner-superellipse/1.1 rounded-3xl"><div class="h-full w-full border-radius-3xl bg-token-bg-elevated-secondary corner-superellipse/1.1 overflow-clip rounded-3xl lxnfua_clipPathFallback"><div class="pointer-events-none absolute end-1.5 top-1 z-2 md:end-2 md:top-1"></div><div class="pe-11 pt-3"><div class="relative z-0 flex max-w-full"><div id="code-block-viewer" dir="ltr" class="q9tKkq_viewer cm-editor z-10 light:cm-light dark:cm-light flex h-full w-full flex-col items-stretch ͼk ͼy"><div class="cm-scroller"><div class="cm-content q9tKkq_readonly"><span>Prediction: firetruck</span><br/><span>Confidence: 0.93</span></div></div></div></div></div></div></div></div></div><div class=""><div class=""></div></div></div></div></div></pre>

---

# Future Work

Potential improvements:

* Real-time streaming audio detection
* Optimize for deployment constraints

---
