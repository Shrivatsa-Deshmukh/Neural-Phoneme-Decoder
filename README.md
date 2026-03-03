# Neural Phoneme Decoder for Speech BCI

A PyTorch implementation of the GRU-based neural phoneme decoder from [Willett et al. (2023)](https://www.nature.com/articles/s41586-023-06377-x), adapted to run on a single consumer GPU. This repo implements the first stage of the full speech BCI pipeline — decoding intracortical neural signals into phoneme probabilities.

**Achieved 78.77% phoneme accuracy on a single consumer GPU**, validating the neural signal quality at the decoder stage without requiring the full Kaldi language model infrastructure.

---

## Table of Contents
- [Background](#background)
- [Full Pipeline vs This Repo](#full-pipeline-vs-this-repo)
- [Brain Areas and Neural Signals](#brain-areas-and-neural-signals)
- [Hardware](#hardware)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [This Implementation](#this-implementation)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [References](#references)

---

## Background

Speech BCIs restore communication to people with paralysis by recording neural activity from motor cortex and decoding intended speech into text — without requiring the user to produce any vocal sound.

Willett et al. (2023) demonstrated a landmark result in *Nature*:
- **9.1% word error rate** on a 50-word vocabulary
- **23.8% word error rate** on a 125,000-word vocabulary — the first successful large-vocabulary speech BCI
- **62 words per minute** — 3.4x faster than the previous state-of-the-art BCI
- Works on **silent speech attempts** — no sound required

---

## Full Pipeline vs This Repo

The complete Willett et al. system is a three-stage pipeline:

```
Stage 1 — Neural Decoder (this repo)
  Intracortical neural signals
       |
  GRU-based RNN
       |
  Phoneme probabilities at each 80ms time step

Stage 2 — Viterbi Search  (not implemented)
  Selects the single most likely phoneme path
  through the per-timestep probability distributions

Stage 3 — Language Model  (not implemented)
  Kaldi-based trigram model over 125,000-word vocabulary
  Beam search selects the most probable word sequence
  given the phoneme path and language statistics
       |
  Final decoded text
```

**This repo covers Stage 1 only.** The 78.77% accuracy reported here is phoneme-level accuracy from the GRU decoder — not word error rate, which requires the full pipeline including Stages 2 and 3.

The original codebase ([cffan/neural_seq_decoder](https://github.com/cffan/neural_seq_decoder)) also covers Stage 1 only. The Kaldi language model is a separate system and is not included. The reason the original required a high-end multi-GPU cluster is primarily the full pipeline — the Kaldi beam search over a 125k-word vocabulary is extremely compute and memory intensive. The GRU decoder stage in isolation is manageable on a single GPU with the right hyperparameter adjustments.

---

## Brain Areas and Neural Signals

### Recording Sites

**Area 6v — Ventral Premotor Cortex** (primary signal source)
- Encodes orofacial movements: jaw, lips, tongue
- Contains fine-grained phoneme and articulator representations
- Decodes 39 phonemes with 62% accuracy using a simple naive Bayes classifier alone
- Tuning to speech articulators is spatially intermixed at the single-electrode level — accurate decoding is possible from a 3.2 x 3.2 mm array
- Retains strong speech tuning even years after paralysis

**Area 44 — Broca's Area**
- Traditionally believed central to speech production
- Showed less than 12% classification accuracy in this study — not useful for decoding
- All decoder training uses Area 6v recordings only

### What Neurons Encode

Individual neurons in Area 6v encode specific phonemes, articulator movements (jaw, lips, tongue, larynx), and whole-word patterns simultaneously. Representations are intermixed rather than neatly organized by articulator type, and mirror the articulatory structure of speech: consonants articulated similarly have similar neural representations, and vowels preserve the two-dimensional front/back and high/low articulatory structure.

---

## Hardware

Four 96-channel Utah arrays (**256 electrodes total**):
- 2 arrays in ventral premotor cortex (Area 6v)
- 2 arrays in Broca's area (Area 44)

**Signals recorded per electrode:**
- **Threshold crossings (TX):** Spike events detected when the signal crosses -3.5x, -4.5x, -5.5x, or -6.5x the channel's RMS noise. Four sensitivity levels, each producing a T x 256 matrix per 20 ms bin.
- **Spike band power (SP):** RMS amplitude in the 300-5,000 Hz band per 20 ms frame (T x 256), reflecting local firing intensity.

Signals are digitized and streamed to an external decoder in real time.

---

## Dataset

**Source:** [Willett et al. (2023) — Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq)

The dataset contains neural recordings from 10,850 attempted speech trials across multiple sessions. Each session has approximately 280-380 sentence trials.

| Field | Description |
|---|---|
| `sentenceText` | UTF-8 sentence strings, one per trial |
| `tx1` to `tx4` | Threshold crossings at four sensitivity levels; each T x 256 per 20 ms bin |
| `spikePow` | 300-5,000 Hz RMS band power per frame (T x 256) |
| `blockIdx` | Recording block label (~10 sentences per block), used for normalization |

### Preprocessing

Neural data is binned every 20 ms. TX and SP features from all 256 channels are z-scored per channel:

- Subtracting the channel mean removes fixed electrode offsets
- Dividing by standard deviation places all 256 channels on the same scale — one unit equals one typical fluctuation on that electrode
- Makes features directly comparable across channels and allows the network to use shared weights rather than learning per-channel scaling

Rolling normalization across blocks handles neural drift within a session.

---

## Model Architecture

```
Input: T x 256 neural features
       (128 electrodes x 2 signal types: threshold crossings + spike band power)
            |
   Gaussian Smoothing (sigma=2.0)
       stabilizes noisy spike signals
            |
   Day-specific Linear Layer
       per-session weight matrix + bias, initialized to identity
       handles electrode drift across recording days
            |
   Softsign Nonlinearity
            |
   Unfold (kernelLen=32, stride=4)
       640ms sliding window, 80ms step
       preserves temporal context while reducing sequence length 4x
            |
   Bidirectional GRU — 5 layers, hidden_dim=512
       sees both past and future context at each step
            |
   Linear -> 41 output classes
       40 phonemes + 1 CTC blank token
            |
   Phoneme probabilities per time step
   (passed to Viterbi + language model in the full pipeline)
```

### Key Design Decisions

**Day-specific input layers:** Electrode signals shift slightly across recording days as arrays settle in tissue. Each day gets its own learned linear transformation (`dayWeights`, `dayBias`), initialized to the identity matrix so training starts as a pass-through and adapts per session.

**Temporal unfolding:** A 32-bin sliding window is stacked into the feature dimension rather than discarding time steps — preserving temporal context while reducing the GRU sequence length by 4x. This gives the GRU 640 ms of context at each 80 ms output step.

**CTC Loss:** Connectionist Temporal Classification enables training without frame-level phoneme alignment labels. The model outputs phoneme probabilities at each step; CTC marginalizes over all valid alignments during backpropagation.

**Bidirectional GRU:** Both past and future neural context inform each time step, improving phoneme boundary detection.

**On-GPU augmentation:**
- White noise (std=0.8): per-sample Gaussian noise on the neural input
- Constant offset noise (std=0.2): per-trial DC shift simulating electrode baseline drift

---

## This Implementation

The original code was designed for a multi-GPU HPC cluster running the full pipeline including Kaldi. This repo isolates and runs the GRU decoder stage independently on a single consumer GPU, with two hyperparameter adjustments to fit within consumer GPU memory:

All hyperparameters live in `train_model.py` — the same file that launches training — with inline comments explaining each change.

| Hyperparameter | Original | This Repo | Rationale |
|---|---|---|---|
| `nUnits` (GRU hidden size) | 1024 | **512** | 2x smaller GRU, ~75% fewer parameters |
| `batchSize` | 64 | **16** | 4x lower GPU memory per step |
| `dropout` | 0.4 | **0.3** | Adjusted for smaller model capacity |
| `nLayers` | 5 | 5 | Unchanged |
| `bidirectional` | True | True | Unchanged |
| `kernelLen` | 32 | 32 | Unchanged |
| `lrStart` / `lrEnd` | 0.02 | 0.02 | Unchanged |

**Result:** The decoder runs end-to-end on a single consumer GPU and achieves 78.77% phoneme accuracy, confirming that the neural signal contains strong phoneme-level information even without the downstream Viterbi and language model stages.

---

## Repository Structure

```
neural_seq_decoder/
├── train_model.py                     # Entry point: all hyperparameters + training launch
├── src/
│   └── neural_decoder/
│       ├── model.py                   # GRUDecoder architecture
│       ├── neural_decoder_trainer.py  # Training loop, evaluation, model saving/loading
│       ├── dataset.py                 # SpeechDataset: loads pickle, returns tensors
│       └── augmentations.py           # WhiteNoise, MeanDriftNoise, GaussianSmoothing
├── notebooks/
│   └── formatCompetitionData.ipynb    # Step 1: converts raw .mat files to pickle
├── setup.cfg
└── README.md
```

---

## Setup and Installation

**Requirements:** Python >= 3.9, PyTorch >= 2.0, CUDA-capable GPU

```bash
git clone https://github.com/YOUR_USERNAME/neural_seq_decoder.git
cd neural_seq_decoder
pip install -e .
```

Download the dataset from [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq) and place the raw `.mat` files locally (e.g. `./data/raw/`).

---

## How to Run

### Step 1 — Format the raw data

```bash
jupyter notebook notebooks/formatCompetitionData.ipynb
```

Run all cells. Set the input path (raw `.mat` files) and output path at the top of the notebook. This z-scores the neural features, converts sentence text to phoneme sequences, and saves a `ptDecoder_ctc` pickle with `train` and `test` splits.

Each split is a list of per-day dicts with keys `sentenceDat` (neural features), `phonemes` (label sequences), and `phoneLens` (sequence lengths).

---

### Step 2 — Train the model

Open `train_model.py` and set your paths:

```python
args['outputDir']   = './outputs'            # where weights + stats are saved
args['datasetPath'] = './data/ptDecoder_ctc' # path to formatted pickle
```

Then run:

```bash
python train_model.py
```

Or pass paths at the command line:

```bash
python train_model.py --output_dir ./outputs --dataset_path ./data/ptDecoder_ctc
```

**Training output** (printed every 100 batches):
```
batch 0,    ctc loss: 3.556234, cer: 0.982341, time/batch: 0.412
batch 100,  ctc loss: 2.103847, cer: 0.743210, time/batch: 0.389
...
batch 9900, ctc loss: 0.847123, cer: 0.212300, time/batch: 0.381
```

`cer` = Character Error Rate on phoneme sequences. Phoneme accuracy = `1 - cer`. The best checkpoint (lowest test CER) is saved automatically to `outputDir/modelWeights`. Training stats are saved to `outputDir/trainingStats`.

---

### Loading a saved model

```python
from neural_decoder.neural_decoder_trainer import loadModel

model = loadModel('./outputs', nInputLayers=24, device='cuda')
model.eval()
```

---

## Results

This repo evaluates Stage 1 of the pipeline only — phoneme decoding accuracy from the GRU output, before Viterbi search or language model post-processing.

| | This Repo | Willett et al. (2023) |
|---|---|---|
| **Metric** | Phoneme accuracy | Word error rate (full pipeline) |
| **Score** | **78.77%** | 9.1% WER (50-word) / 23.8% WER (125k-word) |
| **Pipeline stage** | GRU decoder only | GRU + Viterbi + Kaldi LM |
| **Hardware** | Single consumer GPU | Multi-GPU HPC cluster |

The paper reports a raw phoneme error rate of ~19.7% from the GRU decoder alone (before language model), corresponding to ~80.3% phoneme accuracy. This implementation reaches 78.77% — within 1.5 percentage points — running on consumer hardware by isolating the decoder stage from the full Kaldi infrastructure.

---

## References

- Willett, F.R., Kunz, E.M., Fan, C., et al. (2023). *A high-performance speech neuroprosthesis*. Nature, 620, 1031–1036. https://doi.org/10.1038/s41586-023-06377-x
- PyTorch decoder: [cffan/neural_seq_decoder](https://github.com/cffan/neural_seq_decoder)
- Original MATLAB implementation: [fwillett/speechBCI](https://github.com/fwillett/speechBCI)
- Dataset: [Dryad doi:10.5061/dryad.x69p8czpq](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq)
