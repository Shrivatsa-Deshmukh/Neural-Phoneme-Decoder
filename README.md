# speech-bci-decoder

A PyTorch reimplementation of the intracortical GRU-based phoneme decoder from [Willett et al. (2023)](https://www.nature.com/articles/s41586-023-06377-x), restructured to run on a single consumer GPU. Decodes multichannel neural population activity into phoneme sequences using a bidirectional GRU trained with Connectionist Temporal Classification (CTC).

**Achieved 78.77% phoneme decoding accuracy — within 1.5 percentage points of the published Nature 2023 benchmark — without access to the original HPC infrastructure or Kaldi decoding stack.**

---

## Pipeline Scope

The full Willett et al. system decodes speech through three sequential stages:

```
Stage 1 — GRU Phoneme Decoder     <- this repo
          Maps T x 256 neural features to per-timestep
          phoneme probability distributions via CTC-trained RNN

Stage 2 — Viterbi Search          (not implemented)
          Selects the maximum-likelihood phoneme path
          through the per-timestep posterior distributions

Stage 3 — Kaldi Trigram LM        (not implemented)
          Beam search over 125,000-word vocabulary combines
          phoneme path with language statistics to produce
          final word sequence
```

This repo covers Stage 1. The 78.77% accuracy is phoneme-level CER evaluated directly on GRU output — the paper reports ~80.3% at this same stage (19.7% phoneme error rate) prior to language model decoding.

---

## Model Architecture

```
Input: T x 256
       128 electrodes x 2 feature types:
       threshold crossings (TX) + spike band power (SP), binned at 20ms

Gaussian Smoothing (sigma=2.0)
       depthwise conv smoothing over the time dimension

Day-specific Transform (per session)
       (1) matrix multiply + bias [dayWeights, dayBias], initialized to identity
       (2) per-day Linear layer, weights initialized to I + Linear.weight
       learned correction for inter-session electrode drift

Softsign Nonlinearity

Temporal Unfolding (kernel=32, stride=4)
       stacks a 32-bin (640ms) sliding window into the feature dim
       output: T/4 x (256*32) — reduces sequence length 4x

Bidirectional GRU — 5 layers, hidden_dim=512
       orthogonal init on recurrent weights, Xavier on input weights
       dropout between layers

Linear Projection -> 41 logits
       40 phonemes + 1 CTC blank token

CTC Loss (training) / Greedy Decode (inference)
```

**Day-specific transforms** are critical for multi-session generalization — electrode signals drift across recording days as arrays settle in tissue. Without per-session correction, inter-day drift causes significant accuracy degradation.

**CTC training** removes the need for frame-level phoneme alignment labels, which are unavailable in neural data. The model learns to emit phoneme probabilities at variable positions; the CTC objective marginalizes over all valid alignments during backpropagation.

---

## Implementation

The original codebase targets a HPC cluster running the full pipeline including Kaldi beam search. This implementation isolates Stage 1 from the full Kaldi pipeline and making two targeted adjustments enables training on a single consumer GPU:

| Hyperparameter | Original | This Repo | Notes |
|---|---|---|---|
| `nUnits` (GRU hidden dim) | 1024 | **512** | 2x smaller; ~75% fewer recurrent parameters |
| `batchSize` | 64 | **16** | 4x reduction in per-step GPU memory |
| `dropout` | 0.4 | 0.4 | Unchanged |
| `nLayers` | 5 | 5 | Unchanged |
| `bidirectional` | True | True | Unchanged |
| `kernelLen` | 32 | 32 | Unchanged |
| `lrStart` / `lrEnd` | 0.02 | 0.02 | Unchanged |

All parameters are defined and annotated in `train_model.py`.

---

## Results

| | This Repo | Willett et al. (2023) |
|---|---|---|
| Stage | GRU decoder only | GRU + Viterbi + Kaldi LM |
| Metric | Phoneme accuracy | Word error rate |
| Score | **78.77%** | 9.1% WER (50-word) / ~80.3% phoneme acc. |
| Hardware | Single consumer GPU | Multi-GPU HPC cluster |

Reproduces the Stage 1 phoneme decoding result from a landmark *Nature* 2023 paper to within 1.5 percentage points on consumer hardware, confirming that strong phoneme-level representations are recoverable from the neural signal without the full Kaldi decoding stack.

---

## Repository Structure

```
speech-bci-decoder/
├── train_model.py                     # Entry point: all hyperparameters + training launch
├── src/
│   └── neural_decoder/
│       ├── model.py                   # GRUDecoder architecture
│       ├── neural_decoder_trainer.py  # Training loop, evaluation, model saving/loading
│       ├── dataset.py                 # SpeechDataset: loads pickle, returns tensors
│       └── augmentations.py           # GaussianSmoothing, WhiteNoise, MeanDriftNoise
├── notebooks/
│   └── formatCompetitionData.ipynb    # converts raw .mat files to pickle
└── README.md
```




---

## How to Run

Download the dataset from [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq).

**Step 1 — Preprocess raw data**

```bash
jupyter notebook notebooks/formatCompetitionData.ipynb
```

Converts raw `.mat` session files into a `ptDecoder_ctc` pickle containing z-scored neural features and phoneme label sequences, split into `train` and `test` sets by session.

**Step 2 — Configure and train**

Set paths in `train_model.py`:
```python
args['outputDir']   = './outputs'
args['datasetPath'] = './data/ptDecoder_ctc'
```

```bash
python train_model.py

# or override paths via CLI
python train_model.py --output_dir ./outputs --dataset_path ./data/ptDecoder_ctc
```

Evaluates on the test set every 100 batches, printing CTC loss and phoneme CER. Best checkpoint (minimum CER) saved to `outputDir/modelWeights`.


**Loading a saved model**
```python
from neural_decoder.neural_decoder_trainer import loadModel
model = loadModel('./outputs', device='cuda')
model.eval()
```

---

## References

- [A high-performance speech neuroprosthesis](https://doi.org/10.1038/s41586-023-06377-x)
- PyTorch decoder: [cffan/neural_seq_decoder](https://github.com/cffan/neural_seq_decoder)
- Dataset: [Dryad doi:10.5061/dryad.x69p8czpq](https://datadryad.org/stash/dataset/doi:10.5061/dryad.x69p8czpq)
