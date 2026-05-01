# TFLite Model Maker Studio

A desktop application for the complete end-to-end workflow of building TensorFlow Lite object detection models — from raw video to a deployment-ready `.tflite` file. Built with Python and Tkinter, it provides a tabbed GUI that walks you through every stage of the pipeline.

## Launch

Create your private environment file first:

```bash
copy .env
```

Fill `.env` with your local dataset paths, model settings, and hyperparameters. The real `.env` file is ignored by git so those values stay out of the public repo.

```bash
python GuiModelMaker__MAIN__.py
```

The app opens with eight tabs, each covering one stage of the workflow. Below is the step-by-step walkthrough.

---

## Step 1 — Extract Frames from Video

**Tab: Frame Extraction**

This is where your training data originates. Point the app at a video file (or a directory of videos) and pull out still frames.

- Choose to extract every frame or every Nth frame (e.g. every 5th) to control dataset size.
- Set target resolution and interpolation method (INTER_LINEAR is a good default).
- Adjust JPEG quality and thread count for performance.
- The app creates a subdirectory per video and reports timing statistics when finished.

The output is a folder of images ready for annotation.

---

## Step 2 — Annotate

**Tab: Annotation**

Draw bounding boxes on your extracted frames to create Pascal VOC XML annotations. This is where the image/XML pairs originate.

- Load a directory of images and define your label set (e.g. `kettlebell`, `barbell`).
- Click and drag on the canvas to draw boxes. Each box is assigned the currently selected label.
- Navigate with Previous / Next / Skip. The app writes one XML per image automatically.

The annotation format is Pascal VOC — the same format consumed by the TFLite Model Maker training library downstream.

---

## Step 3 — Clean the Data

**Tab: Data Cleaning**

Now that annotation has produced XML files, clean up the dataset so every image/annotation pair is consistent.

- **Sync XML metadata** — fix filenames and paths inside annotation files so they match the actual images on disk. Supports recursive folder processing with multi-threaded execution.
- **Remove orphans** — delete images that have no matching XML or XMLs that have no matching image.
- **Detect outliers** — run statistical analysis on bounding box dimensions (aspect ratio, area percentile) and preview or auto-remove outliers.
- **Validate dimensions** — verify that image sizes reported in XMLs match the actual pixel dimensions.

This step catches annotation inconsistencies (mismatched paths, orphaned files, bounding box outliers) before you move on to augmentation and training.

---

## Step 4 — Preview and Validate

**Tab: Preview & Validate**

Visually inspect every annotation before committing to training.

- Browse through images with bounding box overlays drawn on the canvas.
- Toggle label text and confidence scores on/off; pick box colors.
- Delete bad samples (image + XML) directly from this view.
- Review dataset statistics: total images, annotated count, and per-class object counts.

Keyboard shortcuts (arrow keys, A/D, Space, Delete) let you move through the dataset quickly.

---

## Step 5 — Augment

**Tab: Augmentation**

Multiply your dataset and introduce variation so the model generalizes better.

**Color transforms:** random brightness, saturation, hue, contrast.
**Geometric transforms:** rotation, horizontal/vertical flip, translation, shear, perspective warp.
**Advanced:** random blur/sharpen, noise injection, GridMask regularization.

Configure the number of augmented copies to generate and the intensity of each transform. Output goes to a separate directory so the originals stay untouched.

---

## Step 6 — Split and Package the Dataset

**Tab: File Operations**

Organize the augmented dataset into the train/validation/test structure that the trainer expects.

- **Dataset Split** — set ratios (default 70/15/15) and the app copies image+XML pairs into `train/`, `validation/`, and `test/` subdirectories. Stratified splitting by class is supported.
- **Directory Comparison** — diff two directories to see what's unique or shared (useful after merging datasets).
- **Bulk Copy** — copy images and their annotation pairs matching a pattern into a target directory.

After this step you have three clean directories ready for training.

---

## Step 7 — Train the Model

**Tab: Training**

### ************ TAB STILL UNDER CONSTRUCTION *************
###     Train from: 

```bash
python main.py
```
### ************ TAB STILL UNDER CONSTRUCTION *************
Configure the EfficientDet-Lite architecture, set hyperparameters, and kick off training with CUDA.

Key settings:
| Category | Examples |
|---|---|
| **Architecture** | EfficientDet-Lite0 (320x320), backbone from efficientnet-b0 to b3 |
| **Basic** | epochs, batch size, learning rate, LR decay |
| **Loss** | alpha, gamma, delta, box loss weight, label smoothing |
| **Optimization** | momentum, weight decay, gradient clipping, moving average decay |
| **Augmentation** | horizontal flip prob, jitter range, AutoAugment policy, GridMask |
| **Anchors** | scales, aspect ratios, number of scales |
| **NMS** | method (Gaussian / standard), IoU threshold, score threshold, max outputs |

Training runs with mixed-precision (float16) on GPU. Checkpoints are saved automatically so you can resume if interrupted. The output is a timestamped directory containing:

- `model.tflite` — quantized, mobile-ready model
- `saved_model/` — full TensorFlow SavedModel for further work
- `label_map.txt` — class name mappings
- `training_log.txt` — epoch-by-epoch metrics
- `training_specs.txt` — full hyperparameter snapshot

---

## Step 8 — A/B Test with the Training Visualizer

### ********* TAB STILL UNDER CONSTRUCTION *************
###     Run visualizer from: 

```bash
python run_visualizer.py
```
### ************ TAB STILL UNDER CONSTRUCTION *************

**Tab: Training Visualizer**

This is where you compare training runs side-by-side to decide which hyperparameter configuration produces the best model.

### How it works

1. Select up to **3 training logs** from previous runs.
2. The visualizer parses each log and generates interactive Plotly charts:

| Plot | What it tells you |
|---|---|
| **Training vs. Validation Loss** | Overall convergence — is the model still learning or has it plateaued? |
| **Loss Components** | Breakdown into detection, classification, box, and regularization loss — pinpoints which loss term is dominating. |
| **Train/Val Gap** | Overfitting indicator — a growing gap means the model memorizes rather than generalizes. |
| **Learning Dynamics** | Learning rate schedule and gradient norm over time — helps diagnose instability. |
| **Loss Change Rate** | Epoch-to-epoch delta — shows when improvements taper off. |
| **Evaluation Metrics** | AP, AP50, AP75, APs, APm, APl — the COCO metrics that actually measure detection quality. |

### Multi-model comparison

When you load multiple logs, the visualizer overlays their curves on the same axes and generates a **comparison statistics table**. This makes it straightforward to see which run converges faster and which one achieves a higher final AP50.

### Export

Results can be exported as:
- **CSV** - one self-contained table with epoch metrics, evaluation metrics, model specs, hyperparameters, and summary stats, ready to paste into a language model.
- **Interactive HTML** — hover, zoom, and pan on every chart.
- **Static PNG** — for embedding in documents.
- **PDF report** — multi-page document with all plots and the full model spec card.

CSV export can be run from the visualizer menu or directly:

```bash
python run_visualizer.py --export-csv path/to/training_log.txt
python run_visualizer.py --export-csv --compare path/to/log1.txt path/to/log2.txt --out visualizer_data.csv
python run_visualizer.py --export-csv --recent path/to/models 10 --out recent_models.csv
```

---

## Typical Iteration Loop

```
Video  ──>  Frames  ──>  Annotate  ──>  Clean  ──>  Validate
                                                        │
         ┌──────────────────────────────────────────────┘
         v
     Augment  ──>  Split  ──>  Train  ──>  Visualize
                                  │              │
                                  └──── A/B ─────┘
                                  (adjust hyperparams
                                   and retrain)
```

You repeat the Train → Visualize loop, changing one or two hyperparameters at a time, until the evaluation metrics meet your target. The final `.tflite` model drops directly into a TensorFlow Lite Android (or iOS) application for inference.
