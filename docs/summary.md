# DCGAN Brain Tumor MRI Synthesis — Implementation Summary

Paper: "Generative Adversarial Synthesis and Deep Feature Discrimination of Brain Tumor MRI Images"
Md Sumon Ali, Muzammil Behzad (King Fahd University of Petroleum and Minerals, Saudi Arabia), arXiv:2511.01574 (November 2025)

> This file contains ONLY facts extracted from the paper (architecture, hyperparameters, dataset,
> loss functions, results) plus implementation tasks derived from them. No solutions, no code.

---

## 1. Core Idea (Abstract, Section I)

The paper addresses the problem of limited medical imaging data (MRI scans are expensive/effortful
to collect). A DCGAN (Deep Convolutional GAN) is trained to generate synthetic brain MRI images.
A separate CNN classifier is then trained on both real and synthetic images to classify brain
tumors — as a test of whether the generated images are "useful enough" to supplement real data.

The project therefore has two parts:
1. **DCGAN**: Generator + Discriminator, produces synthetic 128×128 grayscale MRI images
2. **CNN classifier**: Classifies tumor / no tumor, trained on a mix of real and synthetic images

---

## 2. Dataset (Section III.B)

- **Source**: "Brain Tumor MRI Images for Brain Tumor Detection" from Kaggle (two referenced
  Kaggle datasets, see references [38], [39] in the paper)
- **Size**: 3000 grayscale MRI scans, 1500 labeled "yes" (tumor present), 1500 labeled "no"
  (no tumor)
- **Preprocessing**:
  - Each image resized to 128×128 pixels
  - Converted to grayscale
  - Pixel values normalized to range [-1, 1] (matching the generator's tanh output)
  - Compiled into a NumPy array

---

## 3. DCGAN Architecture (Section III.B "GAN Training")

### 3.1 Generator
- Input: random noise vector of size **400**
- Dense input layer, projects to a feature map of size **32×32×256**
- Followed by a sequence of **transposed convolutional layers** that progressively upscale
  spatial resolution to 128×128
- Each convolutional block uses **LeakyReLU** activations
- Final convolutional layer: **single output channel**, **tanh** activation (normalized
  grayscale MRI images as output)

### 3.2 Discriminator
- **Four convolutional layers** that progressively downsample the input from 128×128 to
  **16×16** feature maps
- Each followed by **LeakyReLU** activations and **dropout** regularization
  (to mitigate overfitting)
- Final layer: flattened feature map → dense layer with **sigmoid** output
  (classification: real vs. synthetic)

### 3.3 Additional enhancements discussed in the paper (Section III.A "Proposed Enhancements")
These are mentioned/discussed in the methodology section but not specified in detail for the
final implementation (see Section 8 "Open Questions" below):
- Progressive increase of resolution/depth during training ("starts training with
  low-resolution images and progressively increases the depth and resolution")
- **Spectral normalization** in the discriminator (against exploding gradients)
- Inspiration from **ACGAN** (Auxiliary Classifier GAN): the discriminator should additionally
  be able to classify tumor type, beyond real/fake discrimination
- Data augmentation via stretching/brightness changes

---

## 4. GAN Training Details (Sections III.B, III.C)

- **Noise vector size**: 400
- **Training epochs**: 10
- **Steps per epoch**: 3750
- **Batch size**: 4
- **Optimizer**: Adam
- **Adam parameters**: learning rate = **0.0002**, β₁ = **0.5**, β₂ = **0.999**

### 4.1 GAN Loss Functions (Section III.C, Equations 1-3)

Standard GAN minimax objective:

```
min_G max_D V(D,G) = E_{x~p_data(x)}[log D(x)] + E_{z~p_z(z)}[log(1 - D(G(z)))]
```

where:
- p_data(x): real data distribution (MRI images)
- p_z(z): prior distribution over input noise (latent space)
- G(z): generator output (fake MRI image)
- D(x): discriminator's estimated probability that x is real

Implemented as **Binary Cross-Entropy Loss**, separately for discriminator and generator:

Discriminator loss:
```
L_D = -E[log D(x)] - E[log(1 - D(G(z)))]
```

Generator loss:
```
L_G = -E[log D(G(z))]
```

---

## 5. CNN Classifier Architecture (Section III.B "CNN Classification")

- Multiple blocks of: **Conv2D layers** + **ReLU** activation + **batch normalization** +
  **MaxPooling2D**, combined with **dropout** regularization
- Final block: **GlobalAveragePooling2D** for dimensionality reduction
- Then: dense layer with **1024 neurons**
- Output layer: **1 neuron**, **sigmoid** activation (binary classification: tumor / no tumor)

### 5.1 CNN Training Details
- **Optimizer**: Adam, learning rate = **0.0005**
- **Training epochs**: up to 200
- Callbacks: **early stopping**, **learning rate reduction** (ReduceLROnPlateau, starting from
  LR 0.0005, reduced based on validation loss), **model checkpointing**
- **Loss function**: cross-entropy loss

```
L_CE = -Σ y_i * log(ŷ_i)
```

where y_i is the ground-truth label and ŷ_i is the predicted probability.

---

## 6. Experiment Workflow (Section IV.A, algorithmic summary)

1. Load and preprocess the MRI dataset (see Section 2)
2. Train the DCGAN: alternating generator-discriminator updates
3. Save synthetic images at specific epochs
4. Merge real and synthetic images and augment them
5. Split the dataset into train/test
6. Train the CNN classifier on the merged (real + synthetic) training set
7. Evaluate the CNN classifier on the merged test set
8. Compare performance metrics between real and synthetic images

### 6.1 Concrete numbers for the dataset split (Section IV.A)
- 400 synthetic "yes" (tumor) images generated, merged with original tumor images
  → 1900 tumor images total
- 400 "no" (no tumor) images additionally augmented (to restore balance, since fewer "no"
  images were available after adding the synthetic tumor images)
- Train/test split: **80/20**
- Test set size: **n=760**

---

## 7. Results from the Paper (for cross-checking)

### 7.1 Qualitative observations on GAN training (Section IV.B)
- Epochs 1-3: generated images blurry, little structure visible
- Epochs 4-5: clearer details, tumor shapes begin to appear
- From epoch 6 onward: markedly improved quality, brain structures and tumors clearly visible
- Epoch 10 (final): images "almost like real" MRI scans with clear tumor details

### 7.2 CNN Classification Results (Table II)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| No Tumor | 1.00 | 0.99 | 0.99 | 380 |
| Tumor | 0.99 | 1.00 | 0.99 | 380 |
| **Accuracy** | | **0.99** | | 760 |
| Macro Avg | 0.99 | 0.99 | 0.99 | 760 |
| Weighted Avg | 0.99 | 0.99 | 0.99 | 760 |

- Final stabilized training/validation accuracy: **~99.7%**
- Confusion matrix (test set): only 5 false positives, 0 false negatives

### 7.3 Comparison with other studies (Table III)

| Study | Dataset | Model | Metric(s) | Best Result |
|---|---|---|---|---|
| This study | Kaggle Brain MRI (3k) | DCGAN, CNN | Accuracy, Precision, Recall, F1 | Accuracy = 99.7%, F1 = 0.99 |
| Han et al. [8] | BraTS 2016 (HGG) | DCGAN vs. WGAN | Visual Turing Test (physician) | WGAN ≈ 53-55% accuracy (chance = 50%) |
| Mukherjee et al. [23] | Brain Tumor & BraTS 2020 | AGGrGAN (DCGAN, WGAN, style transfer) | SSIM, PSNR, KL, SD, CNN accuracy | SSIM=0.83, PSNR=23.7dB, accuracy=94% (VGG-19) |

---

## 8. Known Limitations of GAN Training (from the literature review, Section II.B)

These are general GAN issues discussed in the paper that you should watch for during your own
training:
- **Mode collapse**: generator produces only a few, very similar image variants
- **Vanishing gradients**: unstable training dynamics
- **Missing anatomical detail**: images can look visually realistic but lack important
  diagnostic detail
- Adversarial training is generally sensitive to hyperparameter choice

---

## 9. IMPLEMENTATION TASKS

These tasks describe WHAT should be built, not HOW. The order is a suggested sensible
progression, not a requirement.

### Phase 1 — Data
- [ ] Download the Kaggle dataset "Brain Tumor MRI Images for Brain Tumor Detection"
      (see references [38], [39] in the paper for the exact Kaggle links)
- [ ] Implement the preprocessing: resize to 128×128, grayscale conversion,
      normalization to [-1, 1]
- [ ] Build a sanity check: display a few preprocessed images, verify the value range

### Phase 2 — DCGAN Architecture
- [ ] Implement the generator per Section 3.1 (noise vector → dense →
      32×32×256 → transposed convolutions → 128×128×1, tanh output)
- [ ] Implement the discriminator per Section 3.2 (4 conv layers, 128×128 → 16×16,
      LeakyReLU + dropout, final dense layer with sigmoid)
- [ ] Implement the GAN loss functions (discriminator loss and generator loss,
      Section 4.1) as binary cross-entropy

### Phase 3 — GAN Training
- [ ] Build the training loop: alternating updates of discriminator and generator
- [ ] Use the Adam optimizer with the specified hyperparameters (LR=0.0002, β1=0.5, β2=0.999)
- [ ] Train over 10 epochs with batch size 4 (note: 3750 steps/epoch at batch size 4 implies
      roughly 15,000 training images per epoch — check whether this matches your actual
      dataset size of 3000 images, or whether repetition/resampling is needed — the paper
      does not specify this exactly)
- [ ] Save sample generated images after each epoch (for visualizing training progress,
      see Section 7.1)
- [ ] Track generator loss and discriminator loss over time

### Phase 4 — Extensions (optional, from Section III.A)
- [ ] Add spectral normalization in the discriminator
- [ ] Experiment with progressive training (low resolution → higher resolution)
- [ ] Optional: ACGAN extension (discriminator additionally classifies tumor type)

### Phase 5 — CNN Classifier
- [ ] Implement the CNN architecture per Section 5 (Conv2D blocks with batch norm,
      max pooling, dropout; GlobalAveragePooling2D; dense with 1024 neurons; sigmoid output)
- [ ] Use the Adam optimizer with LR=0.0005, cross-entropy loss
- [ ] Implement early stopping, ReduceLROnPlateau, model checkpointing

### Phase 6 — Combined Experiment
- [ ] Generate 400 synthetic "tumor" images with your trained generator
- [ ] Merge them with the real images (see numbers in Section 6.1)
- [ ] Augment the "no tumor" class accordingly for balance
- [ ] Train the CNN classifier on the mix, 80/20 split
- [ ] Compute precision, recall, F1-score, confusion matrix on the test set
- [ ] Compare your results against Table II (Section 7.2) — do you land in a similar range
      to ~99% accuracy? Larger deviations are a signal to debug

### Phase 7 — Visualization & Demo (your actual goal!)
- [ ] Build a small interactive demo: button "Generate new MRI image" → shows a freshly
      generated sample from the trained generator
- [ ] Show an image gallery of generated images across different training epochs
      (see Fig. 3/4 in the paper — progression from "blurry" to "realistic")
- [ ] Optional: show generator output alongside a CNN classifier prediction
      ("this generated image is classified as 'tumor' by the classifier")

---

## 10. Open Questions / Things the Paper Does NOT Specify (decide yourself!)

- The exact number/structure of transposed convolutional layers in the generator (how many
  layers between 32×32×256 and 128×128×1 exactly?) is not precisely specified — only the
  start and end points
- The exact number and filter sizes of the 4 discriminator conv layers are not specified
  (only "4 layers, 128×128 → 16×16")
- Dropout rate in the discriminator is mentioned but no concrete value given
- How exactly "3750 steps per epoch" at "batch size 4" fits with a dataset of 3000 images
  is not entirely clear (this would be more steps than a full pass through the data would
  need) — your own decision needed on how to handle this (e.g. with repetition/resampling)
  or whether to choose a different number of steps
- The exact kernel sizes and strides of the conv/transposed-conv layers are not given
- Which exact data augmentation techniques ("stretching and changing brightness") were used
  is only vaguely described, without concrete parameters
- For the "Proposed Enhancements" (spectral normalization, ACGAN, progressive training), it's
  unclear whether these were actually used in the final experiments or are only meant as
  discussion/outlook — they are not mentioned again in the methodology section (III.B), only
  in the preceding section III.A