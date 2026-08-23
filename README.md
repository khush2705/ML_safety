# ML Safety — CARLA Autonomous Driving Perception

This project evaluates the safety of a CARLA-based autonomous-driving perception system using machine-learning safety techniques across Exercises 1–9.

The system contains three independent binary **ResNet-18** classifiers:

- Pedestrian detection
- Traffic-light detection
- Vehicle detection

The project goes beyond normal accuracy evaluation and investigates robustness, uncertainty, distribution shift, explainability, adversarial attacks, OOD detection, and training-data security.

---

## 1. Project Structure

Recommended structure:

```text
ML_Safety/
├── data/
│   ├── train/
│   ├── test/
│   ├── test-fog/
│   ├── test-night/
│   └── test-town-01/
├── models/
├── notebooks/
├── outputs/
├── report/
└── README.md
```

During the experiments, the main local project path was:

```text
D:\ML_safty
```

The recorded dataset contained:

- Training: **7,200 images**
- Validation: **3,600 images**
- Test: **3,600 images**

The shifted test sets were `test-fog`, `test-night`, and `test-town-01`.

---

## 2. Environment Setup

The project was developed using Python, VS Code/Jupyter and PyTorch.

Typical packages:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow jupyter
```

Create a virtual environment if required:

```bash
python -m venv .venv
.venv\Scripts\activate
```

The recorded training experiments used:

```text
NVIDIA GeForce RTX 3070 Ti Laptop GPU
CUDA
```

---

# 3. Training the Three Models

All three classifiers use the same basic architecture and training procedure.

### Model configuration

| Setting | Value |
|---|---|
| Backbone | Pretrained ResNet-18 |
| Classes | 2 |
| Loss | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 64 |
| Epochs | 10 |
| Input size | 224 × 224 |

ImageNet normalization:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Training process

For **each** model:

```text
Load dataset
   ↓
Resize + normalize images
   ↓
Load pretrained ResNet-18
   ↓
Train binary classifier
   ↓
Validate after each epoch
   ↓
Keep best validation checkpoint
   ↓
Evaluate on test data
```

The models are kept separate so that pedestrian, traffic-light and vehicle performance can be analysed independently.

---

# 4. What We Did in Exercises 1–9

### Exercise 1 — Safety Context
Established the ML safety and security problem and connected perception failures to possible system-level consequences.

### Exercise 2 — ODD + STPA
Defined the operating conditions, losses, hazards, unsafe control actions and causal loss scenarios for the CARLA system.

### Exercise 3 — Model Training
Explored the dataset, trained the three ResNet-18 classifiers and evaluated accuracy, precision, recall, F1 and confusion matrices.

### Exercise 4 — ODD / Distribution Shift
Tested the models on:

```text
Normal test
Fog
Night
Town-01
```

This showed that performance can degrade significantly when the input distribution changes.

### Exercise 5 — Calibration + Backdoor
Performed temperature scaling to study confidence calibration and conducted a separate poisoning/backdoor experiment on the pedestrian detector.

The backdoor used a **10×10 red-square trigger** and demonstrated the importance of training-data integrity.

### Exercise 6 — Grad-CAM
Generated Grad-CAM explanations for correct and incorrect predictions under normal and shifted conditions to investigate model attention.

### Exercise 7 — Uncertainty / Calibration
Analysed epistemic and aleatoric uncertainty, ECE, reliability diagrams, temperature scaling and safety-related confidence decisions.

### Exercise 8 — FGSM
Generated adversarial examples using FGSM with:

```text
ε = 0.01
ε = 0.05
ε = 0.10
```

Recall degradation was measured between clean and adversarial inputs.

### Exercise 9 — OOD Detection
Compared Maximum Softmax Probability (MSP) with a deep-feature **5-nearest-neighbour (k-NN)** detector.

The Traffic-Light ResNet-18 feature representation was reduced to a **512-dimensional feature vector**, and larger k-NN distances were treated as stronger OOD evidence.

---

# 5. Important Results

### Nominal recall

| Model | Recall |
|---|---:|
| Pedestrian | 10.48% |
| Traffic Light | 97.76% |
| Vehicle | 93.19% |

The pedestrian detector is the main nominal-performance weakness.

### FGSM at ε = 0.05

| Model | Clean Recall | Adversarial Recall | Drop |
|---|---:|---:|---:|
| Pedestrian | 10.48% | 0.00% | 10.48 pp |
| Traffic Light | 97.76% | 0.04% | 97.72 pp |
| Vehicle | 93.19% | 23.22% | 69.96 pp |

Traffic-light and vehicle models show severe adversarial sensitivity.

### k-NN OOD detection

| Condition | AUROC |
|---|---:|
| Fog | 0.9907 |
| Night | 0.9795 |
| Town-01 | 0.9556 |

The deep-feature k-NN detector performed strongly for the evaluated OOD conditions.

---

# 6. Calibration Results

Temperature scaling was optimized using validation data and evaluated on the test set.

| Model | Optimal T | ECE Before | ECE After |
|---|---:|---:|---:|
| Pedestrian | 2.0415 | 0.0551 | 0.0766 |
| Traffic Light | 1.2839 | 0.0459 | 0.0379 |
| Vehicle | 1.4451 | 0.0326 | 0.0238 |

Calibration improved for the traffic-light and vehicle models but worsened for the pedestrian model.

---

# 7. Final Safety Conclusion

The experiments show that the system cannot be considered unconditionally safe.

The major findings are:

- Very low pedestrian recall
- Severe performance degradation under some distribution shifts
- Strong FGSM sensitivity for traffic-light and vehicle detection
- Calibration improves only some models
- k-NN provides strong OOD detection in the evaluated experiment
- The backdoor experiment demonstrates training-data/supply-chain risk
- Human fallback remains an important part of the safety argument

The final recommendation is therefore:

> **Deploy with restrictions**

The system should only be considered in the demonstrated course-level operating context with appropriate human supervision and further safety improvements.

---

# 8. Reproducing the Project

Run the work in this order:

```text
1. Prepare dataset
2. Train pedestrian model
3. Train traffic-light model
4. Train vehicle model
5. Save best checkpoints
6. Run nominal evaluation
7. Test fog/night/Town-01
8. Run Grad-CAM
9. Run calibration
10. Run backdoor experiment separately
11. Run MSP OOD
12. Run k-NN OOD
13. Run FGSM
14. Collect results
15. Update the final safety case
```

Keep the same dataset, preprocessing, checkpoints and experiment configuration when reproducing the reported results.
