# Diabetic Retinopathy Detection - Pre-trained Model

A production-ready inference system for diabetic retinopathy severity classification using continuous regression with optimized threshold mapping.

## Key Technical Decisions

**Regression vs Classification:** The model predicts continuous values (0-4) rather than discrete classes. This preserves the ordinal nature of disease severity and improves performance on ordinal metrics.

**Generalized Mean Pooling:** Standard average pooling loses micro-detail information critical for DR detection. Generalized Mean (GeM) pooling with p=3 acts as a feature concentrator, emphasizing small, sharp features.

**Threshold Optimization:** Rather than standard fixed thresholds, thresholds were optimized using Nelder-Mead simplex method to maximize Quadratic Weighted Kappa on the validation set:
```
[0.765089, 1.470024, 2.670586, 2.948525]
```

**Test-Time Augmentation:** Predictions are averaged across the original image, horizontal flip, and vertical flip to reduce prediction variance from image artifacts and noise.

## Architecture

- **Backbone:** EfficientNet-B3 (pretrained ImageNet)
- **Pooling:** Generalized Mean (p=3)
- **Head:** Single linear output (continuous regression)
- **Loss:** HuberLoss (robust to outliers)

## Preprocessing

1. **Circular Cropping:** Remove black border (~40% of raw image)
2. **Gaussian Contrast Enhancement:** Amplify pathological features (vessels, hemorrhages, microaneurysms)
3. **Normalization:** ImageNet statistics

## Project Structure

```
diabetic-retinopathy-detection/
├── predict.py                      # Inference entry point
├── models/
│   └── best_model.pth              # Pre-trained weights
├── src/
│   ├── models/
│   │   └── architecture.py         # EfficientNet-B3 + GeM
│   ├── pipeline/
│   │   ├── inference.py            # TTA + threshold mapping
│   │   └── threshold_optimizer.py  # Nelder-Mead optimization
│   ├── utils/
│   │   └── predictor.py            # Model loading & inference
│   ├── data_loader.py              # Image preprocessing
│   ├── logger.py
│   └── exception.py
├── notebook/
│   └── kaggle_notebook.ipynb       # Training notebook (reference)
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <repo-url>
cd diabetic-retinopathy-detection

python -m venv venv
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python predict.py path/to/image.png
```

Output:
```
Prediction Value: 2.34
Predicted Class: 2 (Moderate)
```

### Python API

```python
import torch
from src.utils.predictor import load_model
from src.data_loader import ben_graham_processing, get_transforms
import cv2

# Load model
model = load_model('models/best_model.pth')

# Prepare image
img = cv2.imread('image.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = ben_graham_processing(img)

# Get transforms and prepare tensor
_, val_aug = get_transforms()
img_tensor = val_aug(img).unsqueeze(0)

# Inference
model.eval()
with torch.no_grad():
    prediction = model(img_tensor).item()
print(f"Prediction: {prediction:.4f}")
```

### With Test-Time Augmentation

```python
from src.pipeline.inference import get_stable_prediction

pred, stability, status = get_stable_prediction(model, img_tensor)
print(f"Prediction: {pred:.4f}, Stability: {stability:.4f}")
```

## Performance

- **Quadratic Weighted Kappa:** 0.8865
- **Validation Accuracy:** 80.55%
- **Inference Speed:** 100-200ms per image (GPU), 1-2s per image (CPU)

## Disease Severity Scale

- **0:** No DR (No Diabetic Retinopathy)
- **1:** Mild
- **2:** Moderate
- **3:** Severe
- **4:** Proliferative DR

## Training Details

Full training pipeline documented in `notebook/kaggle_notebook.ipynb`:
- 10 epochs, batch size 16
- HuberLoss + AdamW (lr=1e-4)
- 85-15 train/validation split
- APTOS 2019 dataset (~3,600 images)

## Dependencies

```
torch
torchvision
pandas
numpy
opencv-python
scikit-learn
scipy
matplotlib
seaborn
tqdm
pillow
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Model file not found" | Verify `models/best_model.pth` exists |
| "CUDA out of memory" | Use CPU: set `device='cpu'` in predict.py |
| "Module not found" | Run `pip install -r requirements.txt` |
| Slow inference on CPU | Consider using GPU (NVIDIA with CUDA) |

## Disclaimer

This is a research tool for educational and experimental purposes. It is not a medical device and should not be used for clinical diagnosis. Always consult qualified healthcare professionals for medical decisions.

## References

- Dataset: [APTOS 2019 Blindness Detection - Kaggle](https://www.kaggle.com/competitions/aptos2019-blindness-detection/)
- EfficientNet: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- Generalized Mean Pooling: Feature aggregation via power mean
- Ben Graham Preprocessing: Kaggle forum discussion on retinal image normalization

## Contributors

This project was developed collaboratively by:
- [Kavya Baxi](https://github.com/kavay-dev)
- [Swarit Samiran](https://github.com/SwaritSamiran)

## Acknowledgments

This solution draws inspiration from several APTOS 2019 competition winners and top performers. Key techniques including continuous regression, threshold optimization, and test-time augmentation strategies were informed by public solutions and discussions from the competition.
