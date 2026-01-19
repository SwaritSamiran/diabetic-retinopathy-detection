# model loading and prediction utilities
import torch
import os


def load_model(model_class, model_path, device):
    # load trained model checkpoint
    model = model_class(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}")
    return model


def predict_single_image(model, image_path, device, transforms_fn):
    # predict on a single image
    import cv2
    from src.data_loader import ben_graham_processing
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ben_graham_processing(img)
    
    if transforms_fn:
        img = transforms_fn(img)
    
    img = img.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        pred = model(img)
    
    return pred.item()


def predict_with_thresholds(prediction_value, thresholds):
    return np.digitize(prediction_value, thresholds)
