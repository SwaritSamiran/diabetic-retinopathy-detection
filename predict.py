# use the trained model to make predictions on images
import os
import torch
import numpy as np
import argparse
import cv2

from src.data_loader import get_transforms, ben_graham_processing
from src.models.architecture import build_model
from src.utils.predictor import load_model

# loading the model
MODEL_SAVE_PATH = './models/best_model.pth'
# optimal thresholds found from training validation set optimization
THRESHOLDS = np.array([0.76508932, 1.47002369, 2.67058553, 2.94852521])

def predict_image(image_path):
    # predict on a single image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load model
    model = build_model(device)
    model = load_model(lambda d: model, MODEL_SAVE_PATH, device)
    
    # get transforms
    train_aug, val_aug = get_transforms()
    
    # read and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = ben_graham_processing(img)
    img_tensor = val_aug(img).unsqueeze(0).to(device)
    
    # make prediction
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)
    
    prediction_value = pred.item()
    
    # convert to class using thresholds
    class_pred = np.digitize(prediction_value, THRESHOLDS)
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
    print(f"\nPrediction Value: {prediction_value:.4f}")
    print(f"Predicted Class: {class_pred} ({class_names[class_pred]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions on retinal images")
    parser.add_argument("image_path", type=str, help="Path to the image")
    args = parser.parse_args()
    
    predict_image(args.image_path)
