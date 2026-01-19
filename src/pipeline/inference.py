# inference pipeline functions 
import torch
import numpy as np
from tqdm.auto import tqdm


def get_tta_predictions(model, loader, device):
    # tta predictions
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            p1 = model(images)
            p2 = model(torch.flip(images, [3]))
            avg_preds = (p1 + p2) / 2
            all_preds.extend(avg_preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds).flatten(), np.array(all_labels)


def get_stable_prediction(model, img_tensor, device):
    # stability check
    model.eval()
    versions = [img_tensor, torch.flip(img_tensor, [3]), torch.flip(img_tensor, [2]), torch.rot90(img_tensor, 1, [2, 3])]
    results = []
    with torch.no_grad():
        for v in versions:
            results.append(model(v.to(device)).item())
    mean_val = np.mean(results)
    stability = 1.0 - np.std(results)
    verdict = "stable" if stability > 0.85 else "unstable"
    return mean_val, stability, verdict


def get_preds_with_tta(model, loader, device):
    # tta predictions with 3 versions
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            
            # original
            out1 = model(images)
            # horizontal flip
            out2 = model(torch.flip(images, dims=[3]))
            # vertical flip
            out3 = model(torch.flip(images, dims=[2]))
            
            # average the raw regression values
            avg_out = (out1 + out2 + out3) / 3.0
            
            all_preds.extend(avg_out.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.array(all_preds).flatten(), np.array(all_labels)
