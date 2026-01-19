# utility functions for the project
import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional


def get_device() -> torch.device:
    # get device (cuda or cpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device


def set_seed(seed: int = 42):
    # set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    # create directory if it doesn't exist
    path.mkdir(parents=True, exist_ok=True)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    # load model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model


def save_checkpoint(model: torch.nn.Module, checkpoint_path: str):
    # save model checkpoint
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
