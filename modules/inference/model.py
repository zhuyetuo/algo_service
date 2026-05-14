from pathlib import Path

import torch

from config import settings


class BehaviorModel:
    """Loads the behavior-recognition model once and exposes a predict method."""

    def __init__(self):
        self.device = torch.device(settings.inference_device)
        weights = Path(settings.model_weights_path)
        # Replace with your actual model architecture load
        self.model = torch.jit.load(weights, map_location=self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, frames: list) -> dict:
        # frames: list of numpy HWC arrays
        # TODO: preprocess → tensor → model → decode output
        raise NotImplementedError


# Singleton – loaded once at startup
_model: BehaviorModel | None = None


def get_model() -> BehaviorModel:
    global _model
    if _model is None:
        _model = BehaviorModel()
    return _model
