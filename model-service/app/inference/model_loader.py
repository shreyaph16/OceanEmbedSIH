import torch
import os
from ..models import VarunaGATBackbone, RegionConditionedDecoder

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "varuna_regional_final_30ep.pt")
ARTIFACTS_PATH = os.path.join(CHECKPOINT_DIR, "varuna_inference_artifacts.pt")


class ModelBundle:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on device: {self.device}")

        artifacts = torch.load(ARTIFACTS_PATH, map_location=self.device, weights_only=False)
        self.edge_index = artifacts["edge_index"].to(self.device)
        self.static_directions = artifacts["static_directions"]
        self.region_onehot_tensor = artifacts["region_onehot_tensor"].to(self.device)
        self.lat_grid = artifacts["lat_grid"]
        self.lon_grid = artifacts["lon_grid"]
        self.target_depths = artifacts["target_depths"]

        ckpt = torch.load(CHECKPOINT_PATH, map_location=self.device, weights_only=False)

        self.backbone = VarunaGATBackbone(in_channels=4).to(self.device)
        self.backbone.load_state_dict(ckpt["backbone"])
        self.backbone.eval()

        self.decoder = RegionConditionedDecoder(latent_dim=64 * 4, n_regions=3, n_depths=15).to(self.device)
        self.decoder.load_state_dict(ckpt["region_decoder"])
        self.decoder.eval()

        print("Model loaded successfully.")


_bundle = None

def get_model_bundle():
    global _bundle
    if _bundle is None:
        _bundle = ModelBundle()
    return _bundle