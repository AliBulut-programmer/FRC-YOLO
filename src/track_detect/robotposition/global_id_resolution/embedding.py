import torch
import torchvision.transforms.v2 as T
import numpy as np


class ViT:
    def __init__(self, model="dinov2_vits14"):
        self.model = torch.hub.load("facebookresearch/dinov2", model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.transformations = T.Compose(
            [
                T.ToPILImage(),
                T.Resize((252, 252)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def extract(self, crop):
        tensor = self.transformations(crop).unsqueeze(0).to(self.device)
        features = self.model(tensor)
        return features.squeeze().cpu().numpy()
