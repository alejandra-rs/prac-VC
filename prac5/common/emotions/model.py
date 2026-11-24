import cv2
import numpy as np
import torch
import torch.nn.functional as F
from deepface import DeepFace
from deepface.models.demography.pytorch.Emotion import load_model as load_deepface_emotion
from deepface.modules import modeling
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .data import EMOTIONS
from .images import to_unit


class ResNetEmotion(nn.Module):
    def __init__(self, img_size, dropout, pretrained=True):
        super().__init__()
        self.img_size = img_size
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.net.fc.in_features, len(EMOTIONS)))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
        return self.net((x - self.mean) / self.std)

    def head(self):
        return list(self.net.fc.parameters())

    def body(self):
        return [p for name, p in self.net.named_parameters() if not name.startswith("fc.")]

    def freeze_body(self, frozen):
        for p in self.body():
            p.requires_grad_(not frozen)


class DeepFaceEmotion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = load_deepface_emotion()

    def forward(self, x):
        return torch.log(self.net(x).clamp_min(1e-7))


class TorchEmotionPredictor:
    def __init__(self, model, batch_size=128):
        self.model = model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def predict_proba(self, faces):
        device = next(self.model.parameters()).device
        probabilities = []
        for i in range(0, len(faces), self.batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(faces[i:i + self.batch_size])).to(device)
            probabilities.append(torch.softmax(self.model(to_unit(batch)).float(), dim=1).cpu().numpy())
        return np.concatenate(probabilities)

    def predict(self, faces):
        return self.predict_proba(faces).argmax(1)


class DeepFaceAnalyze:
    def __init__(self, device):
        client = modeling.build_model(task="facial_attribute", model_name="Emotion")
        client.model.to(device)
        client.device = torch.device(device)

    def predict_one(self, face):
        result = DeepFace.analyze(img_path=cv2.cvtColor(face, cv2.COLOR_GRAY2BGR), actions=["emotion"],
                                  detector_backend="skip", enforce_detection=False, silent=True)[0]
        probabilities = np.array([result["emotion"][emotion] for emotion in EMOTIONS])
        return probabilities / probabilities.sum()

    def predict(self, faces):
        return np.array([self.predict_one(face).argmax() for face in faces])
