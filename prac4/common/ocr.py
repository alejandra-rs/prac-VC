import gc

import numpy as np
import torch
from paddleocr import TextRecognition

PADDLE_REC_MODEL = "PP-OCRv5_server_rec"


class OCRModel:
    def __init__(self, name):
        self.name = name
        self.model = None
        self.info = {}

    def prefetch(self):
        pass

    def load(self):
        pass

    def read(self, crop_bgr):
        pass

    def close(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class PaddleOCRModel(OCRModel):
    def __init__(self, model_name=PADDLE_REC_MODEL):
        super().__init__("paddleocr")
        self.model_name = model_name

    def prefetch(self):
        TextRecognition(model_name=self.model_name, device="cpu")

    def load(self):
        self.model = TextRecognition(model_name=self.model_name, device="gpu:0" if torch.cuda.is_available() else "cpu")

    def read(self, crop_bgr):
        return self.model.predict(input=np.ascontiguousarray(crop_bgr), batch_size=1)[0]["rec_text"]
