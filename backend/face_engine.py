import os
# Quiet down ONNX Runtime & InsightFace logs
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "3")   # 0-4 (0=verbose, 3=error)
os.environ.setdefault("ORT_LOG_VERBOSITY_LEVEL", "1")
os.environ.setdefault("INSIGHTFACE_LOG_LEVEL", "ERROR")

import logging
logging.getLogger("insightface").setLevel(logging.ERROR)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

import cv2
import numpy as np
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, det_size=(480, 480)):
        """
        Lighter, faster defaults:
        - Smaller det_size (480x480) vs 640x640
        - Only load detection + recognition modules
        """
        try:
            self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection","recognition"])
        except TypeError:
            self.app = FaceAnalysis(name="buffalo_l")
        self.app.prepare(ctx_id=0, det_size=det_size)

    def extract_faces(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = self.app.get(img)
        out = []
        for f in faces:
            bbox = f.bbox.astype(int).tolist()
            emb = f.normed_embedding.astype(np.float32)
            out.append({"bbox": bbox, "embedding": emb})
        return out
