"""Real, CPU model inference. Missing or untrained artifacts are never replaced by predictions."""
import hashlib
import time
import threading
from pathlib import Path
import numpy as np
import torch
from torchvision.models import mobilenet_v2
from torch import nn

from training.data import image_transforms
from training.evaluation import gradcam
from .config import MODEL_PATH,CROP_CLASSES


class ModelUnavailable(RuntimeError):pass


class PlantClassifier:
    def __init__(self,path=MODEL_PATH):
        self.path=Path(path)
        if not self.path.is_file():raise ModelUnavailable("The trained model has not been retrieved yet. Finish the Colab job and import its exported model.")
        artifact=torch.load(self.path,map_location="cpu",weights_only=True)
        if artifact.get("architecture")!="mobilenet_v2" or artifact.get("trained") is not True:raise ModelUnavailable("This file is not a completed MobileNetV2 training artifact.")
        self.classes=artifact.get("classes",[])
        if len(self.classes)<2 or len(set(self.classes))!=len(self.classes):raise ModelUnavailable("The artifact has an invalid class mapping.")
        self.model_version=artifact.get("model_version","v1")
        if self.model_version=="v2" and "class_roles" not in artifact:
            raise ModelUnavailable("Expanded models require an explicit crop/weed mapping.")
        self.class_roles=artifact.get("class_roles",{name:"crop" if name in CROP_CLASSES else "weed" for name in self.classes})
        if set(self.class_roles)!=set(self.classes) or any(role not in {"crop","weed"} for role in self.class_roles.values()):
            raise ModelUnavailable("The artifact has an invalid crop/weed mapping.")
        self.model=mobilenet_v2(weights=None)
        self.model.classifier[1]=nn.Linear(self.model.last_channel,len(self.classes))
        self.model.load_state_dict(artifact["model"],strict=True)
        self.model.eval()
        self.transform=image_transforms(artifact["image_size"],False)
        self.temperature=float(artifact.get("temperature",1.0))
        self.threshold=float(artifact.get("unknown_threshold",0.5))
        if not 0<self.temperature<100 or not 0<=self.threshold<=1:raise ModelUnavailable("Invalid calibration values")
        self.calibrated=bool(artifact.get("calibration",{}).get("fitted",False))
        with self.path.open("rb") as stream:self.model_hash=hashlib.file_digest(stream,"sha256").hexdigest()
        self.artifact=artifact
        self.lock=threading.RLock()
        torch.set_num_threads(2)

    def predict(self,image,quality=None):
        tensor=self.transform(image).unsqueeze(0)
        with self.lock:
            start=time.perf_counter()
            with torch.inference_mode():probabilities=torch.softmax(self.model(tensor)/self.temperature,dim=1)[0].numpy()
            elapsed=(time.perf_counter()-start)*1000
        order=np.argsort(-probabilities)[:min(5,len(self.classes))]
        if not np.isfinite(probabilities).all():raise ModelUnavailable("Model returned non-finite confidence scores.")
        predictions=[{"class_label":self.classes[index],"confidence":float(probabilities[index]),"role":self.class_roles[self.classes[index]]} for index in order]
        below=predictions[0]["confidence"]<self.threshold
        needs_review=below or bool(quality and quality.get("needs_review"))
        crop_probability=float(np.clip(sum(probabilities[index] for index,label in enumerate(self.classes) if self.class_roles[label]=="crop"),0,1))
        return {"predictions":predictions,"needs_review":needs_review,"below_confidence_threshold":below,"unknown_or_uncertain":below,
                "status":"Needs review" if needs_review else "Suggested identification","confidence_threshold":self.threshold,"calibrated":self.calibrated,
                "crop_probability":crop_probability,"weed_probability":1-crop_probability,"crop_weed_label":"crop" if crop_probability>=.5 else "weed",
                "inference_ms":elapsed,"model_sha256":self.model_hash,"model_version":self.model_version,
                "limitation":"An unsupported species can receive high confidence. These scores are not proof of identity or field safety."}

    def explain(self,image,target=None):
        tensor=self.transform(image).unsqueeze(0)
        with self.lock:heat,index=gradcam(self.model,tensor,target)
        import cv2
        original=(tensor[0].permute(1,2,0).numpy()*np.asarray([.229,.224,.225])+np.asarray([.485,.456,.406]))
        original=np.clip(original*255,0,255).astype(np.uint8)
        colour=cv2.cvtColor(cv2.applyColorMap(np.uint8(255*heat),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
        overlay=np.uint8(.60*original+.40*colour)
        return overlay,self.classes[index]
