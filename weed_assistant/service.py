"""Application workflow: validate image, infer, save locally and attach sourced species information."""
import hashlib
from pathlib import Path
from .config import UPLOADS_PATH
from .quality import decode_image,assess_quality


class IdentificationService:
    def __init__(self,classifier,database,kb,uploads=UPLOADS_PATH):
        self.classifier,self.database,self.kb=classifier,database,kb
        self.uploads=Path(uploads);self.uploads.mkdir(parents=True,exist_ok=True)

    def identify(self,data,save=True,crop_context=None):
        image=decode_image(data);quality=assess_quality(image)
        if not quality["usable"]:raise ValueError("The image is too small. Use a photograph at least 32 pixels in each dimension.")
        result=self.classifier.predict(image,quality)
        if crop_context:
            result["crop_context"]=crop_context
            result["crop_context_applied_to_ranking"]=False
        digest=hashlib.sha256(data).hexdigest()
        species=self.kb.get(result["predictions"][0]["class_label"])
        identification={"id":None,"image":image,"image_sha256":digest,"quality":quality,"result":result,"species":species}
        if save:self.save(identification,retain_image=True)
        return identification

    def save(self,identification,retain_image=False):
        """Persist once on an explicit user action; retaining the image is optional."""
        if identification.get("id") and self.database.get_prediction(identification["id"]):return identification["id"]
        destination=None
        if retain_image:
            destination=self.uploads/(identification["image_sha256"]+".jpg")
            if not destination.exists():identification["image"].save(destination,"JPEG",quality=95)
        identifier=self.database.save_prediction(str(destination) if destination else "",identification["image_sha256"],
                    identification["result"],identification["quality"],self.classifier.model_hash)
        identification["id"]=identifier
        return identifier
