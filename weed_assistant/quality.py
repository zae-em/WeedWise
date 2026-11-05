"""Image decoding and advisory quality checks; no botanical diagnosis."""
import io
import warnings
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from .config import MAX_IMAGE_BYTES


def decode_image(data):
    if not data or len(data)>MAX_IMAGE_BYTES:
        raise ValueError("Choose a non-empty image smaller than 12 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error",Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.width*image.height>40_000_000:
                    raise ValueError("Image exceeds the 40 megapixel limit; resize it first.")
                image=ImageOps.exif_transpose(image).convert("RGB")
                image.load()
                return image.copy()
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError,Image.DecompressionBombWarning) as error:
        raise ValueError("This file could not be decoded as a supported photograph.") from error


def assess_quality(image):
    rgb=np.asarray(image.convert("RGB"))
    preview=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)
    gray=cv2.cvtColor(preview,cv2.COLOR_RGB2GRAY)
    blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
    brightness=float(gray.mean())
    hsv=cv2.cvtColor(preview,cv2.COLOR_RGB2HSV)
    green=float(((hsv[:,:,0]>=25)&(hsv[:,:,0]<=95)&(hsv[:,:,1]>35)&(hsv[:,:,2]>35)).mean())
    warnings_list=[]
    if min(image.size)<96:warnings_list.append("Low image resolution; use a closer, sharper photograph.")
    if blur<25:warnings_list.append("The image may be blurred or lack detail.")
    if brightness<35:warnings_list.append("The photograph is very dark.")
    if brightness>225:warnings_list.append("The photograph may be overexposed.")
    if green<0.01:warnings_list.append("Little green plant area is visible. This check can also flag brown or dormant plants.")
    return {"width":image.width,"height":image.height,"blur_score":blur,"brightness":brightness,"green_fraction":green,
            "warnings":warnings_list,"needs_review":bool(warnings_list),"usable":min(image.size)>=32}
