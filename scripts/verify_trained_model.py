"""Verify the real trained export against the full local held-out split and GPU results."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from weed_assistant.inference import PlantClassifier
from training.data import discover,FolderImages,image_transforms
from training.evaluation import collect_logits,calibration_error


def main():
    classifier=PlantClassifier()
    classes,splits,_,fingerprint=discover(ROOT/"dataset/processed")
    assert classes==classifier.classes,"Model class mapping differs from the actual dataset"
    assert fingerprint==classifier.artifact["fingerprint"],"Model was trained on a different split or image collection"
    assert classifier.artifact["calibration"]["fit_split"]=="validation"
    assert classifier.artifact["calibration"]["validation_samples"]==len(splits["val"])
    history=json.loads((ROOT/"models/mobilenet_v2/history.json").read_text())
    assert {row["stage"] for row in history}=={"head","finetune"},"Both real training phases must have run"
    best=torch.load(ROOT/"models/mobilenet_v2/best.pt",map_location="cpu",weights_only=True)
    assert all(torch.equal(value,best["model"][name]) for name,value in classifier.artifact["model"].items()),"Export is not the best selected model"
    dataset=FolderImages(splits["test"],image_transforms(224,False))
    loader=DataLoader(dataset,batch_size=16,shuffle=False,num_workers=0)
    start=time.perf_counter()
    logits,labels=collect_logits(classifier.model,loader,torch.device("cpu"))
    probabilities=torch.softmax(logits/classifier.temperature,1).numpy()
    duration=time.perf_counter()-start
    saved=np.load(ROOT/"evaluation/mobilenet_v2/test_predictions.npz")
    np.testing.assert_array_equal(labels.numpy(),saved["labels"])
    maximum_difference=float(np.abs(probabilities-saved["probabilities"]).max())
    assert maximum_difference<0.005,"Local model outputs differ materially from measured GPU outputs"
    predicted=probabilities.argmax(1);truth=labels.numpy()
    top5=np.argsort(-probabilities,axis=1)[:,:5]
    metrics=json.loads((ROOT/"evaluation/mobilenet_v2/test_metrics.json").read_text())
    accuracy=float((predicted==truth).mean())
    assert abs(accuracy-metrics["top1"])<=1/len(truth),"Local test accuracy differs materially"
    sample=dataset[len(dataset)//2][0].unsqueeze(0)
    with torch.inference_mode():
        for _ in range(3):classifier.model(sample)
        timings=[]
        for _ in range(20):
            start=time.perf_counter();classifier.model(sample);timings.append((time.perf_counter()-start)*1000)
    review=probabilities.max(1)<classifier.threshold
    ece,_=calibration_error(probabilities,truth)
    result={"status":"PASS","real_trained_model":True,"cpu_training_performed":False,"model_sha256":classifier.model_hash,
            "classes":classes,"test_samples":len(truth),"full_test_cpu_inference_seconds":duration,"local_top1":accuracy,
            "local_top5":float((top5==truth[:,None]).any(1).mean()),"ece":ece,"max_probability_difference_from_gpu":maximum_difference,
            "median_cpu_model_inference_ms":float(np.median(timings)),"p95_cpu_model_inference_ms":float(np.quantile(timings,.95)),
            "known_test_images_flagged_by_confidence":int(review.sum()),"known_test_review_fraction":float(review.mean()),
            "unknown_species_detection_validated":False,"external_field_validation":False,"training_epochs_completed":len(history),"best_epoch":int(best["epoch"])+1}
    (ROOT/"evaluation/local_model_verification.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
