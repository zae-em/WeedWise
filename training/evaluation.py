"""Measured held-out evaluation, validation-only calibration, and explainability."""
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay

from .data import NORMALIZE_MEAN, NORMALIZE_STD


def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits.append(model(images.to(device)).float().cpu())
            labels.append(targets.cpu())
    return torch.cat(logits), torch.cat(labels)


def calibration_error(probabilities, labels, bins=15):
    confidences = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    table=[]
    error=0.0
    for index in range(bins):
        low,high=index/bins,(index+1)/bins
        mask=(confidences>=low)&((confidences<=high) if index==bins-1 else (confidences<high))
        count=int(mask.sum())
        accuracy=float(correct[mask].mean()) if count else 0.0
        confidence=float(confidences[mask].mean()) if count else 0.0
        error+=count/len(labels)*abs(accuracy-confidence)
        table.append({"low":low,"high":high,"count":count,"accuracy":accuracy,"confidence":confidence})
    return float(error),table


def fit_calibration(logits, labels):
    def loss(log_temperature):
        return float(torch.nn.functional.cross_entropy(logits/float(np.exp(log_temperature)),labels).item())
    fit=minimize_scalar(loss,bounds=(-2.3,2.3),method="bounded",options={"xatol":1e-4})
    temperature=float(np.exp(fit.x)) if fit.success else 1.0
    probabilities=torch.softmax(logits/temperature,dim=1).numpy()
    correct=probabilities.argmax(1)==labels.numpy()
    threshold=float(np.quantile(probabilities.max(1)[correct],0.05)) if correct.any() else 1.0
    return {"temperature":temperature,"unknown_threshold":threshold,"fitted":bool(fit.success),"fit_split":"validation",
            "threshold_method":"5th percentile of confidence on correctly classified validation examples",
            "validation_samples":len(labels),"validation_nll_before":loss(0.0),"validation_nll_after":loss(np.log(temperature)),
            "unknown_species_validation":"Unavailable: threshold is an uncertainty/review gate, not a proven open-set detector"}


def gradcam(model, tensor, target=None):
    model.eval()
    captured={}
    def hook(module, inputs, output):
        captured["activation"]=output
        output.register_hook(lambda gradient:captured.update(gradient=gradient))
    handle=model.features[-1].register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits=model(tensor.detach().requires_grad_(True))
        index=int(logits.argmax(1)[0]) if target is None else int(target)
        logits[0,index].backward()
        weights=captured["gradient"].mean(dim=(2,3),keepdim=True)
        heat=torch.relu((weights*captured["activation"]).sum(1,keepdim=True))
        heat=torch.nn.functional.interpolate(heat,size=tensor.shape[-2:],mode="bilinear",align_corners=False)[0,0]
        heat=heat/(heat.max()+1e-8)
        return heat.detach().cpu().numpy(),index
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def evaluate_complete(model,test_loader,val_loader,device,classes,destination,history,class_roles=None,original_classes=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    val_logits,val_labels=collect_logits(model,val_loader,device)
    calibration=fit_calibration(val_logits,val_labels)
    logits,labels=collect_logits(model,test_loader,device)
    probabilities=torch.softmax(logits/calibration["temperature"],dim=1).numpy()
    uncalibrated=torch.softmax(logits,dim=1).numpy()
    actual=labels.numpy();predicted=probabilities.argmax(1);confidence=probabilities.max(1)
    k=min(5,len(classes))
    topk=np.argsort(-probabilities,axis=1)[:,:k]
    ece,bins=calibration_error(probabilities,actual)
    ece_before,_=calibration_error(uncalibrated,actual)
    precision,recall,macro_f1,_=precision_recall_fscore_support(actual,predicted,average="macro",zero_division=0)
    _,_,weighted_f1,_=precision_recall_fscore_support(actual,predicted,average="weighted",zero_division=0)
    accepted=confidence>=calibration["unknown_threshold"]
    metrics={"test_samples":len(labels),"top1":float((actual==predicted).mean()),"top5":float((topk==actual[:,None]).any(1).mean()),
             "top5_effective_k":k,"precision_macro":float(precision),"recall_macro":float(recall),"macro_f1":float(macro_f1),"weighted_f1":float(weighted_f1),
             "loss":float(torch.nn.functional.cross_entropy(logits/calibration["temperature"],labels)),"ece_before":ece_before,"ece_after":ece,
             "known_class_acceptance_rate":float(accepted.mean()),"accuracy_on_accepted":float((actual[accepted]==predicted[accepted]).mean()) if accepted.any() else None,
             "training_accuracy_last_epoch":history[-1]["train"]["top1"] if history else None,"validation_accuracy_last_epoch":history[-1]["val"]["top1"] if history else None,
             "external_test_set":False,"unknown_species_test_set":False}
    report=classification_report(actual,predicted,labels=list(range(len(classes))),target_names=classes,output_dict=True,zero_division=0)
    matrix=confusion_matrix(actual,predicted,labels=list(range(len(classes))))
    crop_indices=[i for i,name in enumerate(classes) if (class_roles or {}).get(name)=="crop" or (class_roles is None and name in {"Common wheat","Maize","Sugar beet"})]
    if crop_indices:
        true_crop=np.isin(actual,crop_indices)
        predicted_crop=probabilities[:,crop_indices].sum(1)>=0.5
        metrics["crop_weed_accuracy_supported_classes"]=float((true_crop==predicted_crop).mean())
    sample=next(iter(test_loader))[0][:1].to(device)
    with torch.inference_mode():
        for _ in range(3):model(sample)
        timings=[]
        for _ in range(20):
            if device.type=="cuda":torch.cuda.synchronize()
            start=time.perf_counter();model(sample)
            if device.type=="cuda":torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
    metrics["model_inference_median_ms"]=float(np.median(timings));metrics["model_inference_p95_ms"]=float(np.quantile(timings,.95));metrics["inference_device"]=str(device)
    if original_classes is not None:
        original_indices=[classes.index(name) for name in original_classes]
        original_mask=np.isin(actual,original_indices)
        rows=getattr(test_loader.dataset,"rows",[])
        original_test=np.array([row.get("origin")=="original" for row in rows],dtype=bool)
        groups={"original_classes":original_mask,"new_classes":~original_mask,"combined":np.ones(len(actual),dtype=bool)}
        if len(original_test)==len(actual):groups["original_heldout"]=original_test
        metrics["groups"]={}
        for name,mask in groups.items():
            if not mask.any():
                metrics["groups"][name]={"samples":0,"top1":None,"top5":None}
                continue
            group_report=classification_report(actual[mask],predicted[mask],labels=sorted(set(actual[mask])),output_dict=True,zero_division=0)
            metrics["groups"][name]={"samples":int(mask.sum()),"top1":float((actual[mask]==predicted[mask]).mean()),
                "top5":float((topk[mask]==actual[mask,None]).any(1).mean()),
                "precision_macro":group_report["macro avg"]["precision"],"recall_macro":group_report["macro avg"]["recall"],
                "macro_f1":group_report["macro avg"]["f1-score"],"weighted_f1":group_report["weighted avg"]["f1-score"]}
        if "original_heldout" in metrics["groups"]:
            metrics["original_heldout_top1_delta_from_v1"]=metrics["groups"]["original_heldout"]["top1"]-0.9309927360774818
    for name,value in [("test_metrics.json",metrics),("classification_report.json",report),("calibration.json",calibration),("class_names.json",classes),("history.json",history)]:
        (destination/name).write_text(json.dumps(value,indent=2),encoding="utf-8")
    np.savetxt(destination/"confusion_matrix.csv",matrix,fmt="%d",delimiter=",")
    np.savez_compressed(destination/"test_predictions.npz",probabilities=probabilities,labels=actual)
    rows=getattr(test_loader.dataset,"rows",[])
    with (destination/"test_predictions.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream);writer.writerow(["image","actual","predicted","confidence","needs_review"])
        for i in range(len(actual)):
            writer.writerow([str(rows[i]["path"]) if rows else str(i),classes[actual[i]],classes[predicted[i]],float(confidence[i]),not bool(accepted[i])])
    side=min(30,max(12,len(classes)*.30))
    fig,ax=plt.subplots(figsize=(side,side));ConfusionMatrixDisplay(matrix,display_labels=classes).plot(ax=ax,xticks_rotation=65,colorbar=False,include_values=len(classes)<=25);fig.tight_layout();fig.savefig(destination/"confusion_matrix.png",dpi=150);plt.close(fig)
    if history:
        fig,axes=plt.subplots(1,2,figsize=(13,4))
        epochs=[row["epoch"]+1 for row in history]
        for split in ["train","val"]:
            axes[0].plot(epochs,[row[split]["loss"] for row in history],label=split)
            axes[1].plot(epochs,[row[split]["top1"] for row in history],label=split)
        axes[0].set_title("Loss");axes[1].set_title("Top-1 accuracy")
        for ax in axes:ax.set_xlabel("Epoch");ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(destination/"training_curves.png",dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].plot([0,1],[0,1],"--",color="gray")
    occupied=[b for b in bins if b["count"]]
    axes[0].plot([b["confidence"] for b in occupied],[b["accuracy"] for b in occupied],"o-")
    axes[0].set(xlabel="Confidence",ylabel="Observed accuracy",title=f"Calibrated reliability / ECE {ece:.4f}")
    axes[1].hist(confidence,bins=20,color="#2c7a66");axes[1].axvline(calibration["unknown_threshold"],color="#d78242",label="review threshold");axes[1].set_title("Held-out confidence");axes[1].legend()
    fig.tight_layout();fig.savefig(destination/"calibration.png",dpi=150);plt.close(fig)
    chosen=list(dict.fromkeys([int(np.argmin(confidence)),int(np.argmax(confidence)),int(np.flatnonzero(actual!=predicted)[0]) if np.any(actual!=predicted) else 0]))
    for index in chosen:
        tensor,_=test_loader.dataset[index];tensor=tensor.unsqueeze(0).to(device)
        heat,label=gradcam(model,tensor)
        original=tensor[0].detach().cpu().permute(1,2,0).numpy()*np.array(NORMALIZE_STD)+np.array(NORMALIZE_MEAN)
        fig,axes=plt.subplots(1,2,figsize=(7,4));axes[0].imshow(np.clip(original,0,1));axes[1].imshow(np.clip(original,0,1));axes[1].imshow(heat,alpha=.45,cmap="jet")
        axes[0].set_title(f"True: {classes[actual[index]]}");axes[1].set_title(f"Predicted: {classes[label]}")
        for ax in axes:ax.axis("off")
        fig.tight_layout();fig.savefig(destination/f"gradcam_{index}.png",dpi=140);plt.close(fig)
    lines=["# Measured model evaluation","",f"Held-out test images: {len(labels)}; top-1: {metrics['top1']:.4%}; top-5: {metrics['top5']:.4%}.",
           f"Macro precision: {precision:.4f}; macro recall: {recall:.4f}; macro F1: {macro_f1:.4f}; weighted F1: {weighted_f1:.4f}.",
           f"ECE before temperature scaling: {ece_before:.5f}; after: {ece:.5f}. Temperature {calibration['temperature']:.5f} was fitted using validation images only.",
           f"Review threshold: {calibration['unknown_threshold']:.5f}; known-class test acceptance: {accepted.mean():.2%}.",
           f"Single-image model inference median: {metrics['model_inference_median_ms']:.2f} ms on {device}; excludes application upload and preprocessing.","",
           "## Interpretation and limits","",
           "The test split was held out from model fitting, early stopping and calibration. Duplicate groups do not cross splits; missing original plant/photographer identifiers prevent ruling out all acquisition-level dependence.",
           "No independent external field dataset or labeled unknown-species set was available. The review threshold measures confidence on supported classes and is not a validated unknown-plant detector. A confidently wrong unsupported plant remains possible.",
           f"Species coverage is {len(classes)} dataset classes. Pakistani field performance, adult plants, multiple plants per image, herbicide response and field safety have not been established.",
           "Grad-CAM shows regions influencing a class score; it does not establish botanical causation or correctness. Class metrics and prediction CSVs are included for error review.",""]
    (destination/"EVALUATION_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    return metrics
