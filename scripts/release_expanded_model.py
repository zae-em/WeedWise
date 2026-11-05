"""Verify genuine downloaded v2 predictions locally, build cited records and activate.

Activation requires a successful full held-out inference comparison. It never copies
v2 over v1, and the database is backed up before the app sees an expanded mapping.
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def build_knowledge(classes):
    payload=json.loads((ROOT/'knowledge_base/species.json').read_text())
    taxonomy=json.loads((ROOT/'dataset/cropandweed/taxonomy.json').read_text())['labels']
    by_class={r['class_label']:r for r in payload['species']}
    by_taxon={r['class_label']:r for r in taxonomy.values() if r['included']}
    accessed=datetime.now(timezone.utc).date().isoformat()
    source='cropandweed_labels'
    payload['sources'].append({'id':source,'title':'CropAndWeed authors: original labels and crop/weed mappings',
        'url':'https://github.com/cropandweed/cropandweed-dataset/blob/main/cnw/utilities/datasets.py',
        'context':'Austrian research dataset labels; crop/weed roles follow the authors simplified mapping','accessed':accessed})
    for label in classes:
        if label in by_class:continue
        item=by_taxon[label]
        sid='gbif_'+str(item['gbif_key'])
        if not any(s['id']==sid for s in payload['sources']):
            payload['sources'].append({'id':sid,'title':'GBIF taxonomic record: '+item['scientific_name'],
                'url':f"https://www.gbif.org/species/{item['gbif_key']}",
                'context':'Accepted scientific name, common-name association and botanical family; saved API evidence under dataset/cropandweed/taxonomy','accessed':accessed})
        by_class[label]={'id':'gbif-'+str(item['gbif_key']),'class_label':label,'common_name':item['common_name'],
            'scientific_name':item['scientific_name'],'family':item['family'],'role':item['role'],
            'plant_type':'grass' if item['family']=='Poaceae' else 'Unreviewed',
            'life_cycle':'Not source-verified for this record','tier':'B','aliases':[item['base_label']],
            'taxonomy_source_ids':[sid,source],'pakistan_relevance':'Not independently field-validated',
            'expert_review_status':'Taxonomy source checked; species-specific management and local agronomic review pending',
            'claims':[{'category':'taxonomy','text':f"{item['common_name']} is recorded as {item['scientific_name']} in the family {item['family']}.",
                       'source_ids':[sid],'verification':'GBIF scientific and vernacular-name evidence saved'},
                      {'category':'classification','text':f"The authors simplified dataset mapping treats this label as {item['role']}. Crop/weed roles depend on field context.",
                       'source_ids':[source],'verification':'Author label mapping'}]}
    payload['species']=[by_class[c] for c in classes]
    payload['scope']='Trained v2 classes only. New taxonomy is source-checked; unsupported species-specific management is withheld.'
    target=ROOT/'knowledge_base/species_v2.json'
    target.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    from weed_assistant.knowledge import KnowledgeBase
    KnowledgeBase(target)
    return target


def verify():
    import time
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from training.data import FolderImages,image_transforms
    from training.evaluation import collect_logits
    from training.expanded_train import load_manifest
    from weed_assistant.inference import PlantClassifier
    model_path=ROOT/'models/v2/model.pt'
    classifier=PlantClassifier(model_path)
    manifest,splits=load_manifest(ROOT/'dataset/combined_v2')
    if len(classifier.classes)<=len(manifest['original_classes']):
        raise ValueError('This artifact has no new supported species')
    saved_status=json.loads((ROOT/'models/v2/training_status.json').read_text())
    if saved_status['state']!='complete' or not saved_status['epochs']:
        raise RuntimeError('Training is not complete')
    if classifier.classes!=manifest['classes'] or classifier.artifact['fingerprint']!=manifest['fingerprint']:
        raise ValueError('Model and dataset mapping/fingerprint mismatch')
    if not (ROOT/'evaluation/v2/import_integrity.json').is_file():
        raise ValueError('Downloaded export has not passed full transfer integrity verification')
    loader=DataLoader(FolderImages(splits['test'],image_transforms(classifier.artifact['image_size'],False)),batch_size=32,num_workers=0)
    started=time.perf_counter()
    logits,labels=collect_logits(classifier.model,loader,torch.device('cpu'))
    probabilities=torch.softmax(logits/classifier.temperature,dim=1).numpy()
    with np.load(ROOT/'evaluation/v2/test_predictions.npz') as remote:
        if not np.array_equal(labels.numpy(),remote['labels']):raise ValueError('Remote test-label order differs')
        difference=float(np.max(np.abs(probabilities-remote['probabilities'])))
    if difference>1e-3:raise ValueError(f'Local inference differs materially from remote: {difference}')
    actual=labels.numpy()
    top1=float((probabilities.argmax(1)==actual).mean())
    top5=float((np.argsort(-probabilities,axis=1)[:,:5]==actual[:,None]).any(1).mean())
    metrics=json.loads((ROOT/'evaluation/v2/test_metrics.json').read_text())
    if abs(top1-metrics['top1'])>1/len(actual) or abs(top5-metrics['top5'])>1/len(actual):raise ValueError('Saved metrics differ from local inference')
    sample=loader.dataset[0][0].unsqueeze(0)
    timings=[]
    for _ in range(20):
        tick=time.perf_counter()
        with torch.inference_mode():classifier.model(sample)
        timings.append((time.perf_counter()-tick)*1000)
    original_count=sum(r['origin']=='original' and r['split']=='test' for r in manifest['records'])
    if original_count!=826:raise ValueError('The original 826-image test set changed')
    frozen=json.loads((ROOT/'dataset/processed/manifest.json').read_text())['records']
    expected={(r['processed_sha256'],r['class_name'],r['split']) for r in frozen}
    actual_original={(r['sha256'],r['class_name'],r['split']) for r in manifest['records'] if r['origin']=='original'}
    if actual_original!=expected:raise ValueError('Original images, labels, or split assignments changed')
    result={'state':'verified','model_sha256':classifier.model_hash,'classes':classifier.classes,
        'test_samples':len(actual),'original_test_samples':original_count,'local_top1':top1,'local_top5':top5,
        'max_probability_difference_from_gpu':difference,'cpu_inference_seconds':time.perf_counter()-started,
        'median_cpu_model_inference_ms':float(np.median(timings)),'p95_cpu_model_inference_ms':float(np.quantile(timings,.95))}
    (ROOT/'evaluation/v2/local_verification.json').write_text(json.dumps(result,indent=2))
    build_knowledge(classifier.classes)
    release={'version':'v2','model_sha256':classifier.model_hash,'class_roles':classifier.class_roles,
        'class_count':len(classifier.classes),'dataset_fingerprint':manifest['fingerprint'],
        'verified_utc':datetime.now(timezone.utc).isoformat()}
    (ROOT/'models/v2/release.json').write_text(json.dumps(release,indent=2))
    return release


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--activate',action='store_true')
    args=parser.parse_args();release=verify()
    if args.activate:
        database=ROOT/'app/data/weed_assistant.sqlite3'
        backup=ROOT/'models/v1/database_before_v2.sqlite3'
        if database.exists() and not backup.exists():
            with sqlite3.connect(database) as source,sqlite3.connect(backup) as target:source.backup(target)
        target=ROOT/'models/active.json';temporary=target.with_suffix('.pending')
        temporary.write_text(json.dumps(release,indent=2));temporary.replace(target)
        print('Activated locally verified v2. Restart Streamlit to load its mapping.')
    else:print('Local v2 verification passed. Release manifest prepared; v1 remains active.')
