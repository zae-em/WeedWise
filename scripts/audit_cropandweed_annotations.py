"""Audit every source annotation without claiming crop usability before media verification."""
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'dataset/cropandweed'
    raw=folder/'huggingface/hub/Voxel51/CropAndWeed'
    source=json.loads((folder/'source_manifest.json').read_text())
    path=raw/'samples.json'
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    expected=next(r['sha256'] for r in source['files'] if r['path']=='samples.json')
    if digest!=expected:raise ValueError('Annotation file SHA256 mismatch')
    samples=json.loads(path.read_text())['samples']
    taxonomy=json.loads((folder/'taxonomy.json').read_text())['labels']
    counts=Counter();sessions=defaultdict(set);issues=[];all_sessions=set();species=defaultdict(lambda:{'instances':0,'sessions':set(),'sources':set(),'raw_ids':set()})
    empty=0
    for sample in samples:
        name=Path(sample['filepath']).stem
        pieces=name.split('-')
        session=pieces[0]+':'+sample['session']
        all_sessions.add(session)
        if len(pieces)!=3 or pieces[1]!=sample['session'] or pieces[2]!=sample['image_id']:
            issues.append({'source':name,'issue':'session/image metadata mismatch'})
        detections=(sample.get('ground_truth') or {}).get('detections',[])
        empty+=not bool(detections)
        for detection in detections:
            raw_id=str(detection['label_id'])
            counts[raw_id]+=1;sessions[raw_id].add(session)
            taxon=taxonomy.get(raw_id)
            if not taxon or detection['label']!=taxon['raw_label']:
                issues.append({'source':name,'issue':'numeric/string label mismatch','label':detection['label']})
            if taxon and taxon['included']:
                item=species[taxon['class_label']]
                item['scientific_name']=taxon['scientific_name'];item['role']=taxon['role']
                item['instances']+=1;item['sessions'].add(session);item['sources'].add(name);item['raw_ids'].add(raw_id)
    resolved={label:{**row,'sessions':len(row['sessions']),'sources':len(row['sources']),'raw_ids':sorted(row['raw_ids'],key=int)} for label,row in sorted(species.items())}
    report={'revision':source['revision'],'annotation_sha256':digest,'source_images':len(samples),
        'recording_sessions':len(all_sessions),'empty_detection_images':empty,'total_bounding_boxes':sum(counts.values()),
        'raw_label_counts':{k:{'name':taxonomy[k]['raw_label'],'instances':v,'sessions':len(sessions[k]),'taxonomy_eligible':taxonomy[k]['included']} for k,v in sorted(counts.items(),key=lambda r:int(r[0]))},
        'eligible_present_species_count':len(resolved),'eligible_present_species':resolved,'issues':issues,
        'status':'annotation_audited_only; crop usability and final split coverage remain pending'}
    (ROOT/'evaluation/cropandweed_annotation_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in {'raw_label_counts','eligible_present_species'}},indent=2))


if __name__=='__main__':main()
