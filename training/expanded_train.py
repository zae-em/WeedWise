"""GPU-only v1-to-v2 fine-tuning with full coverage and bounded minority replay."""
import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from .data import FolderImages, image_transforms, NORMALIZE_MEAN, NORMALIZE_STD
from .evaluation import evaluate_complete
from .train import atomic_save, build_model, rng_state, restore_rng, run_epoch


@dataclass
class ExpansionConfig:
    data_dir: str = 'dataset/combined_v2'
    initial_checkpoint: str = 'models/v1/models/mobilenet_v2/model.pt'
    output_dir: str = 'models/v2'
    evaluation_dir: str = 'evaluation/v2'
    batch_size: int = 64
    image_size: int = 224
    head_epochs: int = 3
    finetune_epochs: int | None = None
    min_epochs: int | None = None
    patience: int = 7
    seed: int = 42
    head_lr: float = 0.0005
    finetune_lr: float = 0.00005
    weight_decay: float = 0.0001
    num_workers: int = 2
    finetune_from: int = 10
    distillation_weight: float = 1.0
    resume: str | None = None

    def with_epoch_limits(self, checkpoint_config=None):
        finetune = self.finetune_epochs
        if finetune is None:
            finetune = 22 if checkpoint_config is None else checkpoint_config['finetune_epochs']
        minimum = self.min_epochs
        total = self.head_epochs + finetune
        if minimum is None:
            minimum = min(20, total) if checkpoint_config is None else checkpoint_config.get('min_epochs', 0)
        if self.head_epochs < 0 or finetune < 0 or total < 1 or not 0 <= minimum <= total:
            raise ValueError('Invalid expansion epoch limits')
        return replace(self, finetune_epochs=finetune, min_epochs=minimum)


class BalancedCoverageSampler(Sampler):
    """Every training image occurs each epoch; replay underrepresented classes.

    WeightedRandomSampler with replacement does not guarantee full coverage.
    Replay is bounded to three times each class's original image count.
    """
    def __init__(self, labels, generator):
        self.generator = generator
        self.labels = torch.as_tensor(labels)
        self.counts = torch.bincount(self.labels)
        if torch.any(self.counts == 0):
            raise ValueError('Every class must have training examples')
        target = int(torch.median(self.counts.float()))
        self.extras = {i: max(0, min(target, 3*int(n))-int(n)) for i,n in enumerate(self.counts)}

    def __len__(self):
        return len(self.labels) + sum(self.extras.values())

    def __iter__(self):
        indices = list(range(len(self.labels)))
        for label, extra in self.extras.items():
            members = torch.where(self.labels == label)[0]
            choices = torch.randint(len(members), (extra,), generator=self.generator)
            indices.extend(members[choices].tolist())
        order = torch.randperm(len(indices), generator=self.generator).tolist()
        return iter(indices[i] for i in order)


def expand_model(artifact, classes):
    old_classes = artifact['classes']
    if artifact.get('architecture') != 'mobilenet_v2' or not artifact.get('trained'):
        raise ValueError('Initial checkpoint must be a completed trained MobileNetV2 export')
    if not set(old_classes).issubset(classes):
        raise ValueError('Expanded mapping must preserve every original species')
    old = build_model(len(old_classes), False)
    old.load_state_dict(artifact['model'], strict=True)
    model = build_model(len(classes), False)
    state = {k:v for k,v in artifact['model'].items() if not k.startswith('classifier.1.')}
    incompatible = model.load_state_dict(state, strict=False)
    assert set(incompatible.missing_keys) == {'classifier.1.weight', 'classifier.1.bias'}
    with torch.no_grad():
        for index, name in enumerate(old_classes):
            destination = classes.index(name)
            model.classifier[1].weight[destination].copy_(old.classifier[1].weight[index])
            model.classifier[1].bias[destination].copy_(old.classifier[1].bias[index])
    return model, old


def load_manifest(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder/'manifest.json').read_text())
    integrity=dict(manifest);fingerprint=integrity.pop('fingerprint')
    if hashlib.sha256(json.dumps(integrity,sort_keys=True).encode()).hexdigest()!=fingerprint:
        raise ValueError('Combined manifest fingerprint is invalid')
    classes = manifest['classes']
    if len(classes) != len(set(classes)) or not set(manifest['original_classes']).issubset(classes):
        raise ValueError('Invalid class mapping')
    splits = {s: [] for s in ['train','val','test']}
    groups, exact = {}, {}
    for record in manifest['records']:
        path = (folder/record['path']).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f'Missing or unsafe image: {path}')
        with path.open('rb') as source:
            if hashlib.file_digest(source,'sha256').hexdigest()!=record['sha256']:
                raise ValueError(f'Image content differs from manifest: {path}')
        if classes[record['label']] != record['class_name']:
            raise ValueError('Label index differs from mapping')
        for identifier in [('session',record['session_id']),('source',record['source_image']),('group',record['group_id'])]:
            prior = groups.setdefault(identifier, record['split'])
            if prior != record['split']:
                raise ValueError(f'Acquisition group crosses splits: {identifier}')
        prior = exact.setdefault(record['sha256'], (record['split'],record['label']))
        if prior != (record['split'],record['label']):
            raise ValueError('Exact duplicate crosses splits or labels')
        splits[record['split']].append({**record, 'path': path})
    for split, rows in splits.items():
        if set(r['label'] for r in rows) != set(range(len(classes))):
            raise ValueError(f'{split} does not cover every class')
    return manifest, splits


def train_epoch(model, teacher, loader, device, optimizer, scaler, old_indices, cfg, frozen):
    model.train()
    teacher.eval()
    for block in model.features.children():
        if frozen or not any(p.requires_grad for p in block.parameters()):
            block.eval()
    loss_total = correct = total = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda'):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            original = torch.isin(targets, old_indices)
            if original.any() and cfg.distillation_weight > 0:
                with torch.no_grad():
                    soft = torch.softmax(teacher(images[original])/2, dim=1)
                distill = nn.functional.kl_div(torch.log_softmax(logits[original][:,old_indices]/2,dim=1), soft, reduction='batchmean') * 4
                loss = loss + cfg.distillation_weight*distill
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite training loss')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        scaler.step(optimizer)
        scaler.update()
        total += len(targets)
        correct += int((logits.argmax(1)==targets).sum())
        loss_total += float(loss.detach())*len(targets)
    return {'loss': loss_total/total, 'top1': correct/total, 'samples_including_replay': total}


def train(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required. Expanded training never falls back to CPU.')
    recovery = None
    checkpoint_config = None
    if cfg.resume:
        recovery = torch.load(cfg.resume, map_location='cpu', weights_only=True)
        checkpoint_config = {'min_epochs': 0, **recovery['critical_config']}
    cfg = cfg.with_epoch_limits(checkpoint_config)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda')
    manifest, splits = load_manifest(cfg.data_dir)
    classes, originals = manifest['classes'], manifest['original_classes']
    initial = torch.load(cfg.initial_checkpoint, map_location='cpu', weights_only=True)
    with Path(cfg.initial_checkpoint).open('rb') as source:
        initial_digest=hashlib.file_digest(source,'sha256').hexdigest()
    model, teacher = expand_model(initial, classes)
    model, teacher = model.to(device), teacher.to(device)
    teacher.requires_grad_(False)
    output, evaluation = Path(cfg.output_dir), Path(cfg.evaluation_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    if not cfg.resume and any((output/n).exists() for n in ['model.pt','best.pt','last.pt']):
        raise FileExistsError('Existing v2 artifacts; use an explicit resume checkpoint')
    generator = torch.Generator().manual_seed(cfg.seed)
    sampler = BalancedCoverageSampler([r['label'] for r in splits['train']], generator)
    loaders = {s: DataLoader(FolderImages(rows, image_transforms(cfg.image_size,s=='train')),
        batch_size=cfg.batch_size, sampler=sampler if s=='train' else None,
        num_workers=cfg.num_workers, pin_memory=True, generator=generator if s=='train' else None)
        for s,rows in splits.items()}
    scaler = torch.amp.GradScaler('cuda')
    old_indices = torch.tensor([classes.index(c) for c in originals],device=device)
    best_loss, bad, start, history, previous, best_epoch = float('inf'), 0, 0, [], None, None
    best_state = None
    elapsed_before = 0
    critical = {k:v for k,v in asdict(cfg).items() if k not in {'data_dir','output_dir','evaluation_dir','initial_checkpoint','resume','num_workers'}}
    if recovery is not None:
        if recovery['fingerprint'] != manifest['fingerprint'] or recovery['classes'] != classes or checkpoint_config != critical:
            raise ValueError('Resume dataset, mapping, or configuration mismatch')
        model.load_state_dict(recovery['model'])
        start, best_loss, bad = recovery['epoch']+1, recovery['best_loss'], recovery['bad_epochs']
        best_state, best_epoch, previous = recovery['best_model'], recovery['best_epoch'], recovery['stage']
        history, elapsed_before = recovery['history'], recovery.get('elapsed_seconds',0)
        restore_rng(recovery['rng'],generator)
        scaler.load_state_dict(recovery['scaler'])
        atomic_save({'model':best_state,'classes':classes,'epoch':best_epoch},output/'best.pt')
    (output/'config.json').write_text(json.dumps(asdict(cfg),indent=2))
    (output/'classes.json').write_text(json.dumps(classes,indent=2))
    started = time.monotonic()
    for epoch in range(start,cfg.head_epochs+cfg.finetune_epochs):
        stage = 'head' if epoch < cfg.head_epochs else 'finetune'
        if stage != previous:
            bad = 0
        elif bad >= cfg.patience and len(history) >= cfg.min_epochs:
            if stage == 'head':
                continue
            break
        for index, block in enumerate(model.features):
            block.requires_grad_(stage=='finetune' and index>=cfg.finetune_from)
        if stage != previous:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                lr=cfg.head_lr if stage=='head' else cfg.finetune_lr,weight_decay=cfg.weight_decay)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2,min_lr=1e-7)
        if recovery is not None and recovery['stage']==stage:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg.head_lr,weight_decay=cfg.weight_decay)
            optimizer.load_state_dict(recovery['optimizer'])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2,min_lr=1e-7)
            scheduler.load_state_dict(recovery['scheduler'])
            recovery = None
        training = train_epoch(model,teacher,loaders['train'],device,optimizer,scaler,old_indices,cfg,stage=='head')
        validation, actual, predicted = run_epoch(model,loaders['val'],device)
        actual, predicted = np.array(actual), np.array(predicted)
        old_mask = np.isin(actual,old_indices.cpu().numpy())
        validation['original_top1'] = float((actual[old_mask]==predicted[old_mask]).mean())
        validation['new_top1'] = float((actual[~old_mask]==predicted[~old_mask]).mean()) if (~old_mask).any() else None
        scheduler.step(validation['loss'])
        improved = validation['loss'] < best_loss-1e-4
        if improved:
            best_loss, bad, best_epoch = validation['loss'],0,epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            atomic_save({'model':best_state,'classes':classes,'epoch':epoch},output/'best.pt')
        else:
            bad += 1
        history.append({'epoch':epoch,'stage':stage,'train':training,'val':validation,'learning_rate':optimizer.param_groups[0]['lr']})
        elapsed = elapsed_before + time.monotonic()-started
        state = {'checkpoint_kind':'last','model':model.state_dict(),'classes':classes,'fingerprint':manifest['fingerprint'],
                 'critical_config':critical,'epoch':epoch,'stage':stage,'best_loss':best_loss,'bad_epochs':bad,
                 'best_model':best_state,'best_epoch':best_epoch,'history':history,'elapsed_seconds':elapsed,
                 'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng':rng_state(generator)}
        atomic_save(state,output/'last.pt')
        if (epoch+1)%5==0:
            # Immutable recovery snapshot for reliable transfer while the worker continues.
            recovery_dir=output/'recovery';recovery_dir.mkdir(exist_ok=True)
            atomic_save(state,recovery_dir/f'epoch_{epoch+1:04d}.pt')
        (output/'history.json').write_text(json.dumps(history,indent=2))
        (output/'training_status.json').write_text(json.dumps({'state':'training','epochs':len(history),'gpu':torch.cuda.get_device_name(0),'elapsed_seconds':elapsed,'latest':history[-1]},indent=2))
        print(json.dumps(history[-1]),flush=True)
        previous = stage
    if best_state is None:
        raise RuntimeError('No trained checkpoint produced')
    model.load_state_dict(best_state)
    roles = {r['class_name']:r['role'] for r in manifest['records']}
    metrics = evaluate_complete(model,loaders['test'],loaders['val'],device,classes,evaluation,history,class_roles=roles,original_classes=originals)
    calibration = json.loads((evaluation/'calibration.json').read_text())
    export = {'architecture':'mobilenet_v2','model':model.cpu().state_dict(),'classes':classes,'original_classes':originals,
              'class_roles':roles,'image_size':cfg.image_size,'temperature':calibration['temperature'],
              'unknown_threshold':calibration['unknown_threshold'],'calibration':calibration,'mean':NORMALIZE_MEAN,'std':NORMALIZE_STD,
              'fingerprint':manifest['fingerprint'],'trained':True,'model_version':'v2',
              'source_revision':manifest['source_revision'],'initial_model_sha256':initial_digest,
              'preprocessing':'letterbox_rgb_224_imagenet_normalization'}
    atomic_save(export,output/'model.pt')
    (output/'training_status.json').write_text(json.dumps({'state':'complete','epochs':len(history),'best_epoch':best_epoch,
        'elapsed_seconds':elapsed_before+time.monotonic()-started,'gpu':torch.cuda.get_device_name(0),'test_metrics':metrics},indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    train(ExpansionConfig(**json.loads(Path(args.config).read_text())))
