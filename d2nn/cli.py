"""Portable training and evaluation entry point; all filesystem work is explicit."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .core import (D2NNPhaseOnly, D2NNOpticalClassifier, RingDetector,
                   contrast_loss, mse_onehot_loss, pixel_mse_loss, set_seed)


def build_model(c, device):
    n = c['n_pixels']
    if n < 2 or n % 2 or c['num_layers'] < 1:
        raise ValueError('This preserved implementation requires even N >= 2 and at least one layer.')
    if len(c['distances']) != c['num_layers'] + 1:
        raise ValueError('Provide one distance before each phase layer and one after the last layer.')
    if c['encoding'] != 'amplitude':
        raise ValueError('Phase-input variants remain in legacy/; this entry point uses amplitude encoding.')
    detector = RingDetector(10, n, n, c['ring_distance_px'], c['region_radius_px']).to(device)
    if (detector.masks.sum((-1, -2)) == 0).any():
        raise ValueError('Detector regions must be non-empty.')
    optical = D2NNPhaseOnly(c['num_layers'], c['wavelength'], n, c['pixel_size'], c['distances'], device)
    return D2NNOpticalClassifier(optical, detector).to(device)


def load_checkpoint(model, path, config):
    state = torch.load(path, map_location=model.d2nn.device, weights_only=True)
    if 'state_dict' in state:
        if state['config'] != config:
            raise ValueError('Saved configuration differs; use the config.json saved with the run.')
        state = state['state_dict']
    state = dict(state)
    saved_masks = state.pop('detector.masks', None)
    if saved_masks is not None and not torch.equal(saved_masks, model.detector.masks):
        raise ValueError('Detector masks disagree with checkpoint. Do not silently replace them.')
    # Loading a legacy state_dict also loads physical buffers. Reject mismatches first.
    for name, current in model.state_dict().items():
        if name.endswith(('.kz', '.distance')):
            saved = state.get(name)
            if saved is None or saved.shape != current.shape or not torch.allclose(saved, current, rtol=1e-6, atol=1e-12):
                raise ValueError(f'Physical buffer mismatch: {name}')
    model.load_state_dict(state, strict=True)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def data_loader(c, root, train, batch_size, limit, download):
    ops = [transforms.Resize((c['n_pixels'], c['n_pixels']))]
    if train:
        ops.append(transforms.RandomAffine(degrees=0, translate=(c['translate'],)*2, scale=tuple(c['scale'])))
    ops.append(transforms.ToTensor())
    ds = datasets.MNIST(root=str(root), train=train, download=download, transform=transforms.Compose(ops))
    if limit:
        ds = Subset(ds, range(min(limit, len(ds))))
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0)


def loss_and_scores(model, amplitude, y, name):
    if name == 'pixel_mse':
        intensity = model.forward_intensity(amplitude)
        loss = pixel_mse_loss(intensity, y, model.detector.masks)
        scores = (intensity.squeeze(1).unsqueeze(1) * model.detector.masks.unsqueeze(0)).sum((-1, -2))
    else:
        scores = model(amplitude)
        loss = {'mse': mse_onehot_loss, 'contrast': contrast_loss}[name](scores, y)
    return loss, scores


def run_epoch(model, loader, c, optimizer=None):
    model.train(optimizer is not None)
    cm = np.zeros((10, 10), dtype=np.int64)
    total_loss, count = 0., 0
    for x, y in loader:
        x, y = x.to(model.d2nn.device), y.to(model.d2nn.device)
        amplitude = torch.sqrt(x + 1e-12)
        if optimizer is not None and c['noise_std'] > 0:
            amplitude = (amplitude + torch.randn_like(amplitude)*c['noise_std']).clamp(0, 1)
        with torch.set_grad_enabled(optimizer is not None):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, scores = loss_and_scores(model, amplitude, y, c['loss'])
            if not torch.isfinite(loss) or not torch.isfinite(scores).all():
                raise FloatingPointError('Non-finite loss or scores')
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        predicted = scores.argmax(-1)
        np.add.at(cm, (y.cpu().numpy(), predicted.detach().cpu().numpy()), 1)
        count += len(y)
        total_loss += loss.item()*len(y)
    return {'samples': count, 'correct': int(cm.trace()), 'accuracy': float(cm.trace()/count), 'loss': total_loss/count}, cm


def environment():
    return dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                torchvision=torchvision.__version__, numpy=np.__version__, cuda_available=torch.cuda.is_available(),
                cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)




def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['train', 'evaluate', 'smoke'])
    p.add_argument('--config', type=Path, default=Path('configs/contrast_100.json'))
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, required=True, help='Must be a new directory; prevents overwriting results.')
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int)
    p.add_argument('--limit', type=int, default=0, help='0 means the complete split; positive values are smoke tests only.')
    p.add_argument('--download', action='store_true', help='Explicitly allow MNIST download.')
    a = p.parse_args()
    if a.batch_size < 1 or a.limit < 0 or (a.epochs is not None and a.epochs < 1):
        p.error('Batch size and epochs must be positive; limit must be nonnegative.')
    c = json.loads(a.config.read_text(encoding='utf-8'))
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else ('cpu' if a.device=='auto' else a.device))
    set_seed(c['seed'])
    model = build_model(c, device)
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(c, indent=2)+'\n')
    meta = {'environment': environment(), 'device': str(device), 'batch_size': a.batch_size, 'seed': c['seed'],
            'trainable_parameters': sum(p.numel() for p in model.parameters()), 'command': a.command,
            'limit': a.limit, 'checkpoint_sha256': None,
            'detector_overlap_pixels': int((model.detector.masks.sum(0) > 1).sum().item())}
    if a.command == 'smoke':
        x = torch.rand(2, 1, c['n_pixels'], c['n_pixels'], device=device)
        y = torch.tensor([0,1], device=device)
        opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
        before = model.d2nn.phase[0].detach().clone()
        loss, scores = loss_and_scores(model, torch.sqrt(x+1e-12), y, c['loss'])
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError('Gradient check failed')
        opt.step()
        if torch.equal(before, model.d2nn.phase[0]):
            raise AssertionError('Optimizer made no parameter change')
        meta.update(status='passed', synthetic_samples=2, finite_loss=float(loss.detach()), score_shape=list(scores.shape))
    else:
        test = data_loader(c, a.data_root, False, a.batch_size, a.limit, a.download)
        if a.command == 'evaluate':
            checkpoint = a.checkpoint or (Path(c['checkpoint']) if c.get('checkpoint') else None)
            if checkpoint is None:
                p.error('Evaluation needs --checkpoint or a checkpoint in config.')
            meta['checkpoint_sha256'] = load_checkpoint(model, checkpoint, c)
            started = time.perf_counter()
            metrics, cm = run_epoch(model, test, c)
            meta.update(metrics, elapsed_seconds=time.perf_counter()-started,
                        scope='full MNIST test split' if not a.limit else 'limited subset; not benchmark accuracy')
        else:
            train = data_loader(c, a.data_root, True, a.batch_size, a.limit, a.download)
            opt = torch.optim.Adam(model.parameters(), lr=c['lr'])
            history=[]
            for epoch in range(1, (a.epochs or c['epochs'])+1):
                started = time.perf_counter()
                tr, _ = run_epoch(model, train, c, opt)
                te, cm = run_epoch(model, test, c)
                row = dict(epoch=epoch, train_loss=tr['loss'], train_accuracy=tr['accuracy'],
                           test_loss=te['loss'], test_accuracy=te['accuracy'], seconds=time.perf_counter()-started)
                history.append(row)
                print(json.dumps(row), flush=True)
                with (out/'history.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(history)
            torch.save({'state_dict': model.state_dict(), 'config': c}, out/'model.pt')
            meta.update(te, epochs=len(history), scope='new training run; not historical reproduction')
        np.savetxt(out/'confusion_matrix.csv', cm, delimiter=',', fmt='%d')
    (out/'metrics.json').write_text(json.dumps(meta, indent=2)+'\n')
    print(json.dumps(meta, indent=2), flush=True)
    if a.command != "smoke":
        subprocess.run([sys.executable, str(Path(__file__).with_name("plotting.py")), str(out)], check=True)


if __name__ == '__main__':
    main()
