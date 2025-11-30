# eval_cifar_id.py (robust inference)
import os, glob, re, argparse, collections
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from vit import ViT  # unchanged

# ---------------------- basic utils ----------------------
def pick_checkpoint(path: str) -> str:
    if os.path.isfile(path):
        return path
    for name in ('best.pt','best.pth','ckpt.pt','ckpt.pth','last.pt','last.pth'):
        p = os.path.join(path, name)
        if os.path.isfile(p):
            return p
    files = glob.glob(os.path.join(path, '*.pt')) + glob.glob(os.path.join(path, '*.pth'))
    if not files:
        # allow single file *without* extension named like the folder
        if os.path.isfile(path):
            return path
        raise FileNotFoundError(f'No checkpoint files in: {path}')
    files.sort(key=lambda s: (0 if 'best' in os.path.basename(s).lower() else 1, -os.path.getmtime(s)))
    return files[0]

def _try_get(state: Dict[str, torch.Tensor], suffix: str) -> Optional[torch.Tensor]:
    for k, v in state.items():
        if k.endswith(suffix):
            return v
    return None

def _find_keys_like(state: Dict[str, torch.Tensor], pat: str) -> List[str]:
    rgx = re.compile(pat)
    return [k for k in state.keys() if rgx.search(k)]

def _all_layer_indices(prefix: str, state: Dict[str, torch.Tensor]) -> List[int]:
    pat = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    idxs = []
    for k in state.keys():
        m = pat.match(k)
        if m:
            idxs.append(int(m.group(1)))
    return sorted(set(idxs))

# ---------------------- robust inference ----------------------
def infer_hparams_from_state(state: Dict[str, torch.Tensor], args) -> Dict[str, Any]:
    """
    Infer ViT config from a checkpoint state_dict. Priority:
      1) hdim from patch embed / pos_emb / layer norms / heads
      2) num_heads & vdim from gp params (log_ls / omega / keys.*)
      3) keys_len from omega/keys.*
      4) depth from sgp_layer_list.*
      5) patch_size from pos_emb length or patch proj shape
    """
    info: Dict[str, Any] = {}

    # --- flag_sgp
    has_qv  = any(k.endswith("fc_qv.weight")  for k in state.keys())
    has_qkv = any(k.endswith("fc_qkv.weight") for k in state.keys())
    if has_qv:  info['flag_sgp'] = True
    if has_qkv: info['flag_sgp'] = False
    if 'flag_sgp' not in info:
        info['flag_sgp'] = bool(args.flag_sgp)

    # --- collect candidates for hdim
    cands = []
    w = _try_get(state, "patch_embedding.linear_proj.0.weight")  # conv proj: [hdim, 3, p, p]
    if w is not None and w.ndim == 4:
        cands.append(int(w.shape[0]))
    pos = _try_get(state, "patch_embedding.pos_emb")  # [L, hdim]
    if pos is not None and pos.ndim == 2:
        cands.append(int(pos.shape[1]))
    lnw = _try_get(state, "ln.weight")
    if lnw is not None and lnw.ndim == 1:
        cands.append(int(lnw.shape[0]))
    ch_ln = _try_get(state, "class_head.ln.weight")
    if ch_ln is not None and ch_ln.ndim == 1:
        cands.append(int(ch_ln.shape[0]))
    ch_fc = _try_get(state, "class_head.fc.weight")  # [num_cls, hdim]
    if ch_fc is not None and ch_fc.ndim == 2:
        cands.append(int(ch_fc.shape[1]))
    w_qv = _try_get(state, "fc_qv.weight")  # [2*hdim, hdim]
    if w_qv is not None and w_qv.ndim == 2:
        cands.append(int(w_qv.shape[1]))
    w_qkv = _try_get(state, "fc_qkv.weight")  # [3*hdim, hdim]
    if w_qkv is not None and w_qkv.ndim == 2:
        cands.append(int(w_qkv.shape[1]))

    # heads & vdim from GP params if available
    log_ls = _try_get(state, "log_ls")  # [H, vdim]
    omega_full = _try_get(state, "omega_full")  # [H, vdim, M]
    # keys.* sometimes store learned inducing keys: [H, 1, K, vdim]
    keys_params = []
    for i in range(64):
        kp = state.get(f"keys.{i}", None)
        if kp is not None and kp.ndim == 4:
            keys_params.append(kp)

    if log_ls is not None and log_ls.ndim == 2:
        H, vdim = int(log_ls.shape[0]), int(log_ls.shape[1])
        info['num_heads'] = H
        cands.append(H * vdim)
    if omega_full is not None and omega_full.ndim == 3:
        H, vdim, M = map(int, omega_full.shape)
        info['num_heads'] = H
        info['keys_len'] = M
        cands.append(H * vdim)
    if keys_params:
        H, one, K, vdim = map(int, keys_params[0].shape)
        info['num_heads'] = H
        info['keys_len'] = K
        cands.append(H * vdim)

    # choose hdim = mode of candidates (fallback to CLI)
    if cands:
        hdim = collections.Counter(cands).most_common(1)[0][0]
        info['hdim'] = hdim
    else:
        info['hdim'] = int(args.hdim)

    # if num_heads missing, try to infer from divisors of hdim
    if 'num_heads' not in info:
        for H in (8,6,4,2,1,12):  # common values
            if info['hdim'] % H == 0:
                info['num_heads'] = H
                break
    if 'num_heads' not in info:
        info['num_heads'] = int(args.num_heads)

    # if keys_len missing, try omega/keys; else leave default
    if 'keys_len' not in info:
        if omega_full is not None and omega_full.ndim == 3:
            info['keys_len'] = int(omega_full.shape[-1])
        elif keys_params:
            info['keys_len'] = int(keys_params[0].shape[2])
        else:
            info['keys_len'] = int(args.keys_len)

    # depth from layer list
    layer_idxs = _all_layer_indices("sgp_layer_list", state)
    if layer_idxs:
        info['depth'] = max(layer_idxs) + 1
    else:
        info['depth'] = int(args.depth)

    # patch_size + max_len
    # Prefer pos_emb length if present (L tokens = (32/ps)^2 for CIFAR-32)
    if pos is not None and pos.ndim == 2:
        L = int(pos.shape[0])
        # solve ps from L = (32/ps)^2
        import math
        ps = int(round(32 / math.sqrt(L)))
        if ps > 0 and (32 % ps == 0):
            info['patch_size'] = ps
            info['max_len'] = L
    # else infer from conv proj kernel size
    if 'patch_size' not in info:
        if w is not None and w.ndim == 4 and w.shape[2] == w.shape[3]:
            ps = int(w.shape[2])
            if ps in (2,4,8,16):
                info['patch_size'] = ps
                info['max_len'] = (32 // ps) * (32 // ps)
    # fallback
    if 'patch_size' not in info:
        info['patch_size'] = int(args.patch_size)
        info['max_len'] = (32 // info['patch_size']) * (32 // info['patch_size'])

    return info

def build_model(args, inferred: Dict[str, Any], device):
    kw = dict(
        device=device,
        depth=inferred['depth'],
        patch_size=inferred['patch_size'],
        in_channels=3,
        max_len=inferred['max_len'],
        num_class=args.num_class,
        hdim=inferred['hdim'],
        num_heads=inferred['num_heads'],
        sample_size=args.sample_size,
        jitter=args.jitter,
        drop_rate=args.drop_rate,
        keys_len=inferred['keys_len'],
        kernel_type=args.kernel_type,
        flag_sgp=inferred['flag_sgp'],
    )
    print("==> Inferred/used ViT kwargs:")
    for k in ['depth','patch_size','max_len','hdim','num_heads','keys_len','flag_sgp']:
        print(f"    {k}: {kw[k]}")
    return ViT(**kw).to(device)

def load_state_flex(model, ckpt):
    # accept either raw state_dict or dict containing one
    if isinstance(ckpt, dict):
        state_dict = None
        for k in ('state_dict','model_state','model','weights'):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]; break
        if state_dict is None:
            state_dict = ckpt
    else:
        state_dict = ckpt
    res = model.load_state_dict(state_dict, strict=False)
    print(f"==> load_state_dict: missing={len(res.missing_keys)}  unexpected={len(res.unexpected_keys)}")
    if res.missing_keys:    print("    (first 10 missing)   :", res.missing_keys[:10])
    if res.unexpected_keys: print("    (first 10 unexpected):", res.unexpected_keys[:10])

# ---------------------- metrics ----------------------
def compute_ece_mce(conf, pred, true, n_bins=20) -> Tuple[float, float]:
    import numpy as np
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0; mce = 0.0; N = conf.shape[0]
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(mask.sum())
        if cnt == 0: 
            continue
        acc_bin = (pred[mask] == true[mask]).mean()
        conf_bin = conf[mask].mean()
        gap = abs(acc_bin - conf_bin)
        ece += (cnt / N) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)

@torch.no_grad()
def evaluate(model, loader, device, mc: int):
    total = 0
    correct = 0
    nll_sum = 0.0
    all_conf, all_pred, all_true = [], [], []

    for images, labels in tqdm(loader, desc='Eval'):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if mc <= 1:
            out = model(images)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.softmax(logits.float(), dim=-1)
        else:
            probs_accum = None
            for _ in range(mc):
                out = model(images)
                logits_s = out[0] if isinstance(out, tuple) else out
                ps = torch.softmax(logits_s.float(), dim=-1)
                probs_accum = ps if probs_accum is None else (probs_accum + ps)
            probs = probs_accum / float(mc)

        conf, pred = probs.max(dim=-1)
        nll_sum += (-torch.log(probs.gather(1, labels.view(-1,1)).clamp_min(1e-12))).sum().item()
        correct += (pred == labels).sum().item()
        total += labels.size(0)
        all_conf.append(conf.cpu()); all_pred.append(pred.cpu()); all_true.append(labels.cpu())

    import numpy as np
    conf = torch.cat(all_conf).numpy()
    pred = torch.cat(all_pred).numpy()
    true = torch.cat(all_true).numpy()

    acc = correct/total
    nll = nll_sum/total
    ece, mce = compute_ece_mce(conf, pred, true, n_bins=20)
    return acc, nll, ece, mce

# ---------------------- main ----------------------
def main(args):
    device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    test_set = CIFAR10(root='./data/', train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_set, batch_size=args.batch_size_test,
                             shuffle=False, num_workers=args.workers, pin_memory=True)

    ckpt_path = pick_checkpoint(args.model_path)
    print(f"==> Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # choose dict to inspect for shapes
    state_dict = ckpt['state_dict'] if (isinstance(ckpt, dict) and isinstance(ckpt.get('state_dict', None), dict)) else ckpt

    inferred = infer_hparams_from_state(state_dict, args)
    model = build_model(args, inferred, device)
    load_state_flex(model, ckpt)

    acc, nll, ece, mce = evaluate(model, test_loader, device, mc=args.mc)
    print(f'Accuracy: {acc*100:.2f}%')
    print(f'NLL     : {nll:.4f}')
    print(f'ECE(20) : {ece:.4f}')
    print(f'MCE     : {mce:.4f}')

    if args.out_csv:
        import csv
        write_header = not os.path.exists(args.out_csv)
        with open(args.out_csv, 'a', newline='') as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(['checkpoint','depth','hdim','heads','patch','keys_len','flag_sgp','mc','ACC','NLL','ECE','MCE'])
            w.writerow([
                ckpt_path, inferred['depth'], inferred['hdim'], inferred['num_heads'],
                inferred['patch_size'], inferred['keys_len'], int(inferred['flag_sgp']),
                args.mc, acc, nll, ece, mce
            ])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cuda', type=int, default=0)
    ap.add_argument('--depth', type=int, default=5)
    ap.add_argument('--num_class', type=int, default=10)
    ap.add_argument('--hdim', type=int, default=128)
    ap.add_argument('--num_heads', type=int, default=4)
    ap.add_argument('--sample_size', type=int, default=1)
    ap.add_argument('--jitter', type=float, default=1e-6)
    ap.add_argument('--drop_rate', type=float, default=0.1)
    ap.add_argument('--patch_size', type=int, default=4)
    ap.add_argument('--max_len', type=int, default=64)
    ap.add_argument('--keys_len', type=int, default=16)
    ap.add_argument('--kernel_type', type=str, default='ard')
    ap.add_argument('--flag_sgp', type=bool, default=True)
    ap.add_argument('--batch_size_test', type=int, default=256)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--mc', type=int, default=1)
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--out_csv', type=str, default=None)
    args = ap.parse_args()
    main(args)
