# rffgpa_layer_trans.py
# RFF-GPA: Gaussian-Process attention via Random Fourier Features and
#          per-head kernel ridge regression (deterministic posterior mean).
#
# Mapping per head:
#   out(Q) = F(Q) [ F(K)^T F(K) + ? I ]^{-1} F(K)^T V
#
# Shapes (per batch):
#   Q,K,V: [B,H,L,D]   with D = hdim // num_heads
#   F():  [B,H,L,M2]  with M2 = 2 * rff_m  (cos & sin features)
#
# Returns:
#   samples: [B,S,L,hdim]  (S==sample_size; deterministic output repeated if S>1)
#   reg: None  (pure MLE loss; control smoothness via --layer_ridge and weight_decay)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RFFGPA_LAYER(nn.Module):
    def __init__(self,
                 device,
                 num_heads,
                 max_len,
                 hdim,
                 kernel_type,          # kept for API parity (expects 'std')
                 sample_size,
                 jitter,               # kept for API parity (unused here)
                 keys_len,             # kept for API parity (unused here)
                 drop_rate,
                 flag_cgp,             # kept for API parity (unused here)
                 rff_m: int = 128,     # number of base features per head (cos+sin -> 2*rff_m)
                 lengthscale: float = 1.0,
                 ridge: float = 1e-2,  # ? in (FK^T FK + ? I)
                 shared_features: bool = False,  # False = per-head features; True = shared across heads
                 **kwargs):
        super().__init__()
        self.device = device
        self.num_heads = int(num_heads)
        self.hdim = int(hdim)
        self.vdim = self.hdim // self.num_heads
        self.sample_size = max(1, int(sample_size))
        self.drop_rate = float(drop_rate)
        self.M = int(rff_m)
        self.M2 = 2 * self.M
        self.lengthscale = float(lengthscale)
        self.ridge = float(ridge)
        self.shared_features = bool(shared_features)

        # Projections to Q/K/V (like standard attention)
        self.q_proj = nn.Linear(self.hdim, self.hdim, bias=False)
        self.k_proj = nn.Linear(self.hdim, self.hdim, bias=False)
        self.v_proj = nn.Linear(self.hdim, self.hdim, bias=False)

        # Output projection (matches your other layers)
        self.W_O = nn.Sequential(nn.Linear(self.hdim, self.hdim), nn.Dropout(self.drop_rate))

        # Random Fourier feature parameters.
        # Per-head by default: W ~ N(0, I/l^2), b ~ Uniform(0, 2p)
        if self.shared_features:
            W = torch.randn(self.vdim, self.M) / self.lengthscale               # [D, M]
            b = 2 * math.pi * torch.rand(self.M)                                # [M]
            self.register_buffer("rff_W_shared", W)
            self.register_buffer("rff_b_shared", b)
            self.rff_W = None
            self.rff_b = None
        else:
            W = torch.randn(self.num_heads, self.vdim, self.M) / self.lengthscale  # [H, D, M]
            b = 2 * math.pi * torch.rand(self.num_heads, self.M)                   # [H, M]
            self.register_buffer("rff_W", W)
            self.register_buffer("rff_b", b)
            self.rff_W_shared = None
            self.rff_b_shared = None

        # Normalization factor for cos/sin stacked features
        self.rff_scale = math.sqrt(1.0 / self.M)  # because we concatenate cos and sin -> total 2M, each scaled by 1/sqrt(M)

    # ---------- utils ----------

    def _reshape_heads(self, x):
        # x: [B, L, hdim] -> [B, H, L, D]
        B, L, _ = x.shape
        return x.view(B, L, self.num_heads, self.vdim).permute(0, 2, 1, 3).contiguous()

    def _prep_masks(self, input_mask, L):
        """
        input_mask: [B,L] or [B,1,L]
        Returns:
          qmask: [B,1,L,1]  (zeros out query positions in the output)
          w_k:   [B,1,L,1]  (weights keys/tokens in KRR sums)
        """
        if input_mask.dim() == 3 and input_mask.size(1) == 1:
            mask = input_mask.squeeze(1)
        else:
            mask = input_mask
        mask = mask.float()[:, :L]          # [B,L]
        qmask = mask.unsqueeze(1).unsqueeze(-1)   # [B,1,L,1]
        w_k   = qmask.clone()                      # same shape / meaning for keys
        return qmask, w_k

    def _rff(self, X):
        # X: [B,H,L,D] -> Φ(X): [B,H,L,2M]
        B, H, L, D = X.shape
        if self.shared_features:
            W = self.rff_W_shared.to(device=X.device, dtype=X.dtype)   # [D,M]
            b = self.rff_b_shared.to(device=X.device, dtype=X.dtype)   # [M]
            proj = torch.einsum('bhld,dm->bhlm', X, W)                 # [B,H,L,M]
            proj = proj + b.view(1, 1, 1, -1)                          # [1,1,1,M]
        else:
            W = self.rff_W.to(device=X.device, dtype=X.dtype)          # [H,D,M]
            b = self.rff_b.to(device=X.device, dtype=X.dtype)          # [H,M]
            proj = torch.einsum('bhld,hdm->bhlm', X, W)                # [B,H,L,M]
            proj = proj + b.view(1, H, 1, -1)                          # [1,H,1,M]

        cosf = torch.cos(proj)
        sinf = torch.sin(proj)
        phi = self.rff_scale * torch.cat([cosf, sinf], dim=-1)         # [B,H,L,2M]
        return phi


    # ---------- forward ----------

    def forward(self, x, cur_k, input_mask):
        """
        x:        [B,L,hdim]  (ln'd embeddings from scaffold)
        cur_k:    unused here (inducing-free; can be tied in if desired)
        input_mask: [B,L] or [B,1,L]
        """
        B, L, _ = x.shape

        # standard projections
        q = self._reshape_heads(self.q_proj(x))  # [B,H,L,D]
        k = self._reshape_heads(self.k_proj(x))  # [B,H,L,D]
        v = self._reshape_heads(self.v_proj(x))  # [B,H,L,D]

        # masks
        qmask, w_k = self._prep_masks(input_mask, L)           # both [B,1,L,1]

        # random features for Q and K
        PhiQ = self._rff(q)                                    # [B,H,L,M2]
        PhiK = self._rff(k)                                    # [B,H,L,M2]

        # weight keys by mask (zero out padded tokens)
        PhiK_w = PhiK * w_k                                    # [B,H,L,M2]
        V_w    = v    * w_k                                    # [B,H,L,D]

        # Compute per-(B,H) feature Gram and cross term:
        #   G = FK_w^T FK_w + ridge I  -> [B,H,M2,M2]
        #   T = FK_w^T V_w             -> [B,H,M2,D]
        G = torch.einsum('bhlm,bhln->bhmn', PhiK_w, PhiK_w)    # [B,H,M2,M2]
        # add ridge on the diagonal
        eye = torch.eye(self.M2, device=G.device, dtype=G.dtype).view(1,1,self.M2,self.M2)
        G = G + self.ridge * eye

        T = torch.einsum('bhlm,bhld->bhmd', PhiK_w, V_w)       # [B,H,M2,D]

        # Solve (G) W = T  for W ? R^{M2�D} using Cholesky (batched)
        Lch = torch.linalg.cholesky(G)                         # [B,H,M2,M2]
        # First solve L y = T
        y = torch.linalg.solve_triangular(Lch, T, upper=False)
        # Then solve L^T W = y
        W = torch.linalg.solve_triangular(Lch.transpose(-2, -1), y, upper=True)  # [B,H,M2,D]

        # Predict at Q: out = FQ @ W
        out = torch.einsum('bhlm,bhmd->bhld', PhiQ, W)         # [B,H,L,D]

        # Pack to [B,1,L,hdim] and project out
        samples = out.unsqueeze(2)                              # [B,H,1,L,D]
        samples = torch.flatten(samples.permute(0,2,3,1,4), -2, -1)  # [B,1,L,hdim]
        samples = self.W_O(samples)

        # zero outputs on padded queries
        samples = samples * qmask

        # repeat for sample_size > 1 (deterministic)
        if self.sample_size > 1:
            samples = samples.expand(B, self.sample_size, L, self.hdim).contiguous()

        return samples, None  # no extra regularizer; tune with ridge/weight_decay
