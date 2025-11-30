# sgp_layer_rff_cgp.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Utilities --------------------------------------------------------------
def init_linear_layer(linear_layer, std=1e-2):
    nn.init.normal_(linear_layer.weight, std=std)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)

# RFF features shared across q/o/k for consistent cross-covariances
class SharedRFF(nn.Module):
    def __init__(self, d_lat, m_features, lengthscale=1.0):
        super().__init__()
        self.m = int(m_features)
        # Shared across heads (per-layer)
        omega = torch.randn(d_lat, self.m) / float(lengthscale)
        b = 2 * math.pi * torch.rand(self.m)
        self.register_buffer('omega', omega)
        self.register_buffer('b', b)

    def forward(self, X):
        # X: [B, H, N, D]  -> Phi: [B, H, N, m]
        Z = torch.einsum('bhnd,dm->bhnm', X, self.omega)  # [B,H,N,m]
        Z = Z + self.b.view(1, 1, 1, -1)
        return math.sqrt(2.0 / self.m) * torch.cos(Z)

# ---- Layer ------------------------------------------------------------------
class SGP_LAYER(nn.Module):
    """
    RFF-CGP attention (mean-only) with GP-style regularizer.
    Drop-in for your ViT:
      __init__(device, num_heads, max_len, hdim, kernel_type, sample_size, jitter,
               keys_len, drop_rate, flag_sgp, inference_mode=False, **kwargs)
      forward(x, cur_k) -> (out, kl)
        out: [B, 1, L, hdim]
        kl : scalar regularizer
    """
    def __init__(self, device, num_heads, max_len, hdim, kernel_type,
                 sample_size, jitter, keys_len, drop_rate, flag_sgp,
                 inference_mode=False, **kwargs):
        super().__init__()
        assert hdim % num_heads == 0
        self.device = device
        self.num_heads = num_heads
        self.max_len = max_len
        self.hdim = hdim
        self.vdim = hdim // num_heads
        self.drop_rate = drop_rate
        self.sample_size = int(sample_size)
        self.inference_mode = inference_mode

        # Treat this layer as CGP path (flag_sgp is kept for parity)
        self.flag_cgp = True

        # RFF/CGP hyperparams (configurable via kwargs)
        self.lambda_noise = float(kwargs.get('lambda_noise', 1e-2))  # ?
        self.jitter = float(jitter) if isinstance(jitter, (int, float)) else 1e-6
        self.lengthscale = float(kwargs.get('lengthscale', 1.0))
        self.cgp_eps = float(kwargs.get('cgp_eps', 2e-3))
        self.loss_impl = str(kwargs.get('loss_impl', 'gp_nll_true'))  # 'gp_nll_true' or 'cgp'
        self.rff_m = int(kwargs.get('rff_m', keys_len))               # use keys_len as m by default

        # projections
        self.fc_q  = nn.Linear(hdim, hdim)
        self.fc_k  = nn.Linear(hdim, hdim)
        self.fc_v  = nn.Linear(hdim, hdim)
        self.fc_x0 = nn.Linear(hdim, hdim)   # "o" path (x0)
        self.W_O   = nn.Sequential(nn.Linear(hdim, hdim), nn.Dropout(drop_rate))
        for m in [self.fc_q, self.fc_k, self.fc_v, self.fc_x0, self.W_O[0]]:
            init_linear_layer(m)

        # dot-product scale (used only if we fallback to SDP)
        self.scale = self.vdim ** (-0.5)

        # shared random features (per layer, shared across heads)
        self.rff = SharedRFF(self.vdim, self.rff_m, self.lengthscale)

    # --- helpers -------------------------------------------------------------
    def _qkv(self, x, asym=True):
        # x: [B, L, hdim] -> q,k,v,x0: [B,H,L,D]
        B, L, _ = x.shape
        q = self.fc_q(x).view(B, L, self.num_heads, self.vdim).permute(0, 2, 1, 3)
        x0 = self.fc_x0(x).view(B, L, self.num_heads, self.vdim).permute(0, 2, 1, 3)
        if asym:
            k = self.fc_k(x).view(B, L, self.num_heads, self.vdim).permute(0, 2, 1, 3)
        else:
            k = q.clone()
        v = self.fc_v(x).view(B, L, self.num_heads, self.vdim).permute(0, 2, 1, 3)
        return q, k, v, x0

    def _cgp_lower_bounds(self, Phi_q, Phi_k, Phi_o, y):
        """
        Jensen lower bounds computed purely in mm feature space.
        Phi_*: [B,H,N,m],  y: [B,H,N,D]
        returns LB_q, LB_k: [B,H]
        """
        B, H, N, m = Phi_q.shape
        D = y.shape[-1]
        lam = self.lambda_noise
        eps = self.cgp_eps
        dev = Phi_q.device

        I_m = torch.eye(m, device=dev).view(1, 1, m, m)

        # Feature-space Grams
        G  = torch.matmul(Phi_o.transpose(-1, -2), Phi_o)   # [B,H,m,m]
        Cq = torch.matmul(Phi_q.transpose(-1, -2), Phi_q)   # [B,H,m,m]
        Ck = torch.matmul(Phi_k.transpose(-1, -2), Phi_k)   # [B,H,m,m]
        A  = G + lam * I_m                                  # [B,H,m,m]

        # Inverses via Cholesky (batched)
        L_A = torch.linalg.cholesky(A)                      # [B,H,m,m]
        Iexp = I_m.expand(B, H, m, m).contiguous()
        A_inv = torch.cholesky_solve(Iexp.reshape(B*H, m, m),
                                     L_A.reshape(B*H, m, m)).reshape(B, H, m, m)

        S_q = (1.0/lam) * A + (1.0/eps) * Cq
        S_k = (1.0/lam) * A + (1.0/eps) * Ck
        L_Sq = torch.linalg.cholesky(S_q)
        L_Sk = torch.linalg.cholesky(S_k)
        S_q_inv = torch.cholesky_solve(Iexp.reshape(B*H, m, m),
                                       L_Sq.reshape(B*H, m, m)).reshape(B, H, m, m)
        S_k_inv = torch.cholesky_solve(Iexp.reshape(B*H, m, m),
                                       L_Sk.reshape(B*H, m, m)).reshape(B, H, m, m)

        # Query term
        yy = torch.sum(y * y, dim=(2, 3))                                   # [B,H]
        PhiT_y = torch.matmul(Phi_q.transpose(-1, -2), y)                    # [B,H,m,D]
        Sq_inv_PhiT_y = torch.matmul(S_q_inv, PhiT_y)                        # [B,H,m,D]
        datafit_q = (1.0/eps) * yy - (1.0/(eps*eps)) * torch.sum(PhiT_y * Sq_inv_PhiT_y, dim=(2, 3))

        Cq_Sinv_Cq = torch.matmul(torch.matmul(Cq, S_q_inv), Cq)
        M_q = (1.0/eps) * Cq - (1.0/(eps*eps)) * Cq_Sinv_Cq
        G_Ainv = torch.matmul(G, A_inv)
        model_q = D * torch.einsum('bhmn,bhmn->bh', G_Ainv, M_q)

        Ainv_Cq = torch.matmul(A_inv, Cq)
        Iadd_q = Iexp + (lam/eps) * Ainv_Cq
        _, logdet_Iq = torch.slogdet(Iadd_q)
        logdet_q = N * math.log(eps) + logdet_Iq

        LB_q = -0.5 * (datafit_q + model_q + logdet_q + N * math.log(2*math.pi))  # [B,H]

        # Key term
        tr_Ck  = torch.diagonal(Ck, dim1=-2, dim2=-1).sum(-1)             # [B,H]
        Ck2 = torch.matmul(Ck, Ck)
        Skinv_Ck2 = torch.matmul(S_k_inv, Ck2)
        tr_part = torch.diagonal(Skinv_Ck2, dim1=-2, dim2=-1).sum(-1)
        datafit_k = D * ((1.0/eps) * tr_Ck - (1.0/(eps*eps)) * tr_part)

        Ck_Sinv_Ck = torch.matmul(torch.matmul(Ck, S_k_inv), Ck)
        M_k = (1.0/eps) * Ck - (1.0/(eps*eps)) * Ck_Sinv_Ck
        model_k = D * torch.einsum('bhmn,bhmn->bh', G_Ainv, M_k)

        Ainv_Ck = torch.matmul(A_inv, Ck)
        Iadd_k = Iexp + (lam/eps) * Ainv_Ck
        _, logdet_Ik = torch.slogdet(Iadd_k)
        logdet_k = N * math.log(eps) + logdet_Ik

        LB_k = -0.5 * (datafit_k + model_k + logdet_k + N * math.log(2*math.pi))  # [B,H]
        return LB_q, LB_k

    # --- core ---------------------------------------------------------------
    def forward(self, x, _cur_k_unused=None):
        """
        x: [B, L, hdim]
        returns:
          out: [B, 1, L, hdim]
          kl : scalar regularizer
        """
        B, L, _ = x.shape

        if not self.flag_cgp:
            # Fallback to dot-product (not expected in this layer; kept for parity)
            q, k, v, _ = self._qkv(x, asym=True)
            attn = torch.softmax(self.scale * torch.einsum('bhid,bhjd->bhij', q, k), dim=-1)
            y = torch.einsum('bhij,bhjd->bhid', attn, v)  # [B,H,L,D]
            out = y.unsqueeze(2).permute(0, 2, 3, 1, 4).contiguous().view(B, 1, L, self.hdim)
            return self.W_O(out), None

        # RFF-CGP path
        lam = self.lambda_noise
        q, k, v, x0 = self._qkv(x, asym=True)     # [B,H,L,D] each

        # Random features
        Phi_q = self.rff(q)            # [B,H,L,m]
        Phi_k = self.rff(k)            # [B,H,L,m]
        Phi_o = self.rff(x0)           # [B,H,L,m]

        # Small Grams in feature space (m x m)
        Gk = torch.einsum('bhnm,bhnp->bhmp', Phi_k, Phi_k)  # [B,H,m,m]
        Go = torch.einsum('bhnm,bhnp->bhmp', Phi_o, Phi_o)  # [B,H,m,m]

        Bf, H, m, _ = Gk.shape
        Ik = torch.eye(m, device=x.device).view(1,1,m,m).expand(Bf, H, m, m)
        Io = Ik

        Ak = Gk + lam * Ik       # [B,H,m,m]
        Ao = Go + lam * Io       # [B,H,m,m]

        # Cholesky solves (flatten BH for batched ops)
        Lk = torch.linalg.cholesky(Ak)             # [B,H,m,m]
        Lo = torch.linalg.cholesky(Ao)             # [B,H,m,m]
        Lk_flat = Lk.reshape(Bf*H, m, m)
        Lo_flat = Lo.reshape(Bf*H, m, m)

        # u = (z_k - Phi_k Ak^{-1} Phi_k^T z_k)/?
        Phi_k_T_z = torch.einsum('bhnm,bhnd->bhmd', Phi_k, v)             # [B,H,m,D]
        rhs = Phi_k_T_z.reshape(Bf*H, m, self.vdim)
        Ak_inv_Phi_k_T_z = torch.cholesky_solve(rhs, Lk_flat).reshape(Bf, H, m, self.vdim)
        term = torch.einsum('bhnm,bhmd->bhnd', Phi_k, Ak_inv_Phi_k_T_z)   # [B,H,L,D]
        u = (v - term) / lam                                              # [B,H,L,D]

        # B_o = (Phi_o^T Phi_o) Ao^{-1}
        I_flat = torch.eye(m, device=x.device).expand(Bf*H, m, m).contiguous()
        Ao_inv = torch.cholesky_solve(I_flat, Lo_flat).reshape(Bf, H, m, m)
        B_o = torch.einsum('bhmp,bhpq->bhmq', Go, Ao_inv)                 # [B,H,m,m]

        # y = Phi_q B_o (Phi_k^T u)
        Phi_k_T_u = torch.einsum('bhnm,bhnd->bhmd', Phi_k, u)             # [B,H,m,D]
        mid = torch.einsum('bhmq,bhqd->bhmd', B_o, Phi_k_T_u)             # [B,H,m,D]
        y = torch.einsum('bhnm,bhmd->bhnd', Phi_q, mid)                   # [B,H,L,D]

        # heads -> tokens
        out = y.unsqueeze(2).permute(0, 2, 3, 1, 4).contiguous().view(B, 1, L, self.hdim)
        out = self.W_O(out)

        # Regularizer
        if self.loss_impl == 'cgp':
            LB_q, LB_k = self._cgp_lower_bounds(Phi_q, Phi_k, Phi_o, v)   # [B,H] each
            reg = -(LB_q + LB_k).mean()                                   # scalar
        else:
            # GP feature-space NLL on the key path
            logdet_Ak = 2.0 * torch.sum(torch.log(torch.diagonal(Lk, dim1=-2, dim2=-1)), dim=-1)  # [B,H]
            loglike_det = (L - m) * math.log(lam) + logdet_Ak
            zk_sq = torch.sum(v * v, dim=(2, 3))  # [B,H]
            Ak_inv_t = torch.cholesky_solve(Phi_k_T_z.reshape(Bf*H, m, self.vdim), Lk_flat).reshape(Bf, H, m, self.vdim)
            quad_corr = torch.sum(Phi_k_T_z * Ak_inv_t, dim=(2, 3))  # [B,H]
            quad = (1.0/lam) * (zk_sq - quad_corr)
            reg = 0.5 * (loglike_det + quad + L * math.log(2*math.pi)).mean()

        return out, reg
