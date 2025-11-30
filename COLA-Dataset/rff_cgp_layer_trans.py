import numpy.random as npr
import torch
import torch.nn as nn
from util import kernel_ard, kernel_exp, kernel_std
import torch.nn.functional as F
import math
# ---- Utilities --------------------------------------------------------------
def init_linear_layer(linear_layer, std=1e-2):
    nn.init.normal_(linear_layer.weight, std=std)
    if linear_layer.bias is not None:
        nn.init.zeros_(linear_layer.bias)

# RFF features shared across q/o/k for consistent cross-covariances
class SharedRFF(nn.Module):
    def __init__(self, d_lat, m_features, lengthscale=1.0):
        super().__init__()
        self.m = m_features
        self.register_buffer('omega', torch.randn(d_lat, m_features) / float(lengthscale))
        self.register_buffer('b', 2*math.pi*torch.rand(m_features))

    def forward(self, X):
        # X: [B, H, N, D]  -> Phi: [B, H, N, m]
        Z = torch.einsum('bhnd,dm->bhnm', X, self.omega)  # [B,H,N,m]
        Z = Z + self.b.view(1, 1, 1, -1)
        return math.sqrt(2.0 / self.m) * torch.cos(Z)


class RFFCGP_LAYER(nn.Module):
    def __init__(self, device, num_heads, max_len, hdim, kernel_type, sample_size, jitter, keys_len, drop_rate, flag_cgp,
                 rff_m=128, lambda_noise=1e-2, lengthscale=1.0, cgp_eps=2e-3, loss_impl='gp_nll_true'):
        super(RFFCGP_LAYER, self).__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.hdim = hdim
        self.vdim = self.hdim // self.num_heads
        self.dq = self.vdim
        self.flag_cgp = flag_cgp
        self.keys_len = keys_len
        self.drop_rate = drop_rate

        self.lambda_noise = lambda_noise
        self.cgp_eps = cgp_eps
        self.loss_impl = loss_impl
        
        self.sample_size = sample_size
        self.jitter = jitter
        self.device = device
        self.kernel_type = kernel_type 
        
        # projections
        self.fc_q = nn.Linear(hdim, hdim)
        self.fc_k = nn.Linear(hdim, hdim)
        self.fc_v = nn.Linear(hdim, hdim)
        self.fc_x0_2 = nn.Linear(hdim, hdim)
        self.W_O  = nn.Linear(hdim, hdim)
        for m in [self.fc_q, self.fc_k, self.fc_v,self.fc_x0_2, self.W_O]:
            init_linear_layer(m)

        # scale for dot-product path (kept for completeness)
        self.scale = (self.vdim) ** (-0.5)

        # shared random features (per layer, across heads)
        self.rff = SharedRFF(self.vdim, rff_m, lengthscale)
    
    
    def _qkv(self, x, asym=True):
        # x: [B, N, hdim] -> q,k,v: [B,H,N,D]
        q = self.fc_q(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
        x0 = self.fc_x0_2(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
        if asym:
            k = self.fc_k(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
        else:
            k = q.clone()
        v = self.fc_v(x).view(x.shape[0], x.shape[1], self.num_heads, self.vdim).permute(0,2,1,3)
        return q, k, v, x0


    def _cgp_lower_bounds(self, Phi_q, Phi_k, Phi_o, y):
        """
        LB_q, LB_k: Jensen lower bounds for log E[p(z_q = y | z_o)] and log E[p(z_k | z_o)].
        Computed using only m×m solves.

        Phi_*: [B,H,N,m],  y: [B,H,N,D]
        returns: LB_q, LB_k with shape [B,H]
        """
        B, H, N, m = Phi_q.shape
        D = y.shape[-1]
        lam = self.lambda_noise
        eps = self.cgp_eps
        dev = Phi_q.device

        I_m = torch.eye(m, device=dev).view(1,1,m,m)

        # Feature-space Grams
        G  = torch.matmul(Phi_o.transpose(-1, -2), Phi_o)   # [B,H,m,m]
        Cq = torch.matmul(Phi_q.transpose(-1, -2), Phi_q)   # [B,H,m,m]
        Ck = torch.matmul(Phi_k.transpose(-1, -2), Phi_k)   # [B,H,m,m]
        A  = G + lam * I_m                                  # [B,H,m,m]

        # A^{-1}
        L_A  = torch.linalg.cholesky(A)                     # [B,H,m,m]
        Iexp = I_m.expand(B, H, m, m).contiguous()
        A_inv = torch.cholesky_solve(Iexp.reshape(B*H,m,m), L_A.reshape(B*H,m,m)).reshape(B,H,m,m)

        # S_q = (1/lam)A + (1/eps)Cq  and same for k
        S_q = (1.0/lam) * A + (1.0/eps) * Cq
        S_k = (1.0/lam) * A + (1.0/eps) * Ck
        L_Sq = torch.linalg.cholesky(S_q)
        L_Sk = torch.linalg.cholesky(S_k)
        S_q_inv = torch.cholesky_solve(Iexp.reshape(B*H,m,m), L_Sq.reshape(B*H,m,m)).reshape(B,H,m,m)
        S_k_inv = torch.cholesky_solve(Iexp.reshape(B*H,m,m), L_Sk.reshape(B*H,m,m)).reshape(B,H,m,m)

        # ---------- Query term ----------
        # Data term: (1/eps)||y||^2 - (1/eps^2) || S_q^{-1/2} (Phi_q^T y) ||_F^2
        yy = torch.sum(y*y, dim=(2,3))                                     # [B,H]
        PhiT_y = torch.matmul(Phi_q.transpose(-1, -2), y)                   # [B,H,m,D]
        Sq_inv_PhiT_y = torch.matmul(S_q_inv, PhiT_y)                       # [B,H,m,D]
        datafit_q = (1.0/eps) * yy - (1.0/(eps*eps)) * torch.sum(PhiT_y * Sq_inv_PhiT_y, dim=(2,3))

        # Model term: D * tr( G A^{-1} [ (1/eps)Cq - (1/eps^2) Cq S_q^{-1} Cq ] )
        Cq_Sinv_Cq = torch.matmul(torch.matmul(Cq, S_q_inv), Cq)
        M_q = (1.0/eps) * Cq - (1.0/(eps*eps)) * Cq_Sinv_Cq
        G_Ainv = torch.matmul(G, A_inv)
        model_q = D * torch.einsum('bhmn,bhmn->bh', G_Ainv, M_q)

        # logdet(Σ̃_q) = N log eps + log det( I + (lam/eps) A^{-1} Cq )
        Ainv_Cq = torch.matmul(A_inv, Cq)
        Iadd_q = I_m + (lam/eps) * Ainv_Cq
        _, logdet_Iq = torch.slogdet(Iadd_q)
        logdet_q = N * math.log(eps) + logdet_Iq

        LB_q = -0.5 * (datafit_q + model_q + logdet_q + N * math.log(2*math.pi))  # [B,H]

        # ---------- Key term ----------
        tr_Ck  = torch.diagonal(Ck, dim1=-2, dim2=-1).sum(-1)             # [B,H]
        Ck2 = torch.matmul(Ck, Ck)
        Skinv_Ck2 = torch.matmul(S_k_inv, Ck2)
        tr_part = torch.diagonal(Skinv_Ck2, dim1=-2, dim2=-1).sum(-1)
        datafit_k = D * ((1.0/eps) * tr_Ck - (1.0/(eps*eps)) * tr_part)

        Ck_Sinv_Ck = torch.matmul(torch.matmul(Ck, S_k_inv), Ck)
        M_k = (1.0/eps) * Ck - (1.0/(eps*eps)) * Ck_Sinv_Ck
        model_k = D * torch.einsum('bhmn,bhmn->bh', G_Ainv, M_k)

        Ainv_Ck = torch.matmul(A_inv, Ck)
        Iadd_k = I_m + (lam/eps) * Ainv_Ck
        _, logdet_Ik = torch.slogdet(Iadd_k)
        logdet_k = N * math.log(eps) + logdet_Ik

        LB_k = -0.5 * (datafit_k + model_k + logdet_k + N * math.log(2*math.pi))  # [B,H]
        return LB_q, LB_k
        
    def forward(self, x, cur_k, input_mask):
        # x: [B, N, hdim]
        # input_mask: [B, N] with 1 for valid, 0 for pad
        # returns: [B, 1, N, hdim], regularizer (scalar tensor or None)

        # quick masks
        B, N, _ = x.shape
        mask = input_mask.float()                         # [B, N]
        maskN_1 = mask.unsqueeze(1).unsqueeze(-1)         # [B, 1, N, 1]  (broadcast over H,D)
        attn_pair = mask[:, None, :, None] * mask[:, None, None, :]  # [B,1,N,N]

        if not self.flag_cgp:
            # Fallback: scaled dot-product attention with masking
            q, k, v, x0 = self._qkv(x, asym=True)                     # [B,H,N,D] each
            v = v * maskN_1                                           # zero out padded tokens at values

            logits = self.scale * torch.einsum('bhid,bhjd->bhij', q, k)  # [B,H,N,N]
            # mask: set invalid pairs to a large negative number
            logits = logits.masked_fill(attn_pair == 0, -1e9)

            attn = torch.softmax(logits, dim=-1)                      # [B,H,N,N]
            out = torch.einsum('bhij,bhjd->bhid', attn, v)            # [B,H,N,D]

            out = out.unsqueeze(2).permute(0,2,3,1,4).contiguous().view(B, 1, N, self.hdim)
            return self.W_O(out), None

        # --------------------------- RFF-CGPT path (with masks) ---------------------------
        lam = self.lambda_noise

        q, k, v, x0 = self._qkv(x, asym=True)     # [B,H,N,D]
        # mask values and later Φ features
        v = v * maskN_1

        # random feature maps
        Phi_q = self.rff(q)                        # [B,H,N,m]
        Phi_k = self.rff(k)                        # [B,H,N,m]
        Phi_o = self.rff(x0)                       # [B,H,N,m]

        # zero out padded tokens in feature space
        Phi_q = Phi_q * maskN_1
        Phi_k = Phi_k * maskN_1
        Phi_o = Phi_o * maskN_1

        # small Gram matrices (m×m), summed over tokens N
        Gk = torch.einsum('bhnm,bhnp->bhmp', Phi_k, Phi_k)   # [B,H,m,m]
        Go = torch.einsum('bhnm,bhnp->bhmp', Phi_o, Phi_o)   # [B,H,m,m]

        B, H, N, D = v.shape
        m = Gk.shape[-1]
        device = x.device

        Ik = torch.eye(m, device=device).view(1,1,m,m).expand(B, H, m, m)
        Io = torch.eye(m, device=device).view(1,1,m,m).expand(B, H, m, m)

        Ak = Gk + lam * Ik                                  # [B,H,m,m]
        Ao = Go + lam * Io                                  # [B,H,m,m]

        # batched Cholesky (flatten BH to make cholesky_solve happy)
        Lk = torch.linalg.cholesky(Ak)                      # [B,H,m,m]
        Lo = torch.linalg.cholesky(Ao)                      # [B,H,m,m]
        Lk_flat = Lk.reshape(B*H, m, m)
        Lo_flat = Lo.reshape(B*H, m, m)

        # Φ_k^T v  -> [B,H,m,D]
        Phi_k_T_v = torch.einsum('bhnm,bhnd->bhmd', Phi_k, v)
        rhs = Phi_k_T_v.reshape(B*H, m, D)
        Ak_inv_Phi_k_T_v = torch.cholesky_solve(rhs, Lk_flat).reshape(B, H, m, D)

        # u = (v - Φ_k Ak^{-1} Φ_k^T v) / λ   -> [B,H,N,D]
        proj = torch.einsum('bhnm,bhmd->bhnd', Phi_k, Ak_inv_Phi_k_T_v)
        u = (v - proj) / lam

        # Ao^{-1}
        I_flat = torch.eye(m, device=device).expand(B*H, m, m).contiguous()
        Ao_inv = torch.cholesky_solve(I_flat, Lo_flat).reshape(B, H, m, m)

        # B_o = (Φ_o^T Φ_o) Ao^{-1} = Go Ao^{-1}
        B_o = torch.einsum('bhmp,bhpq->bhmq', Go, Ao_inv)   # [B,H,m,m]

        # y = Φ_q B_o (Φ_k^T u)
        Phi_k_T_u = torch.einsum('bhnm,bhnd->bhmd', Phi_k, u)      # [B,H,m,D]
        mid = torch.einsum('bhmq,bhqd->bhmd', B_o, Phi_k_T_u)      # [B,H,m,D]
        y = torch.einsum('bhnm,bhmd->bhnd', Phi_q, mid)            # [B,H,N,D]

        # assemble heads and project
        out = y.unsqueeze(2).permute(0,2,3,1,4).contiguous().view(B, 1, N, self.hdim)
        out = self.W_O(out)

        # ----------------------------- regularizer -----------------------------
        if self.loss_impl == 'cgp':
            # Lower bounds in feature space (per-head); average to avoid scaling with H
            LB_q, LB_k = self._cgp_lower_bounds(Phi_q, Phi_k, Phi_o, v)  # tensors [B,H] each
            reg = -(LB_q + LB_k).mean()
        else:
            # Feature-space GP log-likelihood (masked)
            # log|K+λI| = (N_eff - m) log λ + log|A|, with A = Φ^TΦ + λI
            Neff = mask.sum(dim=-1, keepdim=True).expand(B, H)          # [B,H]
            logdet_Ak = 2.0 * torch.sum(torch.log(torch.diagonal(Lk, dim1=-2, dim2=-1)), dim=-1)  # [B,H]
            loglike_det = (Neff - m) * math.log(lam) + logdet_Ak

            zk_sq = torch.sum(v * v, dim=(2,3))                         # [B,H]
            quad_corr = torch.sum(Phi_k_T_v * Ak_inv_Phi_k_T_v, dim=(2,3))  # [B,H]
            quad = (1.0 / lam) * (zk_sq - quad_corr)                    # [B,H]

            reg = 0.5 * (loglike_det + quad + Neff * math.log(2 * math.pi)).mean()

        return out, reg
