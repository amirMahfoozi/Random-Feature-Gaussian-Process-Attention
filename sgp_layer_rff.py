import math
import torch
import torch.nn as nn

def _qr_orthogonalize(mat):  # mat: [H, D, M_half]
    """
    Return an orthonormal column matrix per head with shape [H, D, M_half].
    We QR-decompose the [D x M_half] matrix directly (not transposed).
    If M_half > D, we fall back to transposed QR and crop.
    """
    H, D, M = mat.shape
    outs = []
    for h in range(H):
        A = mat[h]  # [D, M]
        if D >= M:
            # QR on D x M -> Q is D x M with orthonormal columns
            Q, _ = torch.linalg.qr(A, mode='reduced')  # [D, M]
            outs.append(Q[:, :M])
        else:
            # rare case: more features than dims; orthogonalize rows then crop
            Q, _ = torch.linalg.qr(A.T, mode='reduced')  # [M, D]
            outs.append(Q[:, :D].T)  # [D, min(D, M)] => [D, D]; we'll rely on M==D here
    return torch.stack(outs, dim=0)  # [H, D, M]



class SGP_LAYER(nn.Module):
    """
    RFF-based GP Attention for ViT (drop-in):
      mean:  Φ_Q (Φ_K^T Φ_K + σ^2 I)^{-1} Φ_K^T V
      var :  σ^2 * diag(Φ_Q (Φ_K^T Φ_K + σ^2 I)^{-1} Φ_Q^T)
      KL  :  exact feature-space KL with Σ = σ^2 (Φ_K^T Φ_K + σ^2 I)^{-1}
    """
    def __init__(self, device, num_heads, max_len, hdim, kernel_type,
                 sample_size, jitter, keys_len, drop_rate, flag_sgp, inference_mode):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.max_len = max_len
        self.hdim = hdim
        self.vdim = hdim // num_heads
        self.sample_size = max(int(sample_size), 1)
        self.jitter = jitter if isinstance(jitter, float) else 1e-6
        self.drop_rate = drop_rate
        self.flag_sgp = flag_sgp
        self.inference_mode = inference_mode

        if kernel_type not in ("ard", "exponential"):
            raise ValueError("kernel_type must be 'ard' or 'exponential'")
        self.kernel_type = kernel_type

        # projections (self-attn: W_q = W_k implicitly)
        self.fc_qv = nn.Linear(hdim, 2 * num_heads * self.vdim, bias=False)
        self.W_O = nn.Sequential(nn.Linear(hdim, hdim), nn.Dropout(drop_rate))

        # GP hyperparams
        self.log_sf = nn.Parameter(torch.zeros(num_heads, 1))           # per-head amplitude σ_f^2
        self.log_ls = nn.Parameter(torch.zeros(num_heads, self.vdim))   # per-head ARD lengthscales
        self.log_sigma2 = nn.Parameter(torch.tensor(-2.0))              # observation noise σ^2

        # --- Random Fourier Features ---
        # Use cos+sin -> effective feature count is 2 * M_half
        self.M_half = keys_len                       # half-bank (cos/sin paired)
        self.M_eff = 2 * self.M_half                 # effective features per head
        # base Gaussian frequencies + QR-orthogonalization to reduce variance
        omega_half = torch.randn(num_heads, self.vdim, self.M_half)
        omega_half = _qr_orthogonalize(omega_half)
        self.register_buffer("omega_half", omega_half)                  # [H, D, M_half]
        # random phases for cos/sin are not needed when using explicit sin/cos pairs
        # but we keep a tiny bias for symmetry breaking
        self.register_buffer("phase", 2 * math.pi * torch.rand(num_heads, self.M_half))

    # ---------- features ----------
    def _phi(self, q):
        """
        q: [B, H, L, D]  ->  phi: [B, H, L, 2*M_half]  (cos ⊕ sin)
        NOTE: ViT already applies LayerNorm before this layer; DO NOT re-normalize q here.
        """
        inv_ls = torch.exp(-self.log_ls)                                  # [H, D]
        if self.kernel_type == "ard":                                     # RBF
            scale = math.sqrt(2.0) * inv_ls                               # √2 / ℓ
        else:                                                             # Laplace (approx)
            scale = inv_ls
        omega = self.omega_half * scale.unsqueeze(-1)                     # [H, D, M_half]
        # project
        proj = torch.einsum("bhld,hdm->bhlm", q, omega) + self.phase.unsqueeze(0).unsqueeze(2)
        c = torch.cos(proj)                                               # [B,H,L,M_half]
        s = torch.sin(proj)
        phi = torch.cat([c, s], dim=-1)                                   # [B,H,L,2*M_half]
        # amplitude so that E[phi phi^T] ≈ σ_f^2 I
        amp = torch.exp(self.log_sf).sqrt().unsqueeze(0).unsqueeze(2)     # [1,H,1,1]
        phi = amp * phi * (1.0 / math.sqrt(self.M_eff))
        return phi

    # ---------- forward ----------
    def forward(self, x, _cur_k_ignored):
        """
        returns: samples [B, S, L, H*D], KL scalar (None in inference)
        """
        # q, v
        q, v = self.fc_qv(x).view(x.shape[0], x.shape[1], self.num_heads, 2 * self.vdim) \
                           .permute(0, 2, 1, 3).chunk(2, dim=-1)  # [B,H,L,D] each

        # RFFs (self-attn: K≡Q)
        phi_q = self._phi(q)                                             # [B,H,L,M_eff]
        phi_k = phi_q

        B, H, L, Dv = v.shape
        M = self.M_eff
        device, dtype = x.device, x.dtype

        # A = Φ_K^T Φ_K + σ^2 I_M
        G = torch.einsum("bhlm,bhln->bhmn", phi_k, phi_k)                # [B,H,M,M]
        I = torch.eye(M, device=device, dtype=dtype).view(1, 1, M, M)
        sigma2 = torch.exp(self.log_sigma2)
        A = G + sigma2 * I

        # Cholesky with jitter backoff
        jitter = max(self.jitter, 1e-6)
        for _ in range(4):
            try:
                R = torch.linalg.cholesky(A + jitter * I, upper=True)
                break
            except RuntimeError:
                jitter *= 10.0
        else:
            R = torch.linalg.cholesky(A + 1e-2 * I, upper=True)

        # whiten features: Φ̃ = Φ R^{-1}
        RT = R.transpose(-1, -2)
        tK = torch.linalg.solve_triangular(RT, phi_k.transpose(-1, -2), upper =False, left=True ).transpose(-1, -2)  # [B,H,L,M]
        tQ = torch.linalg.solve_triangular(RT, phi_q.transpose(-1, -2), upper =False, left=True).transpose(-1, -2)

        # mean: m = Φ̃_Q (Φ̃_K^T V)
        S = torch.matmul(tK.transpose(-2, -1), v)                        # [B,H,M,Dv]
        mean_feat = torch.matmul(tQ, S)                                   # [B,H,L,Dv]

        # variance per token/head
        var_scalar = sigma2 * (tQ ** 2).sum(dim=-1, keepdim=True)         # [B,H,L,1]
        var_diag = var_scalar.expand(-1, -1, -1, Dv)                      # [B,H,L,Dv]

        # sample
        Smp = self.sample_size
        eps = torch.randn(B, H, Smp, L, Dv, device=device, dtype=dtype)
        chol = var_diag.clamp_min(0).sqrt().unsqueeze(2)                  # [B,H,1,L,Dv]
        samples = mean_feat.unsqueeze(2) + chol * eps                     # [B,H,S,L,Dv]

        # merge heads + project
        samples = torch.flatten(samples.permute(0, 2, 3, 1, 4), -2, -1)   # [B,S,L,H*D]
        samples = self.W_O(samples)

        # KL(q(W)||p(W)) with Σ = σ^2 A^{-1}
        W_star = torch.linalg.solve_triangular(R, S, upper=True)          # [B,H,M,Dv]
        mu_norm2 = (W_star ** 2).sum(dim=(-2, -1))                        # [B,H]

        eyeM = I.expand_as(A)
        R_inv = torch.linalg.solve_triangular(R, eyeM, upper=True)        # [B,H,M,M]
        tr_Ainv = (R_inv ** 2).sum(dim=(-2, -1))                          # [B,H]

        logdetA = 2.0 * torch.log(torch.diagonal(R, dim1=-2, dim2=-1)).sum(dim=-1)  # [B,H]
        k = M * Dv
        KL_bh = 0.5 * ( Dv * sigma2 * tr_Ainv + mu_norm2 - k - Dv * ( M * torch.log(sigma2) - logdetA ) )
        kl = KL_bh.sum(dim=1).mean() if not self.inference_mode else None

        if self.inference_mode:
            return samples, None
        return samples, kl
