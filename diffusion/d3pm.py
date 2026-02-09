"""Core D3PM (Discrete Denoising Diffusion Probabilistic Models) implementation."""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schedule import NoiseSchedule
from .transition import TransitionMatrix


class D3PM(nn.Module):
    """Discrete Denoising Diffusion Probabilistic Model.

    Implements the D3PM framework for discrete token diffusion, supporting
    both uniform and absorbing (mask) transition matrices.

    Key features:
    - Forward diffusion via discrete transition matrices
    - Reverse process parameterized to predict x_0 directly
    - Support for VLB and auxiliary denoising losses
    - Few-shot compatible (T=50 or fewer steps)

    Reference: "Structured Denoising Diffusion Models in Discrete State-Spaces"
    (Austin et al., 2021)
    """

    def __init__(
        self,
        vocab_size: int,
        num_timesteps: int = 100,
        transition_type: Literal["uniform", "absorbing"] = "absorbing",
        schedule_type: Literal["linear", "cosine", "sigmoid", "sqrt"] = "cosine",
        loss_type: Literal["vb", "ce", "hybrid"] = "hybrid",
        hybrid_lambda: float = 0.001,
        parametrization: Literal["x0", "xtm1"] = "x0",
    ) -> None:
        """Initialize D3PM.

        Args:
            vocab_size: Size of discrete vocabulary K (without mask token).
            num_timesteps: Total diffusion steps T.
            transition_type: "uniform" or "absorbing".
            schedule_type: Beta schedule type.
            loss_type: "vb" (variational bound), "ce" (cross-entropy), or "hybrid".
            hybrid_lambda: Weight for VB term in hybrid loss.
            parametrization: Predict "x0" or "xtm1" (x_{t-1}).
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.transition_type = transition_type
        self.loss_type = loss_type
        self.hybrid_lambda = hybrid_lambda
        self.parametrization = parametrization

        # Initialize schedule and transition matrix
        self.schedule = NoiseSchedule(
            num_timesteps=num_timesteps,
            schedule_type=schedule_type,
        )

        self.transition = TransitionMatrix(
            vocab_size=vocab_size,
            num_timesteps=num_timesteps,
            transition_type=transition_type,
            schedule=self.schedule,
        )

        # Full vocab includes mask token for absorbing
        self.full_vocab_size = self.transition.full_vocab_size
        self.mask_token_id = self.transition.mask_token_id

    def q_sample(
        self,
        x_0: torch.LongTensor,
        t: torch.Tensor,
    ) -> torch.LongTensor:
        """Sample x_t given x_0 (forward diffusion).

        Args:
            x_0: Original discrete tokens (B, N).
            t: Timestep indices (B,).

        Returns:
            Noisy tokens x_t (B, N).
        """
        return self.transition.q_sample(x_0, t)

    def q_posterior(
        self,
        x_t: torch.LongTensor,
        x_0: torch.LongTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute true posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_t: Noisy tokens (B, N).
            x_0: Original tokens (B, N).
            t: Timestep indices (B,).

        Returns:
            Posterior distribution (B, N, K).
        """
        return self.transition.q_posterior(x_t, x_0, t)

    def p_logits(
        self,
        model_output: torch.Tensor,
        x_t: torch.LongTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert model output to p(x_{t-1} | x_t) logits.

        If parametrization is "x0", model predicts x_0 and we compute
        the posterior. If "xtm1", model directly predicts x_{t-1}.

        Args:
            model_output: Model logits (B, N, K) predicting x_0 or x_{t-1}.
            x_t: Current noisy tokens (B, N).
            t: Timestep indices (B,).

        Returns:
            Log probabilities for x_{t-1} (B, N, K).
        """
        if self.parametrization == "x0":
            # Model predicts x_0, compute posterior
            x_0_probs = F.softmax(model_output, dim=-1)  # (B, N, K)

            # Clamp to valid vocab (exclude mask token from x_0 predictions)
            if self.mask_token_id is not None:
                x_0_probs = x_0_probs[..., :self.vocab_size]
                x_0_probs = x_0_probs / x_0_probs.sum(dim=-1, keepdim=True).clamp(min=1e-10)

            # Sample from predicted x_0 to compute posterior
            # For efficiency, we marginalize over predicted x_0 distribution
            p_xtm1 = self._marginalize_posterior(x_0_probs, x_t, t)
            return torch.log(p_xtm1.clamp(min=1e-30))
        else:
            # Model directly predicts x_{t-1}
            return F.log_softmax(model_output, dim=-1)

    def _marginalize_posterior(
        self,
        x_0_probs: torch.Tensor,
        x_t: torch.LongTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Marginalize posterior over predicted x_0 distribution.

        p(x_{t-1} | x_t) = sum_{x_0} p(x_0 | x_t) * q(x_{t-1} | x_t, x_0)

        Args:
            x_0_probs: Predicted x_0 distribution (B, N, vocab_size).
            x_t: Current tokens (B, N).
            t: Timestep (B,).

        Returns:
            Marginalized distribution (B, N, full_vocab_size).
        """
        B, N, V = x_0_probs.shape
        device = x_0_probs.device
        K = self.full_vocab_size

        # For each possible x_0 value, compute q(x_{t-1} | x_t, x_0)
        # Then weight by p(x_0) and sum

        p_xtm1 = torch.zeros(B, N, K, device=device)

        # This is expensive but correct - in practice use vectorized version
        # For efficiency, we use a batched approach

        # Get transition matrices
        Qt = self.transition.get_Qt(t, device)  # (B, K, K)
        Qt_bar_prev = self.transition.get_Qt_bar((t - 1).clamp(min=0), device)  # (B, K, K)

        # For each position, compute posterior weighted by x_0_probs
        for b in range(B):
            for n in range(N):
                xt_val = x_t[b, n].item()
                for x0_val in range(V):
                    # q(x_{t-1} | x_t, x_0) propto q(x_t | x_{t-1}) * q(x_{t-1} | x_0)
                    log_q_xt_xtm1 = torch.log(Qt[b, :, xt_val].clamp(min=1e-30))  # (K,)
                    log_q_xtm1_x0 = torch.log(Qt_bar_prev[b, x0_val, :].clamp(min=1e-30))  # (K,)

                    log_posterior = log_q_xt_xtm1 + log_q_xtm1_x0
                    posterior = F.softmax(log_posterior, dim=0)

                    p_xtm1[b, n] += x_0_probs[b, n, x0_val] * posterior

        return p_xtm1

    def p_sample(
        self,
        model_output: torch.Tensor,
        x_t: torch.LongTensor,
        t: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.LongTensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t).

        Args:
            model_output: Model logits (B, N, K).
            x_t: Current noisy tokens (B, N).
            t: Timestep indices (B,).
            temperature: Sampling temperature.

        Returns:
            Sampled tokens x_{t-1} (B, N).
        """
        log_probs = self.p_logits(model_output, x_t, t)

        if temperature != 1.0:
            log_probs = log_probs / temperature

        probs = F.softmax(log_probs, dim=-1)

        # Sample from distribution
        B, N, K = probs.shape
        probs_flat = probs.view(-1, K)
        x_tm1_flat = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        x_tm1 = x_tm1_flat.view(B, N)

        return x_tm1

    def p_losses(
        self,
        x_0: torch.LongTensor,
        t: torch.Tensor,
        model_output: torch.Tensor,
        x_t: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute D3PM training losses.

        Args:
            x_0: Ground truth tokens (B, N).
            t: Timestep indices (B,).
            model_output: Model logits (B, N, K).
            x_t: Noisy tokens (optional, computed if not provided).

        Returns:
            Dictionary with loss values.
        """
        B, N = x_0.shape
        device = x_0.device

        if x_t is None:
            x_t = self.q_sample(x_0, t)

        losses = {}

        if self.loss_type in ["ce", "hybrid"]:
            # Cross-entropy loss on x_0 prediction
            if self.parametrization == "x0":
                # Model predicts x_0 directly
                ce_loss = F.cross_entropy(
                    model_output.view(-1, model_output.size(-1)),
                    x_0.view(-1),
                    reduction="mean",
                )
            else:
                # Need to infer x_0 from x_{t-1} prediction
                # Use the posterior mean as target
                posterior = self.q_posterior(x_t, x_0, t)
                ce_loss = F.cross_entropy(
                    model_output.view(-1, model_output.size(-1)),
                    posterior.view(-1, posterior.size(-1)),
                    reduction="mean",
                )
            losses["ce_loss"] = ce_loss

        if self.loss_type in ["vb", "hybrid"]:
            # Variational bound loss
            # KL(q(x_{t-1} | x_t, x_0) || p(x_{t-1} | x_t))

            q_posterior = self.q_posterior(x_t, x_0, t)  # (B, N, K)
            log_p = self.p_logits(model_output, x_t, t)  # (B, N, K)

            # KL divergence
            log_q = torch.log(q_posterior.clamp(min=1e-30))
            kl = (q_posterior * (log_q - log_p)).sum(dim=-1)  # (B, N)

            # Mask out t=0 (no KL at t=0)
            t_mask = (t > 0).float().unsqueeze(-1)  # (B, 1)
            kl = (kl * t_mask).sum() / t_mask.sum().clamp(min=1)

            losses["vb_loss"] = kl

        # Compute total loss
        if self.loss_type == "ce":
            losses["loss"] = losses["ce_loss"]
        elif self.loss_type == "vb":
            losses["loss"] = losses["vb_loss"]
        else:  # hybrid
            losses["loss"] = losses["ce_loss"] + self.hybrid_lambda * losses["vb_loss"]

        return losses

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int],
        denoiser: nn.Module,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        temperature: float = 1.0,
        progress: bool = False,
    ) -> torch.LongTensor:
        """Generate samples via reverse diffusion.

        Args:
            shape: (batch_size, num_tokens).
            denoiser: Model that predicts p(x_0 | x_t, t, condition).
            condition: Optional conditioning tensor.
            device: Target device.
            temperature: Sampling temperature.
            progress: Whether to show progress bar.

        Returns:
            Generated discrete tokens (B, N).
        """
        B, N = shape
        device = device or next(denoiser.parameters()).device

        # Initialize with mask tokens (for absorbing) or random (for uniform)
        if self.transition_type == "absorbing":
            x_t = torch.full((B, N), self.mask_token_id, device=device, dtype=torch.long)
        else:
            x_t = torch.randint(0, self.vocab_size, (B, N), device=device)

        timesteps = range(self.num_timesteps - 1, -1, -1)
        if progress:
            try:
                from tqdm import tqdm
                timesteps = tqdm(timesteps, desc="Sampling")
            except ImportError:
                pass

        for t_val in timesteps:
            t = torch.full((B,), t_val, device=device, dtype=torch.long)

            # Get model prediction
            if condition is not None:
                model_output = denoiser(x_t, t, condition)
            else:
                model_output = denoiser(x_t, t)

            # Sample x_{t-1}
            if t_val > 0:
                x_t = self.p_sample(model_output, x_t, t, temperature=temperature)
            else:
                # At t=0, take argmax
                if self.parametrization == "x0":
                    x_t = model_output[..., :self.vocab_size].argmax(dim=-1)
                else:
                    x_t = model_output.argmax(dim=-1)

        return x_t


__all__ = ["D3PM"]
