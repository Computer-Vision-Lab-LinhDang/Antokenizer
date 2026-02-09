"""Transition matrices for discrete diffusion (D3PM)."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


class TransitionMatrix:
    """Manages discrete diffusion transition matrices for D3PM.

    Implements two main transition types:
    1. Uniform (random replacement): With probability beta, replace current
       token with a uniform random token from vocabulary.
    2. Absorbing (mask): With probability beta, transition to a special [MASK]
       token that acts as an absorbing state.

    The absorbing transition is generally preferred for generation tasks as it
    allows the model to focus on predicting masked positions.
    """

    def __init__(
        self,
        vocab_size: int,
        num_timesteps: int = 1000,
        transition_type: Literal["uniform", "absorbing"] = "absorbing",
        schedule: Optional["NoiseSchedule"] = None,
    ) -> None:
        """Initialize transition matrix.

        Args:
            vocab_size: Size of discrete vocabulary K.
            num_timesteps: Total diffusion timesteps T.
            transition_type: "uniform" or "absorbing".
            schedule: NoiseSchedule instance for beta values.
        """
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.transition_type = transition_type

        # For absorbing, we add [MASK] token at index vocab_size
        self.mask_token_id = vocab_size if transition_type == "absorbing" else None
        self.full_vocab_size = vocab_size + 1 if transition_type == "absorbing" else vocab_size

        if schedule is None:
            from .schedule import NoiseSchedule
            schedule = NoiseSchedule(num_timesteps=num_timesteps, schedule_type="cosine")
        self.schedule = schedule

        # Precompute cumulative transition matrices
        self._precompute_transitions()

    def _precompute_transitions(self) -> None:
        """Precompute Q_t and Q_bar_t matrices for efficiency."""
        betas = self.schedule.get_betas()
        K = self.full_vocab_size

        # Store log probabilities for numerical stability
        self.log_Qt = []
        self.log_Qt_bar = []

        log_Qt_bar_accum = torch.zeros(K, K)
        torch.fill_(log_Qt_bar_accum.diagonal(), 0.0)  # log(1) = 0

        for t in range(self.num_timesteps):
            beta_t = betas[t].item()

            if self.transition_type == "uniform":
                # Q_t[i,j] = (1-beta) if i==j else beta/K
                Qt = torch.full((K, K), beta_t / K)
                Qt.fill_diagonal_(1.0 - beta_t + beta_t / K)
            else:  # absorbing
                # Q_t[i,j]:
                #   - If j == MASK: beta_t (probability to become masked)
                #   - If i == j and j != MASK: 1 - beta_t (stay same)
                #   - If i == MASK and j == MASK: 1 (absorbing state)
                #   - Otherwise: 0
                Qt = torch.zeros(K, K)
                # Non-mask tokens can stay same or become masked
                for i in range(self.vocab_size):
                    Qt[i, i] = 1.0 - beta_t  # Stay same
                    Qt[i, self.mask_token_id] = beta_t  # Become masked
                # Mask token is absorbing
                Qt[self.mask_token_id, self.mask_token_id] = 1.0

            log_Qt = torch.log(Qt.clamp(min=1e-30))
            self.log_Qt.append(log_Qt)

            # Q_bar_t = Q_1 @ Q_2 @ ... @ Q_t
            # In log space: log(Q_bar_t) = log(Q_bar_{t-1} @ Q_t)
            if t == 0:
                log_Qt_bar_accum = log_Qt.clone()
            else:
                # Matrix multiply in log space using logsumexp
                log_Qt_bar_accum = self._log_matmul(log_Qt_bar_accum, log_Qt)

            self.log_Qt_bar.append(log_Qt_bar_accum.clone())

        self.log_Qt = torch.stack(self.log_Qt)  # (T, K, K)
        self.log_Qt_bar = torch.stack(self.log_Qt_bar)  # (T, K, K)

    def _log_matmul(self, log_A: torch.Tensor, log_B: torch.Tensor) -> torch.Tensor:
        """Matrix multiplication in log space using logsumexp."""
        # log(A @ B)[i,j] = logsumexp_k(log_A[i,k] + log_B[k,j])
        K = log_A.size(0)
        result = torch.zeros(K, K)
        for i in range(K):
            for j in range(K):
                result[i, j] = torch.logsumexp(log_A[i, :] + log_B[:, j], dim=0)
        return result

    def get_Qt(
        self,
        t: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get transition matrix Q_t for timestep t.

        Args:
            t: Timestep indices (B,).
            device: Target device.

        Returns:
            Q_t matrices (B, K, K) in probability space.
        """
        log_Qt = self.log_Qt.to(device or t.device)
        return torch.exp(log_Qt[t])  # (B, K, K)

    def get_Qt_bar(
        self,
        t: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Get cumulative transition matrix Q_bar_t = Q_1 @ ... @ Q_t.

        Args:
            t: Timestep indices (B,).
            device: Target device.

        Returns:
            Q_bar_t matrices (B, K, K) in probability space.
        """
        log_Qt_bar = self.log_Qt_bar.to(device or t.device)
        return torch.exp(log_Qt_bar[t])  # (B, K, K)

    def q_sample(
        self,
        x_0: torch.LongTensor,
        t: torch.Tensor,
    ) -> torch.LongTensor:
        """Sample x_t given x_0 using forward diffusion.

        q(x_t | x_0) = Cat(x_t; p = x_0 @ Q_bar_t)

        Args:
            x_0: Original tokens (B, N).
            t: Timestep indices (B,).

        Returns:
            Noisy tokens x_t (B, N).
        """
        B, N = x_0.shape
        device = x_0.device

        # Get transition probabilities: p[b, n, k] = Q_bar_t[x_0[b,n], k]
        Qt_bar = self.get_Qt_bar(t, device)  # (B, K, K)

        # Index into Q_bar: for each position, get row corresponding to x_0
        x_0_flat = x_0.view(-1)  # (B*N,)
        t_expanded = t.unsqueeze(1).expand(-1, N).reshape(-1)  # (B*N,)

        # Get transition probabilities for each token
        Qt_bar_flat = self.log_Qt_bar.to(device)[t_expanded]  # (B*N, K, K)

        # Index rows by x_0
        log_probs = Qt_bar_flat[torch.arange(B * N, device=device), x_0_flat]  # (B*N, K)
        probs = torch.exp(log_probs)

        # Sample from categorical
        x_t_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
        x_t = x_t_flat.view(B, N)

        return x_t

    def q_posterior(
        self,
        x_t: torch.LongTensor,
        x_0: torch.LongTensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute posterior q(x_{t-1} | x_t, x_0).

        Using Bayes rule:
        q(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) * q(x_{t-1} | x_0)

        Args:
            x_t: Noisy tokens at time t (B, N).
            x_0: Original tokens (B, N).
            t: Timestep indices (B,).

        Returns:
            Posterior probabilities (B, N, K).
        """
        B, N = x_t.shape
        device = x_t.device
        K = self.full_vocab_size

        # Handle t=0 case
        t_is_zero = (t == 0).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        # Get log Q_t and log Q_bar_{t-1}
        log_Qt = self.log_Qt.to(device)
        log_Qt_bar = self.log_Qt_bar.to(device)

        # q(x_t | x_{t-1}): log_Qt[t][x_{t-1}, x_t]
        # q(x_{t-1} | x_0): log_Qt_bar[t-1][x_0, x_{t-1}]

        # Expand for all possible x_{t-1} values
        x_t_expanded = x_t.unsqueeze(-1).expand(-1, -1, K)  # (B, N, K)
        x_0_expanded = x_0.unsqueeze(-1).expand(-1, -1, K)  # (B, N, K)

        # Get log probabilities
        t_clamped = t.clamp(min=1)  # For indexing Qt (t >= 1)
        t_prev = (t - 1).clamp(min=0)  # For indexing Qt_bar (t-1 >= 0)

        # log q(x_t | x_{t-1} = k) for all k
        # Shape: (B, K, K) -> index by [t, k, x_t] -> (B, N, K)
        log_Qt_t = log_Qt[t_clamped]  # (B, K, K)

        # Gather: log_Qt_t[b, k, x_t[b,n]] for all k
        log_q_xt_given_xtm1 = torch.zeros(B, N, K, device=device)
        for b in range(B):
            for n in range(N):
                log_q_xt_given_xtm1[b, n] = log_Qt_t[b, :, x_t[b, n]]

        # log q(x_{t-1} = k | x_0) for all k
        log_Qt_bar_prev = log_Qt_bar[t_prev]  # (B, K, K)

        log_q_xtm1_given_x0 = torch.zeros(B, N, K, device=device)
        for b in range(B):
            for n in range(N):
                log_q_xtm1_given_x0[b, n] = log_Qt_bar_prev[b, x_0[b, n], :]

        # Posterior: log q(x_{t-1} | x_t, x_0) ∝ log_q_xt_given_xtm1 + log_q_xtm1_given_x0
        log_posterior_unnorm = log_q_xt_given_xtm1 + log_q_xtm1_given_x0

        # Normalize
        log_posterior = log_posterior_unnorm - torch.logsumexp(log_posterior_unnorm, dim=-1, keepdim=True)
        posterior = torch.exp(log_posterior)

        # At t=0, posterior should be delta at x_0
        x_0_onehot = F.one_hot(x_0, num_classes=K).float()
        posterior = torch.where(t_is_zero, x_0_onehot, posterior)

        return posterior

    def get_mask_token_id(self) -> int:
        """Get the mask token ID (for absorbing transition)."""
        if self.mask_token_id is None:
            raise ValueError("Mask token only available for absorbing transition")
        return self.mask_token_id


__all__ = ["TransitionMatrix"]
