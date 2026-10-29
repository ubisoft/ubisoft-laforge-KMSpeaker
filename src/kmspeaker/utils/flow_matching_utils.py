"""
Conditional-OT flow matching utilities (path sampling, velocity conversion, ODE solving).

Derived directly from the conditional-OT (a.k.a. linear / rectified-flow) interpolation

    x_t = (1 - t) * x_0 + t * x_1,  t in [0, 1]

between a source sample x_0 and a target sample x_1 (Lipman et al., "Flow Matching
Guide and Code", arXiv:2412.06264, section 2).

The velocity-from-prediction formula in
`CondOTProbPath.velocity_from_x1` is obtained by solving the interpolation
equation for x_0 and substituting into dx_t/dt.
"""

from typing import Callable, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torchdiffeq import odeint


class CondOTProbPath:
    """
    The straight-line path between a source sample x_0 and a target sample x_1:

        x_t = (1 - t) * x_0 + t * x_1

    Differentiating with respect to t gives a drift that is constant along any single
    (x_0, x_1) pair:

        dx_t/dt = x_1 - x_0
    """

    @staticmethod
    def _as_column(t: Tensor, like: Tensor) -> Tensor:
        """Reshape a length-batch time vector to broadcast against `like`'s trailing dims."""
        return t.reshape(t.shape[0], *([1] * (like.ndim - 1)))

    def interpolate(self, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (x_t, dx_t): the point on the path at time t and its velocity."""
        assert t.ndim == 1, f"t must be a 1-D batch of times, got shape {t.shape}"
        assert x_0.shape[0] == x_1.shape[0] == t.shape[0], (
            f"x_0, x_1 and t must share a batch dimension; got "
            f"x_0={x_0.shape}, x_1={x_1.shape}, t={t.shape}"
        )

        t_col = self._as_column(t, x_1)
        x_t = (1 - t_col) * x_0 + t_col * x_1
        dx_t = x_1 - x_0

        return x_t, dx_t

    def velocity_from_x1(self, x_1: Tensor, x_t: Tensor, t: Tensor) -> Tensor:
        """
        Recover dx_t/dt at (x_t, t) from a predicted endpoint x_1, without knowing x_0.

        From x_t = (1 - t) * x_0 + t * x_1, the implied source point is
        x_0 = (x_t - t * x_1) / (1 - t). Substituting into dx_t/dt = x_1 - x_0 and
        simplifying gives the one-line formula below.
        """
        t_col = self._as_column(t, x_t) if t.ndim >= 1 else t
        return (x_1 - x_t) / (1 - t_col)


class ModelWrapper(nn.Module):
    """Adapts a `forward(x, t, **kwargs)` model to the velocity-field signature `ODESolver` expects."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        return self.model(x=x, t=t, **kwargs)


class ODESolver:
    """Integrates dx/dt = velocity_model(x, t) forward in time with `torchdiffeq.odeint`."""

    def __init__(self, velocity_model: Union[ModelWrapper, Callable]):
        self.velocity_model = velocity_model

    def sample(
        self,
        initial_noise: Tensor,
        integration_schedule: Tensor,
        *,
        solver_method: str,
        fixed_step_size: Optional[float] = None,
        rel_tolerance: float = 1e-5,
        abs_tolerance: float = 1e-5,
        **velocity_kwargs,
    ) -> Tensor:
        integration_schedule = integration_schedule.to(initial_noise.device)
        step_opts = {} if fixed_step_size is None else {"step_size": fixed_step_size}

        def velocity_at(t: Tensor, x: Tensor) -> Tensor:
            return self.velocity_model(x=x, t=t, **velocity_kwargs)

        with torch.no_grad():
            trajectory = odeint(
                velocity_at,
                initial_noise,
                integration_schedule,
                method=solver_method,
                options=step_opts,
                atol=abs_tolerance,
                rtol=rel_tolerance,
            )

        return trajectory[-1]
