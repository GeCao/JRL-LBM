import torch


def UnionPhiObs(phi_obs1: torch.Tensor, phi_obs2: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha should be in the range [0,1].")
    if alpha == 1.0:
        result = torch.min(phi_obs1, phi_obs2)
    else:
        result = (
            1.0
            / (1.0 + alpha)
            * (phi_obs1 + phi_obs2 - torch.sqrt(phi_obs1 * phi_obs1 + phi_obs2 * phi_obs2 - 2 * alpha * phi_obs1 * phi_obs2))
        )
    
    return result