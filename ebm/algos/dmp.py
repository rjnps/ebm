import torch
import torch.nn as nn
import numpy as np


class DMP:
    def __init__(self, args):
        self.cfg = args
        self.c = self._init_centers().to("cuda")  # Centers of basis functions
        self.sigma_sq = self._init_sigmas().to("cuda")
        self.trajectory_length = int(self.cfg.tau/self.cfg.dt)

    def _init_centers(self):
        return torch.exp(-self.cfg.alpha_y *
                         torch.linspace(0, 1, self.cfg.num_basis_fns))

    def _init_sigmas(self):
        """"
        calculated for gaussian basis functions
        """
        sigmas = torch.zeros(self.cfg.num_basis_fns)
        for i in range(self.cfg.num_basis_fns - 1):
            sigmas[i] = (self.c[i] - self.c[i + 1]) ** 2 / (2 * np.log(1 / self.cfg.overlap_factor))
        sigmas[-1] = sigmas[-2]
        return sigmas

    def _canonical_system(self, x):
        """
        tau * x_dot = - alpha_y * x
        """
        return -self.cfg.alpha_y * x / self.cfg.tau

    def _forcing_term(self, x, weights, y0, g):
        """
        forcing term based on gaussian basis functions
        """
        # activations for all the basis gaussians

        x = x.unsqueeze(-1)
        centers = self.c.unsqueeze(-1)
        sigma_sqs = self.sigma_sq.unsqueeze(-1)

        psi = torch.exp(-0.5 * ((x.unsqueeze(-1) - centers)**2) / sigma_sqs)
        sum_psi = torch.sum(psi, dim=1, keepdim=True)
        sum_psi_safe = torch.where(sum_psi < 1e-10, torch.full_like(sum_psi, 1e-10), sum_psi)
        weighted_sum_psi = torch.sum(psi * weights, dim=1, keepdim=True)
        f = (weighted_sum_psi / sum_psi_safe) * x.unsqueeze(-1) * (g - y0).unsqueeze(1)

        return f.squeeze(-1)

    def integrate(self, g, weights, y_0, y_dot_0):
        positions_list = [y_0]
        velocities_list = [y_dot_0]
        accelerations_list = [torch.zeros_like(y_0, device=y_0.device, dtype=torch.float)]

        y = y_0.clone().detach()  # Position
        dy = y_dot_0.clone().detach()  # Velocity
        ddy = torch.zeros(y_0.shape[0], device=y_0.device, dtype=torch.float)

        x = torch.ones(y_0.shape[0], device=y_0.device, dtype=torch.float)

        for _ in range(self.trajectory_length):
            dx = self._canonical_system(x)
            x = x + dx * self.cfg.dt

            f = self._forcing_term(x, weights, y_0, g).squeeze(1)

            ddy = (self.cfg.alpha * (self.cfg.beta * (g - y) - dy) + f) / self.cfg.tau
            dy = dy + ddy * self.cfg.dt
            y = y + dy * self.cfg.dt

            positions_list.append(y)
            velocities_list.append(dy)
            accelerations_list.append(ddy)

        positions = torch.stack(positions_list).permute(1, 0, 2)
        velocities = torch.stack(velocities_list).permute(1, 0, 2)
        accelerations = torch.stack(accelerations_list).permute(1, 0, 2)

        return positions, velocities, accelerations
