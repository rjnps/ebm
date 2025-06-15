import torch
import torch.nn as nn
import numpy as np


class DMP:
    def __init__(self, args):
        self.cfg = args
        self.c = self._init_centers()  # Centers of basis functions
        self.sigma_sq = self._init_sigmas()
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
        if weights.dim() > 1:
            weights = weights.squeeze()

        # activations for all the basis gaussians
        psi = torch.exp(-0.5 * ((x - self.c)**2) / self.sigma_sq)

        sum_psi = torch.sum(psi)
        if sum_psi < 1e-10:
            # return zero forcing function, to avoid division by zero
            return torch.zeros_like(x)

        f = torch.dot(psi, weights) * x * (g - y0) / sum_psi
        return f

    def integrate(self, g, weights, y_0, y_dot_0):
        positions = [y_0]
        velocities = [y_dot_0]
        accelerations = [torch.tensor(0.0, device=y_0.device, dtype=torch.float)]

        y = y_0.clone().detach()  # Position
        dy = y_dot_0.clone().detach()  # Velocity
        ddy = torch.tensor(0.0, device=y_0.device, dtype=torch.float)

        x = torch.tensor(1.0, device=y_0.device, dtype=torch.float)

        for _ in range(self.trajectory_length):
            dx = self._canonical_system(x)
            x = x + dx * self.cfg.dt

            f = self._forcing_term(x, weights, y_0, g)

            ddy = (self.cfg.alpha * (self.cfg.beta * (g - y) - dy) + f) / self.cfg.tau
            dy = dy + ddy * self.cfg.dt
            y = y + dy * self.cfg.dt

            positions.append(y)
            velocities.append(dy)
            accelerations.append(ddy)

        return torch.stack(positions), torch.stack(velocities), torch.stack(accelerations)
