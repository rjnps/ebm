import torch


def langevin_dynamics(state_curr,
                      latent_curr,
                      cond_curr,
                      energy_model,
                      num_steps,
                      step_size_state,
                      step_size_latent,
                      noise_scale_state,
                      noise_scale_latent,
                      clip_grad_norm):

    """
    Takes in the energy model, state, latent and conditional tensors
    and performs MCMC
    """

    # clone and detach, so that we start a fresh chain
    state_t, latent_t = state_curr.clone().detach(), latent_curr.clone().detach()

    for lang_step in range(num_steps):
        state_t.requires_grad_(True)
        latent_t.requires_grad_(True)

        # todo: Modify this when you write code for energy models
        energy_curr = energy_model(state_curr, latent_curr, cond_curr)

        # computing gradients of energy w.r.t state and latent
        grad_state, grad_latent = torch.autograd.grad(energy_curr, (state_t, latent_t))

        # gradient clipping for stability
        if clip_grad_norm is not None:
            grad_state = torch.nn.utils.clip_grad_norm_([grad_state], clip_grad_norm)
            grad_latent = torch.nn.utils.clip_grad_norm_([grad_latent], clip_grad_norm)

        # langevin updates
        state_t = (state_t - (0.5 * step_size_state * grad_state) +
                   (noise_scale_state * torch.randn_like(state_t)))
        latent_t = (latent_t - (0.5 * step_size_latent * grad_latent) +
                    (noise_scale_latent * torch.randn_like(latent_t)))

        state_t = state_t.detach()
        latent_t = latent_t.detach()

    return state_t, latent_t

