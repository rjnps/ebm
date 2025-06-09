import torch
import torch.nn as nn
import torch.optim as optim
import random


# --- 1. Define the Energy Function (E_theta) ---
# This is a placeholder for your actual Energy-Based Model architecture.
# It should take 'a', 'b', and 'c' as input and output a scalar energy value.
# 'c' is assumed to be an integer ID representing the conditional context.
class EnergyFunction(nn.Module):
    def __init__(self, a_dim: int, b_dim: int, c_num_classes: int, hidden_dim: int = 128):
        super().__init__()
        self.c_num_classes = c_num_classes
        # A simple neural network for the energy function
        self.net = nn.Sequential(
            nn.Linear(a_dim + b_dim + c_num_classes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single scalar energy
        )

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Computes the energy E(a, b; c).
        Args:
            a: Observed variable tensor.
            b: Latent variable tensor.
            c: Conditioning variable tensor (assumed to be integer IDs).
        Returns:
            Scalar energy tensor for each sample in the batch.
        """
        # Convert c (integer ID) to one-hot encoding
        # This allows the network to learn distinct representations for each condition
        c_one_hot = torch.nn.functional.one_hot(c.long(), num_classes=self.c_num_classes).float()

        # Concatenate a, b, and the one-hot encoded c
        # Ensure c_one_hot is broadcastable if a, b are batched
        if a.dim() > 1 and a.shape > 1 and c_one_hot.dim() == 1:
            c_one_hot = c_one_hot.expand(a.shape, -1)

        combined_input = torch.cat([a, b, c_one_hot], dim=-1)
        return self.net(combined_input).squeeze(-1)  # Squeeze to get scalar energy per sample


# --- 2. Placeholder for f(b) (o derived from b) ---
# The user specified that 'o' is an output label derived from 'b'.
# This function defines that relationship. It's not directly part of the EBM's energy function
# but is relevant for understanding the data structure.
def derive_o_from_b(b: torch.Tensor) -> torch.Tensor:
    """
    Placeholder function to derive output label 'o' from latent variable 'b'.
    In a real application, this would be your specific mapping.
    """
    return b.mean(dim=-1)  # Example: mean of 'b' as a dummy 'o'


# --- 3. Langevin Dynamics Helper Function ---
def langevin_dynamics(
        current_a: torch.Tensor,
        current_b: torch.Tensor,
        current_c: torch.Tensor,
        energy_fn: EnergyFunction,
        num_steps: int,
        step_size_a: float,
        step_size_b: float,
        noise_scale_a: float,
        noise_scale_b: float,
        clip_grad_norm: float = None
) -> tuple:
    """
    Performs Langevin Dynamics steps to perturb (a, b) given c.
    This function simulates sampling from the EBM's distribution p(a, b | c).

    Args:
        current_a: Initial state of observed variable 'a'.
        current_b: Initial state of latent variable 'b'.
        current_c: The conditioning variable 'c' (fixed during MCMC).
        energy_fn: The energy function E_theta.
        num_steps: Number of Langevin steps to run.
        step_size_a: Step size for 'a' updates.
        step_size_b: Step size for 'b' updates.
        noise_scale_a: Scale of Gaussian noise for 'a' (typically sqrt(2 * step_size_a)).
        noise_scale_b: Scale of Gaussian noise for 'b' (typically sqrt(2 * step_size_b)).
        clip_grad_norm: Optional gradient clipping value to stabilize MCMC.

    Returns:
        (a_perturbed, b_perturbed): The final perturbed 'a' and 'b' tensors.
    """
    # Clone and detach to ensure we don't modify original tensors and start fresh chains
    a_t, b_t = current_a.clone().detach(), current_b.clone().detach()

    for _ in range(num_steps):
        # Enable gradient computation for current 'a' and 'b'
        a_t.requires_grad_(True)
        b_t.requires_grad_(True)

        # Compute energy for the current state (a_t, b_t, current_c)
        # .sum() is used because energy_fn returns a batch of energies, and we need a scalar for grad()
        energy = energy_fn(a_t, b_t, current_c).sum()

        # Compute gradients of energy w.r.t. a_t and b_t
        grad_a, grad_b = torch.autograd.grad(energy, (a_t, b_t))

        # Optional: Gradient clipping for stability
        if clip_grad_norm is not None:
            grad_a = torch.nn.utils.clip_grad_norm_([grad_a], clip_grad_norm)
            grad_b = torch.nn.utils.clip_grad_norm_([grad_b], clip_grad_norm)

        # Langevin updates: x_new = x_old - 0.5 * step_size * grad_E(x_old) + noise
        a_t = a_t - 0.5 * step_size_a * grad_a + noise_scale_a * torch.randn_like(a_t)
        b_t = b_t - 0.5 * step_size_b * grad_b + noise_scale_b * torch.randn_like(b_t)

        # Detach to prevent gradients from accumulating across Langevin steps
        a_t = a_t.detach()
        b_t = b_t.detach()

    return a_t, b_t


# --- 4. Main Training Loop Function ---
def train_ebm_with_combined_sampling(
        energy_model: EnergyFunction,
        optimizer: optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,  # Yields (a_batch, o_batch, c_batch)
        all_datasets: dict,  # Dictionary: {c_val: [(a_list, o_scalar),...]} for cross-conditional sampling
        a_dim: int, b_dim: int, c_num_classes: int,  # Dimensions for tensor creation
        num_epochs: int = 100,
        K_infer_b: int = 50,  # MCMC steps for inferring b for positive samples and neg1
        K_mcmc_joint: int = 10,  # MCMC steps for joint (a,b) perturbation for neg2
        step_size_a: float = 1e-3,
        step_size_b: float = 1e-3,
        step_size_b_infer: float = 1e-3,  # Step size for b inference MCMC
        noise_scale_a: float = None,  # Will be sqrt(2 * step_size_a) if None
        noise_scale_b: float = None,  # Will be sqrt(2 * step_size_b) if None
        noise_scale_b_infer: float = None,  # Will be sqrt(2 * step_size_b_infer) if None
        clip_grad_norm: float = 1.0,  # Optional gradient clipping for stability
        device: str = 'cpu'
):
    """
    Trains an Energy-Based Model using a combined negative sampling strategy.

    Args:
        energy_model: The EBM (EnergyFunction instance).
        optimizer: Optimizer for the EBM parameters.
        data_loader: DataLoader providing batches of (a, o, c) positive samples.
        all_datasets: A dictionary mapping c_val (int) to a list of (a_list, o_scalar) tuples.
                      Used for cross-conditional negative sampling.
        a_dim, b_dim, c_num_classes: Dimensions of a, b, and number of c classes.
        num_epochs: Number of training epochs.
        K_infer_b: Number of Langevin steps for inferring latent 'b' (inner MCMC).
        K_mcmc_joint: Number of Langevin steps for joint (a,b) perturbation.
        step_size_a, step_size_b, step_size_b_infer: Langevin step sizes.
        noise_scale_a, noise_scale_b, noise_scale_b_infer: Langevin noise scales.
        clip_grad_norm: Value for gradient clipping.
        device: 'cpu' or 'cuda'.
    """
    energy_model.to(device)

    # Set default noise scales if not explicitly provided (common practice for Langevin)
    if noise_scale_a is None: noise_scale_a = (2 * step_size_a) ** 0.5
    if noise_scale_b is None: noise_scale_b = (2 * step_size_b) ** 0.5
    if noise_scale_b_infer is None: noise_scale_b_infer = (2 * step_size_b_infer) ** 0.5

    # Get all unique conditional values for cross-conditional sampling
    all_c_values = list(all_datasets.keys())
    if not all_c_values:
        raise ValueError("`all_datasets` cannot be empty for cross-conditional sampling.")

    print("Starting EBM training with combined negative sampling...")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (a_pos_batch, o_pos_batch, c_pos_batch) in enumerate(data_loader):
            a_pos_batch = a_pos_batch.to(device)
            o_pos_batch = o_pos_batch.to(device)  # 'o' is not directly used in E_theta, but part of data
            c_pos_batch = c_pos_batch.to(device)  # c as integer IDs

            batch_size = a_pos_batch.shape

            # --- 1. Infer latent 'b' for positive samples ---
            # For each positive (a, c) pair, we need to find the 'b' that minimizes E(a, b; c).
            # This is typically done with MCMC. Initialize 'b' randomly or from a replay buffer.
            b_pos_inferred_batch = torch.randn(batch_size, b_dim, device=device)

            # Run MCMC to find optimal b for each positive (a, c) in the batch
            # This is an inner loop for b inference, often done per-sample or batched carefully
            # For simplicity, we'll run it per-sample here.
            for i in range(batch_size):
                a_p_single = a_pos_batch[i:i + 1]  # Keep batch dimension for Langevin
                c_p_single = c_pos_batch[i:i + 1]
                b_p_init_single = b_pos_inferred_batch[i:i + 1]

                # MCMC to find b_p_inferred = argmin_b E_theta(a_p, b, c_p)
                # Note: step_size_a and noise_scale_a are 0.0 as 'a' is fixed during 'b' inference
                _, b_pos_inferred_batch[i:i + 1] = langevin_dynamics(
                    current_a=a_p_single,
                    current_b=b_p_init_single,
                    current_c=c_p_single,
                    energy_fn=energy_model,
                    num_steps=K_infer_b,
                    step_size_a=0.0,
                    step_size_b=step_size_b_infer,
                    noise_scale_a=0.0,
                    noise_scale_b=noise_scale_b_infer,
                    clip_grad_norm=clip_grad_norm
                )

            # --- 2. Calculate Energy for Positive Samples (E_pos) ---
            # Re-enable gradients for b_pos_inferred_batch as it's part of the loss computation
            b_pos_inferred_batch.requires_grad_(True)
            e_pos = energy_model(a_pos_batch, b_pos_inferred_batch, c_pos_batch)

            # --- 3. Generate Negative Samples (Type 1: Cross-Conditional) ---
            # For each positive sample (a_p, o_p, c_p), we create a negative sample
            # by taking an (a_k, o_k) from a *different* c_neg_val, and inferring b_k_star
            # under the *original positive sample's condition* c_p.
            a_neg1_batch = torch.empty_like(a_pos_batch)
            b_neg1_batch = torch.empty_like(b_pos_inferred_batch)
            c_neg1_batch = torch.empty_like(c_pos_batch)  # This will be the positive c_p

            for i in range(batch_size):
                c_p_val = c_pos_batch[i].item()  # Current positive condition value

                # Select a different conditional context c_neg_val
                available_c_for_neg = [val for val in all_c_values if val != c_p_val]
                if not available_c_for_neg:
                    # Fallback: if only one 'c' exists, or no other 'c' has data,
                    # we use the same 'c'. This makes it less "cross-conditional" but prevents errors.
                    c_neg_val = c_p_val
                    # print(f"Warning: Only one 'c' value ({c_p_val}) or no other data available. Cross-conditional sampling will use same 'c'.")
                else:
                    c_neg_val = random.choice(available_c_for_neg)

                # Sample (a_k, o_k) from the dataset corresponding to c_neg_val
                if not all_datasets[c_neg_val]:
                    raise ValueError(
                        f"No data found for conditional value {c_neg_val}. Cannot perform cross-conditional sampling.")

                a_k_list, o_k_scalar = random.choice(all_datasets[c_neg_val])
                a_k_single = torch.tensor(a_k_list, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dim
                # o_k_scalar is not directly used in E_theta, but conceptually part of the sample

                # Infer b_k_star for a_k under the *positive sample's condition* c_p
                b_k_star_init = torch.randn(1, b_dim, device=device)  # Initialize b randomly

                _, b_k_star_single = langevin_dynamics(
                    current_a=a_k_single,
                    current_b=b_k_star_init,
                    current_c=c_pos_batch[i:i + 1],  # Condition on the positive sample's c
                    energy_fn=energy_model,
                    num_steps=K_infer_b,
                    step_size_a=0.0,  # Do not perturb 'a'
                    step_size_b=step_size_b_infer,
                    noise_scale_a=0.0,
                    noise_scale_b=noise_scale_b_infer,
                    clip_grad_norm=clip_grad_norm
                )
                a_neg1_batch[i] = a_k_single.squeeze(0)
                b_neg1_batch[i] = b_k_star_single.squeeze(0)
                c_neg1_batch[i] = c_pos_batch[i]  # The condition for neg1 is the positive sample's c

            # Calculate Energy for Negative Samples (Type 1)
            e_neg1 = energy_model(a_neg1_batch, b_neg1_batch, c_neg1_batch)

            # --- 4. Generate Negative Samples (Type 2: Joint MCMC Perturbation) ---
            # Start MCMC from positive samples (a_pos_batch, b_pos_inferred_batch)
            # and perturb jointly (a, b) while keeping c_pos_batch fixed.
            a_neg2_batch, b_neg2_batch = langevin_dynamics(
                current_a=a_pos_batch,
                current_b=b_pos_inferred_batch,
                current_c=c_pos_batch,  # Condition on the positive sample's c
                energy_fn=energy_model,
                num_steps=K_mcmc_joint,
                step_size_a=step_size_a,
                step_size_b=step_size_b,
                noise_scale_a=noise_scale_a,
                noise_scale_b=noise_scale_b,
                clip_grad_norm=clip_grad_norm
            )

            # Calculate Energy for Negative Samples (Type 2)
            e_neg2 = energy_model(a_neg2_batch, b_neg2_batch, c_pos_batch)

            # --- 5. Compute CD Loss ---
            # Loss = E_pos - E_neg1 - E_neg2 (push down positive energy, push up negative energies)
            loss = (e_pos - e_neg1 - e_neg2).mean()

            # --- 6. Optimization Step ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training finished.")


# --- Example Usage (Dummy Data and Model Setup) ---
if __name__ == "__main__":
    # Define dimensions for dummy data
    A_DIM = 10  # Dimension of observed variable 'a'
    B_DIM = 5  # Dimension of latent variable 'b'
    C_NUM_CLASSES = 3  # Number of unique conditional values (e.g., 0, 1, 2)
    BATCH_SIZE = 32
    NUM_SAMPLES_PER_C = 100  # Number of samples for each conditional context

    # Create dummy datasets for simulation
    # all_datasets: {c_val: [(a_list, o_scalar),...]}
    dummy_datasets = {}
    for c_val in range(C_NUM_CLASSES):
        dummy_datasets[c_val] =
        for _ in range(NUM_SAMPLES_PER_C):
            dummy_a = torch.randn(A_DIM).tolist()  # 'a' as a list for storage
            dummy_b_true = torch.randn(B_DIM)  # True 'b' (not directly used by EBM)
            dummy_o = derive_o_from_b(dummy_b_true).item()  # Dummy 'o' derived from 'b'
            dummy_datasets[c_val].append((dummy_a, dummy_o))

    # Create a dummy DataLoader for positive samples
    # In a real scenario, you would load your actual dataset here.
    all_positive_samples =
    for c_val, data_list in dummy_datasets.items():
        for a_val, o_val in data_list:
            all_positive_samples.append((
                torch.tensor(a_val, dtype=torch.float32),
                torch.tensor(o_val, dtype=torch.float32),
                torch.tensor(c_val, dtype=torch.float32)  # 'c' as float, will be cast to long
            ))


    # Simulate a DataLoader
    class DummyDataLoader:
        def __init__(self, samples: list, batch_size: int):
            self.samples = samples
            self.batch_size = batch_size
            self.num_batches = (len(samples) + batch_size - 1) // batch_size

        def __iter__(self):
            random.shuffle(self.samples)  # Shuffle samples for each epoch
            for i in range(self.num_batches):
                batch = self.samples[i * self.batch_size: (i + 1) * self.batch_size]
                # Stack tensors to form batches
                a_batch = torch.stack([s for s in batch])
                o_batch = torch.stack([s[1] for s in batch])
                c_batch = torch.stack([s[2] for s in batch])
                yield a_batch, o_batch, c_batch

        def __len__(self):
            return self.num_batches


    dummy_data_loader = DummyDataLoader(all_positive_samples, BATCH_SIZE)

    # Initialize the Energy-Based Model and Optimizer
    ebm_model = EnergyFunction(a_dim=A_DIM, b_dim=B_DIM, c_num_classes=C_NUM_CLASSES)
    optimizer = optim.Adam(ebm_model.parameters(), lr=1e-4)

    # Determine device (GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Train the EBM with the combined sampling strategy
    train_ebm_with_combined_sampling(
        energy_model=ebm_model,
        optimizer=optimizer,
        data_loader=dummy_data_loader,
        all_datasets=dummy_datasets,
        a_dim=A_DIM, b_dim=B_DIM, c_num_classes=C_NUM_CLASSES,
        num_epochs=5,  # Reduced number of epochs for a quick example
        K_infer_b=20,  # Fewer MCMC steps for faster dummy run
        K_mcmc_joint=5,  # Fewer MCMC steps for faster dummy run
        step_size_a=1e-2,
        step_size_b=1e-2,
        step_size_b_infer=1e-2,
        clip_grad_norm=1.0,
        device=device
    )