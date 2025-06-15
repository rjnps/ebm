import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration Parameters ---
NUM_BASIS_FUNCTIONS = 25  # Number of Gaussian basis functions for the forcing term
DMP_DT = 0.01  # Time step for DMP integration
DMP_TAU = 1.0  # Time constant for DMP, controls duration of movement
DMP_ALPHA_Z = 25  # Gain for velocity term in DMP
DMP_BETA_Z = DMP_ALPHA_Z / 4  # Gain for position term in DMP
DMP_ALPHA_Y = 25  # Gain for position term in canonical system

TRAJECTORY_LENGTH = int(DMP_TAU / DMP_DT)  # Number of steps in a trajectory


# --- 1. Dynamic Movement Primitive (DMP) Class ---
class DMP:
    """
    A simple implementation of a Dynamic Movement Primitive (DMP)
    for a single degree of freedom.
    """

    def __init__(self, n_bfs=NUM_BASIS_FUNCTIONS, dt=DMP_DT, tau=DMP_TAU,
                 alpha_z=DMP_ALPHA_Z, beta_z=DMP_BETA_Z, alpha_y=DMP_ALPHA_Y):
        self.n_bfs = n_bfs
        self.dt = dt
        self.tau = tau
        self.alpha_z = alpha_z
        self.beta_z = beta_z
        self.alpha_y = alpha_y

        # Initialize canonical system parameters
        self.c = self._init_centers()  # Centers of basis functions
        self.sigma_sq = self._init_sigmas()  # Variances of basis functions

    def _init_centers(self):
        """Initializes the centers of the Gaussian basis functions."""
        # Centers are spaced logarithmically to cover the phase variable range
        # from 0 to 1 (or slightly more to ensure last basis function is active at end)
        return torch.exp(-self.alpha_y * torch.linspace(0, 1, self.n_bfs))

    def _init_sigmas(self):
        """Initializes the variances (widths) of the Gaussian basis functions."""
        # Sigmas are set to ensure sufficient overlap between basis functions
        # For a uniform distribution, sigma = (distance between centers) / 2
        sigmas = torch.zeros(self.n_bfs)
        for i in range(self.n_bfs - 1):
            sigmas[i] = (self.c[i] - self.c[i + 1]) ** 2 / (2 * np.log(1 / 0.7))  # Overlap factor 0.7
        sigmas[-1] = sigmas[-2]  # Last one same as second to last
        return sigmas

    def _canonical_system(self, x):
        """
        Integrates the canonical system.
        x_dot = -alpha_y * x / tau
        """
        return -self.alpha_y * x / self.tau

    def _forcing_term(self, x, weights, start_pos, goal_pos):
        """
        Calculates the forcing term 'f' based on Gaussian basis functions.
        f = (sum(psi_i * w_i) / sum(psi_i)) * x * (goal_pos - start_pos)
        """
        # Ensure weights are a 1D tensor
        if weights.dim() > 1:
            weights = weights.squeeze()

        # Calculate activation of each basis function (psi)
        psi = torch.exp(-0.5 * ((x - self.c) ** 2) / self.sigma_sq)

        # Avoid division by zero if all psi are very small (e.g., at end of trajectory)
        sum_psi = torch.sum(psi)
        if sum_psi < 1e-10:
            return torch.zeros_like(x)  # Return zero forcing term if no basis functions are active

        # Weighted sum of basis functions
        f = torch.dot(psi, weights) * x * (goal_pos - start_pos) / sum_psi
        return f

    def integrate(self, start_pos, start_vel, goal_pos, weights):
        """
        Integrates the DMP forward in time to generate a trajectory.

        Args:
            start_pos (torch.Tensor): Initial position.
            start_vel (torch.Tensor): Initial velocity.
            goal_pos (torch.Tensor): Target position (goal).
            weights (torch.Tensor): Weights for the forcing function.

        Returns:
            tuple: (positions, velocities, accelerations) as torch.Tensors
        """
        positions = [start_pos]
        velocities = [start_vel]
        accelerations = [torch.tensor(0.0)]  # Initialize with zero acceleration

        y = start_pos.clone().detach()  # Position
        dy = start_vel.clone().detach()  # Velocity
        ddy = torch.tensor(0.0)  # Acceleration

        x = torch.tensor(1.0)  # Canonical system phase variable, starts at 1, goes to 0

        # Loop through time steps
        for _ in range(TRAJECTORY_LENGTH - 1):
            # Update canonical system
            dx = self._canonical_system(x)
            x = x + dx * self.dt

            # Calculate forcing term
            f = self._forcing_term(x, weights, start_pos, goal_pos)

            # Update DMP equations for acceleration
            ddy = (self.alpha_z * (self.beta_z * (goal_pos - y) - dy) + f) / self.tau
            dy = dy + ddy * self.dt
            y = y + dy * self.dt

            positions.append(y)
            velocities.append(dy)
            accelerations.append(ddy)

        return torch.stack(positions), torch.stack(velocities), torch.stack(accelerations)


# --- 2. MLP for Predicting DMP Parameters ---
class DMPParameterMLP(nn.Module):
    """
    Multilayer Perceptron (MLP) to predict DMP parameters:
    - Goal position adjustment (relative to start or absolute)
    - Forcing function weights
    """

    def __init__(self, input_dim, num_basis_functions, output_scale_goal=10.0):
        super(DMPParameterMLP, self).__init__()
        self.num_basis_functions = num_basis_functions
        self.output_scale_goal = output_scale_goal  # To scale goal predictions

        # Define the network architecture
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)

        # Output layers: one for goal adjustment, one for weights
        # We predict a *goal adjustment* relative to some base goal,
        # or it could be an absolute goal depending on design.
        # For simplicity, let's predict the absolute goal and the weights directly.
        self.goal_output = nn.Linear(64, 1)  # Predicts the goal position
        self.weights_output = nn.Linear(64, num_basis_functions)  # Predicts forcing function weights

    def forward(self, x):
        """
        Forward pass of the MLP.
        Args:
            x (torch.Tensor): Input context for the MLP (e.g., initial state, task info).
        Returns:
            tuple: (predicted_goal, predicted_weights)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)

        # Apply a scaling to the goal output if needed (e.g., to match expected range)
        predicted_goal = self.goal_output(x) * self.output_scale_goal

        # Weights can be positive/negative, use tanh or sigmoid if you want specific ranges
        # For now, linear output for weights is fine as DMP handles scaling
        predicted_weights = self.weights_output(x)

        return predicted_goal.squeeze(-1), predicted_weights  # Squeeze goal to make it 1D


# --- 3. Forward Process: Predicting Actions (Positions) ---
def forward_process(mlp_model, dmp_instance, mlp_input, initial_pos, initial_vel):
    """
    Performs the forward pass:
    1. MLP predicts DMP parameters (goal, weights).
    2. DMP integrates to produce a trajectory (positions).

    Args:
        mlp_model (nn.Module): The trained MLP model.
        dmp_instance (DMP): An instance of the DMP class.
        mlp_input (torch.Tensor): Input context for the MLP.
        initial_pos (torch.Tensor): Starting position of the movement.
        initial_vel (torch.Tensor): Starting velocity of the movement.

    Returns:
        torch.Tensor: Predicted trajectory positions.
    """
    # Step 1: MLP predicts goal and weights
    predicted_goal, predicted_weights = mlp_model(mlp_input)

    # Ensure predicted_goal is a scalar tensor for DMP, and initial_pos/vel are too
    if predicted_goal.dim() == 0:
        predicted_goal = predicted_goal.unsqueeze(0)
    if initial_pos.dim() == 0:
        initial_pos = initial_pos.unsqueeze(0)
    if initial_vel.dim() == 0:
        initial_vel = initial_vel.unsqueeze(0)

    # Step 2: DMP integrates with predicted parameters
    positions, _, _ = dmp_instance.integrate(initial_pos, initial_vel, predicted_goal, predicted_weights)

    return positions


# --- 4. Backward Process: Training the MLP ---
def backward_process(mlp_model, dmp_instance, optimizer, loss_fn,
                     train_data_mlp_input, train_data_initial_pos,
                     train_data_initial_vel, train_data_target_trajectories,
                     num_epochs=1000):
    """
    Trains the MLP model using a backward process (gradient descent).

    Args:
        mlp_model (nn.Module): The MLP model to be trained.
        dmp_instance (DMP): The DMP instance.
        optimizer (torch.optim.Optimizer): Optimizer for MLP parameters.
        loss_fn (torch.nn.Module): Loss function (e.g., MSELoss).
        train_data_mlp_input (torch.Tensor): Inputs to the MLP for training.
        train_data_initial_pos (torch.Tensor): Initial positions for training trajectories.
        train_data_initial_vel (torch.Tensor): Initial velocities for training trajectories.
        train_data_target_trajectories (torch.Tensor): Ground truth target trajectories.
        num_epochs (int): Number of training epochs.
    """
    mlp_model.train()  # Set MLP to training mode
    losses = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Clear gradients

        batch_predicted_trajectories = []
        # Loop through each training example to generate predictions
        for i in range(train_data_mlp_input.shape[0]):
            mlp_input = train_data_mlp_input[i].unsqueeze(0)  # Add batch dim
            initial_pos = train_data_initial_pos[i]
            initial_vel = train_data_initial_vel[i]

            # Forward pass through MLP and DMP
            predicted_trajectory = forward_process(mlp_model, dmp_instance,
                                                   mlp_input, initial_pos, initial_vel)
            batch_predicted_trajectories.append(predicted_trajectory)

        # Stack predicted trajectories to form a batch
        predicted_trajectories = torch.stack(batch_predicted_trajectories)

        # Calculate loss between predicted and target trajectories
        # Ensure target trajectories also have the correct shape
        loss = loss_fn(predicted_trajectories, train_data_target_trajectories)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f"Training complete. Final Loss: {losses[-1]:.4f}")
    return losses


# --- Main Execution ---
if __name__ == "__main__":
    # --- Setup DMP and MLP ---
    dmp = DMP(n_bfs=NUM_BASIS_FUNCTIONS, dt=DMP_DT, tau=DMP_TAU)

    # Input to MLP could be anything that describes the task, e.g., desired end effector position,
    # or even just a dummy input for a fixed task for demonstration.
    # Here, let's use a dummy input of dimension 2.
    MLP_INPUT_DIM = 2
    mlp_model = DMPParameterMLP(input_dim=MLP_INPUT_DIM, num_basis_functions=NUM_BASIS_FUNCTIONS)

    # --- Generate Dummy Training Data ---
    # We will create 3 different target trajectories for training.
    # Each target trajectory will correspond to a different MLP input context.

    NUM_TRAINING_EXAMPLES = 3
    # Dummy MLP inputs (e.g., representing different task contexts)
    train_mlp_inputs = torch.randn(NUM_TRAINING_EXAMPLES, MLP_INPUT_DIM)

    # Initial positions and velocities for the target trajectories
    train_initial_positions = torch.tensor([0.0, 0.0, 0.0])  # Start from 0 for all
    train_initial_velocities = torch.tensor([0.0, 0.0, 0.0])  # Start with 0 velocity for all

    # Define various target goals for the dummy trajectories
    dummy_target_goals = torch.tensor([5.0, -3.0, 8.0])  # Varying target goals

    train_target_trajectories = []
    print("\nGenerating dummy target trajectories...")
    for i in range(NUM_TRAINING_EXAMPLES):
        # For simplicity, let's create a simple linear interpolation for target trajectory
        # from initial_pos to dummy_target_goals[i]
        t = torch.linspace(0, 1, TRAJECTORY_LENGTH).unsqueeze(1)
        target_traj = train_initial_positions[i] + t * (dummy_target_goals[i] - train_initial_positions[i])
        train_target_trajectories.append(target_traj.squeeze())  # Remove last dimension if it exists

    train_target_trajectories = torch.stack(train_target_trajectories)
    print("Dummy target trajectories generated.")
    print(f"Shape of target trajectories: {train_target_trajectories.shape}")

    # --- Backward Process (Training) ---
    print("\nStarting backward process (training MLP)...")
    optimizer = optim.Adam(mlp_model.parameters(), lr=0.01)
    loss_function = nn.MSELoss()  # Mean Squared Error Loss

    training_losses = backward_process(mlp_model, dmp, optimizer, loss_function,
                                       train_mlp_inputs, train_initial_positions,
                                       train_initial_velocities, train_target_trajectories,
                                       num_epochs=5000)
    print("Backward process complete.")

    # --- Evaluate Forward Process after Training ---
    print("\nEvaluating forward process after training...")
    mlp_model.eval()  # Set MLP to evaluation mode

    plt.figure(figsize=(12, 8))
    plt.suptitle("DMP Trajectories: Target vs. MLP-Predicted", fontsize=16)

    with torch.no_grad():  # No need to calculate gradients during evaluation
        for i in range(NUM_TRAINING_EXAMPLES):
            # Use the same training inputs for evaluation to see how well it learned
            eval_mlp_input = train_mlp_inputs[i].unsqueeze(0)
            eval_initial_pos = train_initial_positions[i]
            eval_initial_vel = train_initial_velocities[i]
            target_traj = train_target_trajectories[i]

            # Predict trajectory using the trained MLP and DMP
            predicted_trajectory = forward_process(mlp_model, dmp,
                                                   eval_mlp_input, eval_initial_pos, eval_initial_vel)

            predicted_goal_eval, predicted_weights_eval = mlp_model(eval_mlp_input)

            print(f"\nExample {i + 1}:")
            print(f"  Target Goal: {dummy_target_goals[i].item():.2f}")
            print(f"  Predicted Goal: {predicted_goal_eval.item():.2f}")
            print(f"  Predicted Weights (first 5): {predicted_weights_eval[:5].numpy()}")

            # Plotting
            plt.subplot(NUM_TRAINING_EXAMPLES, 1, i + 1)
            time_steps = np.linspace(0, DMP_TAU, TRAJECTORY_LENGTH)
            plt.plot(time_steps, target_traj.numpy(),
                     label=f'Target Trajectory (Goal: {dummy_target_goals[i].item():.2f})', linestyle='--',
                     color='blue')
            plt.plot(time_steps, predicted_trajectory.numpy(),
                     label=f'MLP-Predicted Trajectory (Goal: {predicted_goal_eval.item():.2f})', color='red')
            plt.title(f"Trajectory Example {i + 1}")
            plt.xlabel("Time (s)")
            plt.ylabel("Position")
            plt.legend()
            plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap
    plt.show()

    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses)
    plt.title("MLP Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()
