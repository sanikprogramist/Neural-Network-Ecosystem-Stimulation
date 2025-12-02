import torch
import torch.nn as nn
import torch.nn.functional as F
from game_functions import *

class AnimalBrain(nn.Module):
    """
    Neural controller for an artificial animal agent in an evolving ecosystem.

    This model takes as input a perception vector derived from the animal's
    ray-based environment sensing system. Each ray provides information such as
    distance to food, walls, or other animals. The network outputs two continuous
    values representing:
        1. Forward movement speed
        2. Angular turning rate

    The AnimalBrain is intentionally small, enabling efficient simulation of
    hundreds or thousands of agents evolving through mutation. It is designed
    for neuro-evolution rather than gradient-based learning.
    """

    def __init__(
        self, 
        n_ray_sections: int,
        n_types_of_info_in_each_section: int = 2,
        hidden_dim_1: int = 12,
        hidden_dim_2: int = 12
    ):
        """
        Parameters
        ----------
        n_ray_sections : int
            Number of raycast segments in the agent's field of view.
            Each section encodes environmental information.

        n_types_of_info_in_each_section : int, optional
            How many distinct features are captured per raycast segment
            (e.g., food density, predator presence). Defaults to 2.

        hidden_dim_1 : int, optional
            Dimensionality of the first hidden layer. Increasing this expands
            representational capacity at the cost of computation.

        hidden_dim_2 : int, optional
            Dimensionality of the second hidden layer.

        Notes
        -----
        The input dimensionality formula is:
            (n_ray_sections + 1) * n_types_of_info + 2
        The additional +2 captures internal agent states (e.g., current speed,
        energy level, or similar self-awareness signals).
        """

        super().__init__()

        # Calculate number of input features from perception and agent state.
        input_dim = (n_ray_sections + 1) * n_types_of_info_in_each_section + 2

        # Output: [forward_speed, angular_velocity]
        output_dim = 2

        # A compact fully-connected architecture, no biases for evolutionary simplicity.
        self.fc1 = nn.Linear(input_dim, hidden_dim_1, bias=False)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2, bias=False)
        self.out = nn.Linear(hidden_dim_2, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Execute a forward pass through the neural network.

        Applies ReLU activations to the hidden layers to introduce non-linearity.
        The final output uses tanh to keep movement bounds in [-1, 1], which is
        convenient for interpreting direction and speed.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, input_dim]

        Returns
        -------
        torch.Tensor
            Movement command tensor [speed, angular_speed]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        return x

    def mutate(self, mutation_rate: float, mutation_strength: float) -> None:
        """
        Evolves the animal's brain through random parameter mutations.

        Each parameter tensor receives random Gaussian noise with a probability
        defined by mutation_rate. This drives open-ended behavioral evolution
        without gradients or backpropagation.

        Parameters
        ----------
        mutation_rate : float
            Probability that an individual weight will mutate.

        mutation_strength : float
            Standard deviation of the Gaussian noise added to weights.
        """
        for param in self.parameters():
            mask = torch.rand_like(param) < mutation_rate
            noise = torch.randn_like(param) * mutation_strength
            param.data += mask * noise  # element-wise selective mutation

    def get_dim_sizes(self):
        """
        Utility function returning hidden layer sizes, useful for logging,
        debugging evolutionary hyperparameter sweeps, or constructing
        crossover mechanisms between genomes.

        Returns
        -------
        (int, int)
            Tuple -> (size of first hidden layer, size of second hidden layer)
        """
        return self.fc1.out_features, self.fc2.out_features
