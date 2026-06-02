import torch
import torch.nn as nn
import torch.nn.functional as F
from game_functions import *

class AnimalBrain(nn.Module):
    """
    Neural controller for an artificial animal agent in an evolving ecosystem.

    This model takes as input a perception vector derived from the animal's
    vision function (currently its distance and angle to every detectable object type (food, predator, etc)).
    The network outputs two continuous values representing:
        1. Forward movement speed
        2. Angular turning rate

    It is designed for neuro-evolution rather than gradient-based learning.
    It is not a true NEAT algorithm due to the absence of skip connections, but it has dynamic topology augmentation.
    """

    def __init__(
        self, 
        n_external_infos: int,
        n_self_infos: int,
        hidden_dims: list = [],  # Pass a list like [12, 12, 18] or [6] or []
        initial_weight_std: float = 0.1,
        initial_bias_std: float = 0.05
    ):
        """
        Parameters
        ----------
        n_external_infos : int
            Number of external information features the agent can perceive.

        n_self_infos : int
            Number of self-related features (e.g., speed, satiety).

        hidden_dims : list
            Dimensionality of the hidden layers. 
            Defaults to no hidden layers (a default input - output connection)

        initial_weight_std: float
            standard deviation of newly initialised weights
        
        initial_bias_std: float
            standard deviation of newly initialised biases

        Notes
        -----
        The input dimensionality formula is:
            n_external_infos + n_self_infos
        The additional terms capture internal agent states (e.g., current speed,
        energy level, or similar self-awareness signals).
        """

        super().__init__()
        self.input_dim = n_external_infos + n_self_infos
        # Output: [forward_speed, angular_velocity]
        self.output_dim = 2

        self.hidden_dims = hidden_dims

        # Build layers dynamically using nn.ModuleList
        self.layers = nn.ModuleList()
        prev_dim = self.input_dim
        for h_dim in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim, bias=True))
            prev_dim = h_dim
            
        # Final output layer
        self.out = nn.Linear(prev_dim, self.output_dim, bias=True)

        # Custom weight and bias initialization
        self.init_weights(initial_weight_std, initial_bias_std)

    def init_weights(self, weight_std, bias_std):
        with torch.no_grad():
            for layer in self.layers:
                nn.init.normal_(layer.weight, mean=0, std=weight_std)
                nn.init.normal_(layer.bias, mean=0, std=bias_std)
            nn.init.normal_(self.out.weight, mean=0, std=weight_std)
            nn.init.normal_(self.out.bias, mean=0, std=bias_std)


    def forward(self, x: torch.Tensor, return_activations: bool = False) -> torch.Tensor:
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
    def forward(self, x: torch.Tensor, return_activations: bool = False):
        activations = []
        h = x
        
        # Pass sequentially through all hidden layers
        for layer in self.layers:
            h = F.relu(layer(h))
            if return_activations:
                activations.append(h)
                
        out = torch.tanh(self.out(h))
        
        if return_activations:
            return out, activations
        return out

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
        return list(self.hidden_dims)

    def get_network_weights(self):
        # Dynamically serializes whatever structural state exists
        data = {"input_dim": self.input_dim, "output_dim": self.output_dim}
        for i, layer in enumerate(self.layers):
            data[f"layer_{i}_weights"] = layer.weight.data.cpu().numpy().tolist()
        data["out_weights"] = self.out.weight.data.cpu().numpy().tolist()
        data["hidden_dims"] = self.hidden_dims
        return data
