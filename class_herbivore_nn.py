import torch
import torch.nn as nn
import torch.nn.functional as F
from game_functions import *

class AnimalBrain(nn.Module):
    def __init__(self, n_ray_sections, n_types_of_info_in_each_section = 2, hidden_dim_1=12, hidden_dim_2=12):
        """
        n_rays: Number of raycasts (e.g., 5)
        n_channels: Channels per ray (e.g., 3 for food, wall, predator)
        hidden_dim: Size of hidden layers (controls NN capacity)
        """
        
        super().__init__()

        input_dim = (n_ray_sections+1) * n_types_of_info_in_each_section  + 2
        output_dim = 2   #speed and angle speed
        self.fc1 = nn.Linear(input_dim, hidden_dim_1, bias=False)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2, bias=False)
        self.out = nn.Linear(hidden_dim_2, output_dim, bias=False)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        return x
    
    def mutate(self,mutation_rate,mutation_strength):
        for param in self.parameters():
            mask = torch.rand_like(param) < mutation_rate
            noise = torch.randn_like(param) * mutation_strength
            param.data += mask * noise
    
    def get_dim_sizes(self):
        return self.fc1.out_features, self.fc2.out_features
