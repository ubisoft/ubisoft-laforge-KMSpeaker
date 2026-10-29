# External Modules
import torch
import torch.nn as nn


class ScalarLayerCombiner(nn.Module):
    """
    Combines hidden layers using a learned scalar weight per layer, followed by a linear projection.
    """

    def __init__(self, num_layers, input_dim, output_dim, use_softmax=False):
        """
        Args:
            num_layers (int): Number of hidden layers to combine (e.g., 25 for w2v-bert-2.0).
            input_dim (int): Dimension of each hidden layer (e.g., 1024 for w2v-bert-2.0).
            output_dim (int): Desired output dimension (model_dim).
            use_softmax (bool): Whether to apply softmax to the layer weights.
        """
        super().__init__()
        self.num_layers = num_layers
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_softmax = use_softmax

        # Learnable weights for each layer, initialized to be uniform
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

        # Linear layer to project the weighted sum of layers to the target output_dim
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, stacked_hidden_states):
        """
        Args:
            stacked_hidden_states (torch.Tensor):
                Shape (N x T_audio_model x n_layers x n_hidden_dim).

        Returns:
            torch.Tensor: Combined features (N x T_audio_model x model_dim).
        """
        if self.use_softmax:
            normalized_weights = torch.softmax(self.layer_weights, dim=0)
        else:
            normalized_weights = self.layer_weights

        # Reshape weights for broadcasting: (1, 1, n_layers, 1)
        weights_reshaped = normalized_weights.view(1, 1, self.num_layers, 1)

        weighted_sum_features = (stacked_hidden_states * weights_reshaped).sum(dim=2)

        combined_output = self.projection(weighted_sum_features)

        return combined_output
