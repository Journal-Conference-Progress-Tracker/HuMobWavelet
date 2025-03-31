
import torch

from torch import nn

from torch.nn import functional as F
import itertools


import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, embed_dim]
        """
        Q = self.query(x)  # [batch, seq_len, embed_dim]
        K = self.key(x)    # [batch, seq_len, embed_dim]
        V = self.value(x)  # [batch, seq_len, embed_dim]

        # Compute attention scores: QK^T
        # scores shape: [batch, seq_len, seq_len]
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        # Convert scores to probabilities
        attn_weights = torch.softmax(scores, dim=-1)  # [batch, seq_len, seq_len]

        # Weighted sum of the values
        out = torch.bmm(attn_weights, V)  # [batch, seq_len, embed_dim]

        return out, attn_weights

class EmbedNNWithAttention(nn.Module):
    def __init__(
        self, 
        vocab_size,      # same as your old output_size + 1
        embed_dim=10,
        hidden_dim=32,
        n_layers=3,
        output_dim=10,   # final output size (e.g., classification classes)
    ):
        super().__init__()

        # 1) Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2) Self-attention (single-head or multi-head)
        self.attention = SelfAttention(embed_dim)
        # or use PyTorch's built-in multi-head:
        # self.attention = nn.MultiheadAttention(embed_dim, num_heads=1, batch_first=True)

        # 3) Feed-forward layers
        layers = []
        in_dim = embed_dim  # after attention, we still have embed_dim per token
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        # 4) Final linear to get desired output
        self.ff = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: [batch_size, seq_len]
        """
        # 1) Embed
        x = x.long()  # ensure x is long type for embedding
        x = self.embedding(x)  # [batch, seq_len, embed_dim]

        # 2) Apply self-attention
        #    out shape: [batch, seq_len, embed_dim]
        attn_out, attn_weights = self.attention(x)

        # 3) Pool or flatten the attention output
        #    Option A: Average pool over the sequence
        #    This reduces [batch, seq_len, embed_dim] -> [batch, embed_dim]
        attn_pooled = attn_out.mean(dim=1)

        # 4) Feed-forward
        #    Now we pass [batch, embed_dim] into the FF layers
        z = self.ff(attn_pooled)

        # 5) Final output
        out = self.out(z)  # [batch, output_dim]
        return out


class MembershipLayer(nn.Module):
    def __init__(self, input_dim, num_membership_functions):
        """
        input_dim: Number of input features
        num_membership_functions: Number of Gaussian functions per input
        """
        super(MembershipLayer, self).__init__()
        
        # Learnable parameters: Mean and Standard Deviation
        self.mean = nn.Parameter(torch.randn(input_dim, num_membership_functions))
        self.sigma = nn.Parameter(torch.randn(input_dim, num_membership_functions))
    
    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, input_dim)
        Output: Membership values of shape (batch_size, input_dim, num_membership_functions)
        """
        # Compute Gaussian Membership Function
        x = x.unsqueeze(-1)  # Shape: (batch_size, input_dim, 1)
        gaussian_output = torch.exp(-((x - self.mean) ** 2) / (self.sigma ** 2))

        return gaussian_output

class FuzzyNNWithReducedRules(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,
            latent_dim = 32,
            num_memberships = 2,
        ):
        super().__init__()        
        self.input_size = input_size
        self.output_size = output_size,
        self.latent_dim = latent_dim
        self.num_memberships = num_memberships

        self.membership_layers = nn.ModuleList([
            MembershipLayer(1, num_memberships) for _ in range(input_size)
        ])
        
        self.hidden_layer = nn.Sequential(
            nn.Linear(num_memberships, latent_dim),
            nn.LayerNorm(latent_dim, elementwise_affine=False),
            nn.Tanh()
        )
        self.output_layer = nn.Linear(latent_dim, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.o_r_norm = nn.LayerNorm(num_memberships, elementwise_affine=False)

    def forward(self, x):
        batch = x.size(0)
        x_expanded = x.squeeze(-1)
        membership_values = torch.stack([
            layer(x_expanded[:, i]) for i, layer in enumerate(self.membership_layers)
        ], dim=1)

        o_r = torch.ones((batch, self.num_memberships)).to(x.device)
                
        for i in range(membership_values.shape[-1]):
            o_r[:, i] = torch.prod(membership_values[:, :, i], dim=1)
        o_r = self.o_r_norm(o_r)
        o_h = self.hidden_layer(o_r)
        return self.output_layer(o_h)

class FuzzyNN(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,
            projection_size = 15,
            latent_dim = 32,
            num_memberships = 2,
        ):
        super().__init__()        
        self.input_size = input_size
        self.output_size = output_size,
        self.latent_dim = latent_dim
        self.projection_size = projection_size
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, projection_size),
            nn.LayerNorm(projection_size, elementwise_affine=False),
            nn.Tanh()
        )
        self.num_memberships = num_memberships

        self.membership_layers = nn.ModuleList([
            MembershipLayer(1, num_memberships) for _ in range(projection_size)
        ])
        num_layers = num_memberships ** projection_size
        
        self.hidden_layer = nn.Sequential(
            nn.Linear(num_layers, latent_dim),
            nn.LayerNorm(latent_dim, elementwise_affine=False),
            nn.Tanh()
        )
        self.output_layer = nn.Linear(latent_dim, output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.o_r_norm = nn.LayerNorm(num_layers, elementwise_affine=False)
        self.rule_indices = torch.tensor(list(
            itertools.product(*[range(self.num_memberships)] * projection_size)))
    def forward(self, x):
        batch = x.size(0)
        x = self.input_projection(x)
        x_expanded = x.squeeze(-1)
        membership_values = torch.stack([layer(x_expanded[:, i]) for i, layer in enumerate(self.membership_layers)], dim=1)

        o_r = torch.ones((batch, self.rule_indices.shape[0])).to(x.device)
        for i in range(self.projection_size):
            o_r *= membership_values[:, i, self.rule_indices[:, i]]
        o_r = self.o_r_norm(o_r)
        o_h = self.hidden_layer(o_r)
        return self.output_layer(o_h)
    
class FCFuzzyNN(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,
            latent_dim = 32,
            num_memberships = 2,
        ):
        super().__init__()        
        self.input_size = input_size
        self.output_size = output_size,
        self.latent_dim = latent_dim
        self.num_memberships = num_memberships

        self.membership_layers = nn.ModuleList([
            MembershipLayer(1, num_memberships) for _ in range(input_size)
        ])
        num_layers = num_memberships * input_size
        
        self.hidden_layer = nn.Sequential(
            nn.Linear(num_layers, latent_dim),
            nn.LayerNorm(latent_dim, elementwise_affine=False),
            nn.Mish()
        )
        self.output_layer = nn.Linear(latent_dim, output_size)
    def forward(self, x):
        batch = x.size(0)
        x_expanded = x.squeeze(-1)
        membership_values = torch.stack([layer(x_expanded[:, i]) for i, layer in enumerate(self.membership_layers)], dim=1).view(batch, -1)
        o_h = self.hidden_layer(membership_values)
        return self.output_layer(o_h)
    
class EmbedNN(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,
            start_dim=32,
            embed_size=10,
            n_layers=3,
        ):
        super().__init__()
        self.embedding = nn.Embedding(output_size + 1, embed_size)
        self.net = [self._block(input_size * embed_size, start_dim)]
        current_dim = start_dim
        for _ in range(n_layers - 1):
            self.net.append(self._block(current_dim, current_dim * 2))
            current_dim = current_dim * 2
        self.net.append(nn.Linear(current_dim, output_size))
        self.net = nn.Sequential(*self.net)
    def _block(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Tanh()
        )
    def forward(self, x):
        x = x.long()
        batch = x.size(0)
        x = self.embedding(x)
        x = x.view(batch, -1)
        x = self.net(x)
        return x
    
class NN(nn.Module):
    def __init__(
            self, 
            input_size,
            output_size,
            start_dim,
            n_layers,
        ):
        super().__init__()
        self.net = [self._block(input_size, start_dim)]
        current_dim = start_dim
        for _ in range(n_layers - 1):
            self.net.append(self._block(current_dim, current_dim * 2))
            current_dim = current_dim * 2
        self.net.append(nn.Linear(current_dim, output_size))
        self.net = nn.Sequential(*self.net)
    def _block(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.net(x)
        return x
    
class RBF(nn.Module):
    """
    Radial Basis Function (RBF) activation layer.
    
    Args:
        centers (Tensor): Learnable centers for the RBF.
        gamma (float): Scaling factor for the RBF kernel.
    """
    def __init__(self, centers, gamma=1.0):
        super(RBF, self).__init__()
        self.centers = nn.Parameter(centers)  # Learnable centers
        self.gamma = gamma  # Scaling factor

    def forward(self, x):
        """
        Apply RBF activation.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len).
        
        Returns:
            Tensor: RBF-transformed output.
        """
        x = x.unsqueeze(-1)  # Add dimension to match centers
        return torch.exp(-self.gamma * torch.sum((x - self.centers) ** 2, dim=-1))

class FNN(nn.Module):
    """
    Feedforward Neural Network (FNN) with customizable hidden layers and activation function.

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list of int): List of sizes for each hidden layer.
        act (callable): Activation function to use between layers (default: F.tanh).
        output_dim (int): Number of output features (default: 2).
        rbf (bool): Whether to add rbf
    """
    def __init__(self, input_dim, hidden_dims, act=F.tanh, output_dim=2, rbf: bool = False):
        super(FNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.act = act
        if rbf:
            centers = torch.arange(input_dim).float()  # Initialize centers as a range of vocab indices
            self.rbf = RBF(centers, gamma=1.0)
        else:
            self.rbf = None
        # Define layers
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:  # First hidden layer
                self.layers.append(nn.Linear(input_dim, hidden_dims[i]))
            else:  # Subsequent hidden layers
                self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        if self.rbf is not None:
            x = self.rbf(x)
        for layer in self.layers:
            x = self.act(layer(x))  # Apply activation function after each hidden layer
        return self.output_layer(x)  # Output layer (no activation for logits)
