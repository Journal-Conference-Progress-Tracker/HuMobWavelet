import torch
from torch import nn
from torch.nn import functional as F
class CNN(nn.Module):
    def __init__(self, look_back, look_ahead, start_dim=32, n_layers=2):
        super().__init__()
        self.encoder = nn.ModuleList()
        # The first convolution uses look_back as the number of input channels.
        self.encoder.append(self.__conv(look_back, start_dim))
        curr_dim = start_dim
        for i in range(n_layers - 1):
            new_dim = curr_dim // 2  # integer division to keep dimensions as int
            self.encoder.append(self.__conv(curr_dim, new_dim))
            curr_dim = new_dim

        self.decoder = nn.ModuleList()
        for i in range(n_layers - 1):
            new_dim = curr_dim * 2
            self.decoder.append(self.__deconv(curr_dim, new_dim))
            curr_dim = new_dim
        # The final layer outputs "look_ahead" channels.
        self.decoder.append(nn.ConvTranspose2d(curr_dim, look_ahead, kernel_size=3, padding=1))

    def __conv(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1),
            # Note: LayerNorm for CNNs typically expects the normalized shape. Adjust if needed.
            nn.BatchNorm2d(output_dim, affine=False),
            nn.LeakyReLU(0.2)
        )

    def __deconv(self, input_dim, output_dim):
        return nn.Sequential(
            nn.ConvTranspose2d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_dim, affine=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = x.float()
        # Iterate through each layer in the encoder.
        for layer in self.encoder:
            x = layer(x)
        # Then iterate through each layer in the decoder.
        for layer in self.decoder:
            x = layer(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        return x

# CNN Model Definition
class MyCNN(nn.Module):
    """
    Custom Convolutional Neural Network for text classification.
    
    Args:
        vocab_size (int): Size of the vocabulary.
        seq_len (int): Maximum sequence length of the input text.
        num_class (int): Number of target classes for classification.
        model_d (int): Embedding dimension and number of input channels for CNN layers.
        kernel (int): size of the kernel for the convolution
    """
    def __init__(
            self, 
            vocab_size, 
            seq_len, 
            num_class, 
            model_d, 
            kernel_size
        ):
        super(MyCNN, self).__init__()
        F.RBF
        # Embedding layer for converting input indices to dense vectors
        self.embed = nn.Embedding(vocab_size, model_d)

        # First convolutional block
        self.conv1 = nn.Conv1d(
            in_channels=model_d,    # Input channels (embedding dimension)
            out_channels=model_d,   # Output channels (same as input for simplicity)
            kernel_size=kernel_size         # Kernel size for the convolution
        )
        self.bn1 = nn.BatchNorm1d(num_features=model_d)  # Batch normalization for conv1 output
        self.pool1 = nn.MaxPool1d(kernel_size=2)         # Max pooling layer to reduce dimensions

        # Second convolutional block
        self.conv2 = nn.Conv1d(
            in_channels=model_d,    # Input channels (from previous conv layer)
            out_channels=32,        # Output channels for this layer
            kernel_size=kernel_size          # Kernel size for the convolution
        )
        self.bn2 = nn.BatchNorm1d(num_features=32)       # Batch normalization for conv2 output
        self.pool2 = nn.MaxPool1d(kernel_size=2)         # Max pooling layer to further reduce dimensions

        # Determine the number of input features for the first fully connected layer
        x_dummy = torch.zeros(1, model_d, seq_len)       # Dummy tensor to simulate input
        x_dummy = self.pool1(F.relu(self.bn1(self.conv1(x_dummy))))  # Apply conv1 block
        x_dummy = self.pool2(F.relu(self.bn2(self.conv2(x_dummy))))  # Apply conv2 block
        in_features = x_dummy.shape[1] * x_dummy.shape[2]  # Flatten the output to find in_features

        # Fully connected layers
        self.fc1 = nn.Linear(in_features, 32)           # First fully connected layer
        self.fc2 = nn.Linear(32, num_class)            # Second fully connected layer for final output

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, num_class).
        """
        # Embedding lookup and reshape for convolution
        x = self.embed(x.long()).permute(0, 2, 1)       # Convert (batch_size, seq_len, embed_dim)
                                                       # to (batch_size, embed_dim, seq_len)

        # Apply first convolutional block
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))

        # Apply second convolutional block
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))

        # Final output layer
        return self.fc2(x)




class FlexibleCNN(nn.Module):
    """
    Custom Convolutional Neural Network with user-defined number of layers for text classification.

    Args:
        vocab_size (int): Size of the vocabulary.
        seq_len (int): Maximum sequence length of the input text.
        num_class (int): Number of target classes for classification.
        model_d (int): Embedding dimension and number of input channels for CNN layers.
        kernel_size (int): Size of the kernel for the convolution.
        num_layers (int): Number of convolutional layers.
    """
    def __init__(self, vocab_size, seq_len, num_class, model_d, kernel_size, num_layers):
        super(FlexibleCNN, self).__init__()

        # Embedding layer
        self.embed = nn.Embedding(vocab_size, model_d)

        # Dynamically create convolutional layers
        self.conv_layers = nn.ModuleList()
        input_channels = model_d  # Start with embedding dimension as input channels
        for i in range(num_layers):
            conv = nn.Conv1d(
                in_channels=input_channels,
                out_channels=model_d if i < num_layers - 1 else 32,  # Last layer reduces to 32 channels
                kernel_size=kernel_size
            )
            bn = nn.BatchNorm1d(num_features=model_d if i < num_layers - 1 else 32)
            self.conv_layers.append(nn.Sequential(conv, bn, nn.ReLU(), nn.MaxPool1d(kernel_size=2)))
            input_channels = model_d if i < num_layers - 1 else 32  # Update input channels for next layer

        # Determine the number of input features for the first fully connected layer
        x_dummy = torch.zeros(1, model_d, seq_len)  # Dummy tensor to simulate input
        for layer in self.conv_layers:
            x_dummy = layer(x_dummy)
        in_features = x_dummy.shape[1] * x_dummy.shape[2]  # Flatten output dimensions

        # Fully connected layers
        self.fc1 = nn.Linear(in_features, 32)
        self.fc2 = nn.Linear(32, num_class)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            Tensor: Output tensor of shape (batch_size, num_class).
        """
        # Embedding and reshape for convolution
        x = self.embed(x.long()).permute(0, 2, 1)

        # Apply convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        return self.fc2(x)
