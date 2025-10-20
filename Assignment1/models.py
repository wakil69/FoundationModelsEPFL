import torch.nn as nn
from torchvision.models import resnet18, resnet50, resnet34
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    """
    A simple neural network template for self-supervised learning.

    Structure:
    1. Encoder: Maps an input image of shape 
       (input_channels, input_dim, input_dim) 
       into a lower-dimensional feature representation.
    2. Projector: Transforms the encoder output into the final 
       embedding space of size `proj_dim`.

    Notes:
    - DO NOT modify the fixed class variables: 
      `input_dim`, `input_channels`, and `feature_dim`.
    - You may freely modify the architecture of the encoder 
      and projector (layers, activations, normalization, etc.).
    - You may add additional helper functions or class variables if needed.
    """

    ####### DO NOT MODIFY THE CLASS VARIABLES #######
    input_dim: int = 64
    input_channels: int = 3
    feature_dim: int = 1000
    proj_dim: int = 128
    #################################################

    def __init__(self):
        super().__init__()


        ######################## TODO: YOUR CODE HERE ########################
        enc = resnet18(weights=None) # Initialize a ResNet-18 backbone without pretrained weights
        enc.conv1 = nn.Conv2d(self.input_channels, 64, 3, 1, 1, bias=False) # Replace the first convolution layer
        enc.maxpool = nn.Identity() # Remove the initial max-pooling layer to keep higher spatial resolution
        enc.fc = nn.Identity() # Remove the final fully connected classification layer 
        self.encoder = enc # Register the modified ResNet as the encoder backbone

        self.encoder_output = nn.Linear(512, self.feature_dim) # Add a linear layer to map the 512 ResNet output to our desired feature dimension (self.feature_dim)
        
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, self.proj_dim),
        )

        
        ######################################################################

    def normalize(self, x, eps=1e-8):
        """
        Normalizes a batch of feature vectors.
        """
        return x / (x.norm(dim=-1, keepdim=True) + eps)
        #return F.normalize(x, dim=-1) # Added by Chaimae to test 
        
    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output embedding of shape (batch_size, proj_dim).
        """
        features = self.encoder(x) # Pass input images through the ResNet encoder backbone
        features = self.encoder_output(features)
        projected_features = self.normalize(self.projector(features))
        return projected_features
    
    def get_features(self, x):
        """
        Get the features from the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output features of shape (batch_size, feature_dim).
        """
        features = self.encoder(x)
        features = self.encoder_output(features)
        return features