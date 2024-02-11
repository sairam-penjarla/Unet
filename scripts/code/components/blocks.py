import torch
import torch.nn as nn
from code.logging import logger  # Importing logger module for logging

class DoubleConv(nn.Module):
    """
    Defines a double convolution block.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # Using ReLU activation function
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)  # Using ReLU activation function
        )

    def forward(self, x):
        """
        Forward pass of the double convolution block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv_op(x)


class DownSample(nn.Module):
    """
    Defines a downsampling block.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)  # Double convolution
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Max pooling operation

    def forward(self, x):
        """
        Forward pass of the downsampling block.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Tuple containing downsampled tensor and pooled tensor.
        """
        down = self.conv(x)  # Convolutional operation
        p = self.pool(down)  # Pooling operation

        return down, p


class UpSample(nn.Module):
    """
    Defines an upsampling block.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)  # Transposed convolution for upsampling
        self.conv = DoubleConv(in_channels, out_channels)  # Double convolution

    def forward(self, x1, x2):
       """
       Forward pass of the upsampling block.
       
       Args:
           x1 (torch.Tensor): Tensor from the previous layer.
           x2 (torch.Tensor): Tensor from the skip connection.
       
       Returns:
           torch.Tensor: Output tensor.
       """
       x1 = self.up(x1)  # Upsampling operation
       x = torch.cat([x1, x2], 1)  # Concatenating feature maps
       return self.conv(x)  # Convolutional operation
