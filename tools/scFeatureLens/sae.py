"""
Sparse Autoencoder implementation for scFeatureLens.
"""

import torch
import torch.nn as nn
from typing import Tuple


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder for mechanistic interpretability of embeddings."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super(SparseAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Tuple of (decoded, encoded) tensors
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature activations."""
        return self.encoder(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to input space."""
        return self.decoder(x)
    
    def get_feature_weights(self) -> torch.Tensor:
        """Get the decoder weights (feature directions)."""
        return self.decoder[0].weight.data
    
    def get_encoder_weights(self) -> torch.Tensor:
        """Get the encoder weights."""
        return self.encoder[0].weight.data
    
    def sparsity_loss(self, activations: torch.Tensor, l1_penalty: float = 1e-3) -> torch.Tensor:
        """Calculate L1 sparsity loss."""
        return l1_penalty * torch.mean(torch.abs(activations))
    
    def reconstruction_loss(self, input_data: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """Calculate reconstruction loss."""
        return nn.MSELoss()(reconstructed, input_data)
    
    def total_loss(self, input_data: torch.Tensor, l1_penalty: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate total loss (reconstruction + sparsity).
        
        Returns:
            Tuple of (total_loss, reconstruction_loss, sparsity_loss)
        """
        reconstructed, encoded = self.forward(input_data)
        recon_loss = self.reconstruction_loss(input_data, reconstructed)
        sparse_loss = self.sparsity_loss(encoded, l1_penalty)
        total = recon_loss + sparse_loss
        
        return total, recon_loss, sparse_loss
