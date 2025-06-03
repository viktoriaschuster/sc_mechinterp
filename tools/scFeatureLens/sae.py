"""
Sparse Autoencoder implementation for scFeatureLens.
"""

import os
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

def train_sae(embeddings, config, logger):
    """Train a Sparse Autoencoder on the embeddings."""
    if embeddings is None:
        logger.error("No embeddings loaded for SAE training")
        return

    logger.info("Training Sparse Autoencoder...")

    # Create model
    input_size = embeddings.shape[1]
    hidden_size = config.sae_hidden_size
    sae_model = SparseAutoencoder(input_size, hidden_size)
    sae_model.to(config.device)

    # Training parameters
    lr = config.sae_learning_rate
    l1_weight = config.sae_l1_weight
    batch_size = config.sae_batch_size
    epochs = config.sae_epochs

    # Create optimizer
    optimizer = torch.optim.Adam(sae_model.parameters(), lr=lr)

    # Create dataset
    dataset = torch.utils.data.TensorDataset(embeddings)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    device = torch.device(config.device)
    for epoch in range(epochs):
        running_loss = 0.0
        running_mse_loss = 0.0
        running_l1_loss = 0.0
        
        for batch in dataloader:
            # Get batch and move to device
            x = batch[0].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            recon, encoded = sae_model(x)
            
            # Calculate losses
            mse_loss = torch.nn.functional.mse_loss(recon, x)
            l1_loss = torch.mean(torch.abs(encoded))
            loss = mse_loss + l1_weight * l1_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            running_loss += loss.item()
            running_mse_loss += mse_loss.item()
            running_l1_loss += l1_loss.item()
        
        # Print progress
        if config.verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.6f}, "
                    f"MSE: {running_mse_loss/len(dataloader):.6f}, L1: {running_l1_loss/len(dataloader):.6f}")
    
    # Save model
    model_path = os.path.join(config.output_dir, "sae_model.pt")
    torch.save(sae_model.state_dict(), model_path)
    logger.info(f"SAE model saved to {model_path}")

    return sae_model