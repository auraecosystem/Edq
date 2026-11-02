# EDQ-BRAIN: New network + new weights + quantum-inspired entropy
# Requirements: torch, (optional) qiskit
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# 1) Define brand-new network
# ------------------------------
class EDQNumeric(nn.Module):
    """Numeric branch (new structure, new weights)"""
    def __init__(self, input_dim=16, hidden_dims=(128, 64)):
        super().__init__()
        self.l1 = nn.Linear(input_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # custom initializer: Xavier uniform for weights, small constant for bias
        for layer in (self.l1, self.l2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = F.gelu(self.l1(x))
        x = F.gelu(self.l2(x))
        return x  # shape: (batch, hidden_dims[1])

class BrainText(nn.Module):
    """Text branch (small transformer-ish encoder from scratch)"""
    def __init__(self, vocab_size, embed_dim=64, nhead=4, nhid=128, nlayers=1, seq_len=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=nhid, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pool = nn.Linear(embed_dim, embed_dim)
        # initialize new weights
        nn.init.xavier_uniform_(self.pool.weight)
        nn.init.zeros_(self.pool.bias)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embed(x) * math.sqrt(self.embed.embedding_dim)  # (batch, seq_len, embed_dim)
        # transformer expects (seq_len, batch, embed_dim)
        t_in = emb.permute(1, 0, 2)
        t_out = self.transformer(t_in)  # (seq_len, batch, embed_dim)
        # mean-pool
        pooled = t_out.mean(dim=0)  # (batch, embed_dim)
        return F.gelu(self.pool(pooled))

class FusionEDQBrain(nn.Module):
    """Fusion network combining both branches + quantum modulation"""
    def __init__(self, vocab_size, numeric_dim=16, seq_len=32, out_dim=16):
        super().__init__()
        self.numeric = EDQNumeric(input_dim=numeric_dim, hidden_dims=(128,64))
        self.text = BrainText(vocab_size=vocab_size, embed_dim=64, nhead=4, nhid=128, nlayers=1, seq_len=seq_len)
        
        # Determine the output dimension of the numeric and text branches
        # EDQNumeric output: (batch, 64)
        # BrainText output: (batch, 64) (embed_dim)
        
        fusion_input_dim = 64 + 64 # Sum of output dimensions from numeric and text branches
        
        # New fusion layers
        self.fusion_l1 = nn.Linear(fusion_input_dim, 128)
        self.fusion_l2 = nn.Linear(128, out_dim)
        
        # New custom initializer for fusion layers
        for layer in (self.fusion_l1, self.fusion_l2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, numeric_input, text_input, quantum_scalar=1.0):
        # Numeric branch processing
        numeric_features = self.numeric(numeric_input) # (batch, 64)
        
        # Text branch processing
        text_features = self.text(text_input) # (batch, 64)
        
        # Concatenate features
        fused_features = torch.cat((numeric_features, text_features), dim=1) # (batch, 128)
        
        # Fusion layers
        x = F.gelu(self.fusion_l1(fused_features))
        output = self.fusion_l2(x) # (batch, out_dim)
        
        # Apply quantum modulation: scale the output
        output = output * quantum_scalar 
        
        return output

# Example usage (can be moved to a test or run script later)
if __name__ == '__main__':
    vocab_size = 10000
    numeric_dim = 16
    seq_len = 32
    out_dim = 16 # e.g., representing action probabilities or decision vector

    model = FusionEDQBrain(vocab_size, numeric_dim, seq_len, out_dim).to(DEVICE)
    print("EDQ-BRAIN Model Architecture:")
    print(model)

    # Example inputs
    numeric_in = torch.randn(1, numeric_dim).to(DEVICE) # Batch size 1
    text_in = torch.randint(0, vocab_size, (1, seq_len)).to(DEVICE) # Batch size 1

    # Simulate quantum scalar from the dedicated utility
    # (In a real scenario, this would come from utilities/QuantumEntropySource.py)
    # For standalone test, we simulate it here.
    quantum_scalar_simulated = 0.85 # Placeholder

    output = model(numeric_in, text_in, quantum_scalar=quantum_scalar_simulated)
    print("\nExample Output Shape:", output.shape)
    print("Example Output:\n", output)

    # Demonstrate training loop concept (simplified)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss() # Example loss

    # Dummy target for demonstration
    target = torch.randn(1, out_dim).to(DEVICE)

    # Training step
    optimizer.zero_grad()
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(f"\nSimulated training step - Loss: {loss.item():.4f}")

