# Standard Library
import math

# External Modules
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaLN(nn.Module):
    """Adaptive Layer Normalization - applies FiLM conditioning to LayerNorm"""

    def __init__(self, normalized_shape, conditioning_dim, eps=1e-5, sequential_condition=False):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Standard LayerNorm parameters (will be modulated)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        # Conditioning network to generate scale and shift
        self.conditioning_net = nn.Sequential(nn.Linear(conditioning_dim, conditioning_dim), nn.SiLU(), nn.Linear(conditioning_dim, 2 * normalized_shape))
        self.sequential_condition = sequential_condition

    def forward(self, x, conditioning):
        """
        Args:
            x: (batch_size, seq_len, normalized_shape) - input features
            conditioning: (batch_size, conditioning_dim) - conditioning vector
        """
        # Standard layer normalization
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Apply standard weight and bias
        out = normalized * self.weight + self.bias

        # Generate adaptive scale and shift from conditioning
        conditioning_params = self.conditioning_net(conditioning)  # (batch_size, 2 * normalized_shape)
        scale, shift = conditioning_params.chunk(2, dim=-1)  # Each: (batch_size, normalized_shape)

        # Apply adaptive modulation
        # Scale with sigmoid to ensure positive values, shift can be any value
        scale = torch.sigmoid(scale)  # Ensures scale is in (0, 1)

        # # Expand for broadcasting with sequence dimension
        if not self.sequential_condition:
            scale = scale.unsqueeze(1)  # (batch_size, 1, normalized_shape)
            shift = shift.unsqueeze(1)  # (batch_size, 1, normalized_shape)

        # Apply adaptive scaling and shifting
        out = scale * out + shift

        return out


class MultiHeadAttentionAdaLN(nn.Module):
    """Multi-head attention with AdaLN conditioning"""

    def __init__(self, d_model, nhead, conditioning_dim, dropout=0.1, sequential_condition=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        # Standard attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # AdaLN for pre-attention normalization
        self.norm = AdaLN(d_model, conditioning_dim, sequential_condition=sequential_condition)

    def forward(self, x, conditioning, memory=None, mask=None, key_padding_mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model) - input sequence
            conditioning: (batch_size, conditioning_dim) - conditioning vector
            memory: (batch_size, mem_len, d_model) - memory for cross-attention
            mask: (seq_len, seq_len) or (seq_len, mem_len) - attention mask
            key_padding_mask: (batch_size, seq_len) or (batch_size, mem_len) - padding mask
        """
        batch_size, seq_len, d_model = x.shape

        # Apply AdaLN conditioning
        x_norm = self.norm(x, conditioning)

        # Prepare queries from conditioned input
        q = self.q_proj(x_norm)  # (batch_size, seq_len, d_model)

        # Keys and values come from memory (cross-attention) or input (self-attention)
        if memory is not None:
            # Cross-attention
            k = self.k_proj(memory)  # (batch_size, mem_len, d_model)
            v = self.v_proj(memory)  # (batch_size, mem_len, d_model)
            mem_len = memory.shape[1]
        else:
            # Self-attention
            k = self.k_proj(x_norm)  # (batch_size, seq_len, d_model)
            v = self.v_proj(x_norm)  # (batch_size, seq_len, d_model)
            mem_len = seq_len

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)  # (batch_size, nhead, seq_len, head_dim)
        k = k.view(batch_size, mem_len, self.nhead, self.head_dim).transpose(1, 2)  # (batch_size, nhead, mem_len, head_dim)
        v = v.view(batch_size, mem_len, self.nhead, self.head_dim).transpose(1, 2)  # (batch_size, nhead, mem_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (batch_size, nhead, seq_len, mem_len)

        # Apply attention mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # Apply key padding mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, nhead, seq_len, head_dim)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Final linear projection
        output = self.out_proj(attn_output)

        return output


class TransformerDecoderLayerAdaLN(nn.Module):
    """Transformer decoder layer with AdaLN conditioning"""

    def __init__(self, d_model, nhead, dim_feedforward, conditioning_dim, dropout=0.1, activation="gelu", sequential_condition=False):
        super().__init__()

        # Self-attention with AdaLN
        self.self_attn = MultiHeadAttentionAdaLN(d_model, nhead, conditioning_dim, dropout, sequential_condition=sequential_condition)

        # Cross-attention with AdaLN
        self.cross_attn = MultiHeadAttentionAdaLN(d_model, nhead, conditioning_dim, dropout, sequential_condition=sequential_condition)

        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

        # AdaLN for feedforward
        self.norm_ff = AdaLN(d_model, conditioning_dim, sequential_condition=sequential_condition)

        # Activation
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, tgt, memory, conditioning, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: (batch_size, tgt_len, d_model) - target sequence
            memory: (batch_size, mem_len, d_model) - memory sequence
            conditioning: (batch_size, conditioning_dim) - conditioning vector
            tgt_mask: (tgt_len, tgt_len) - target mask
            memory_mask: (tgt_len, mem_len) - memory mask
            tgt_key_padding_mask: (batch_size, tgt_len) - target padding mask
            memory_key_padding_mask: (batch_size, mem_len) - memory padding mask
        """
        # Self-attention with residual connection
        tgt2 = self.self_attn(tgt, conditioning, mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        # Cross-attention with residual connection
        tgt2 = self.cross_attn(tgt, conditioning, memory=memory, mask=memory_mask, key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)

        # Feedforward with AdaLN and residual connection
        tgt_norm = self.norm_ff(tgt, conditioning)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt_norm))))
        tgt = tgt + self.dropout(tgt2)

        return tgt


class TransformerDecoderAdaLN(nn.Module):
    """Transformer decoder with AdaLN conditioning"""

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, tgt, memory, conditioning, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: (batch_size, tgt_len, d_model) - target sequence
            memory: (batch_size, mem_len, d_model) - memory sequence
            conditioning: (batch_size, conditioning_dim) - conditioning vector
        """
        output = tgt

        for layer in self.layers:
            output = layer(
                output, memory, conditioning, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
            )

        return output


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: (bs, 1, 1, 1)
        Returns:
        - embeddings: (bs, dim)
        """
        t = t.view(-1, 1)  # (bs, 1)
        freqs = t * self.weights * 2 * math.pi  # (bs, half_dim)
        sin_embed = torch.sin(freqs)  # (bs, half_dim)
        cos_embed = torch.cos(freqs)  # (bs, half_dim)
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2)  # (bs, dim)
