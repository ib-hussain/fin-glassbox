import torch
import torch.nn as nn
import torch.nn.functional as F

from config import device
import os
import dotenv
dotenv.load_dotenv()  # Load environment variables from .env file
debugOption = bool(int(os.getenv("DEBUG_MODE", "0")))


"""
This module defines the FourierGNN neural network architecture for analyzing temporal structures.
It does not expect system arguments from the command line.
"""
class FGN(nn.Module):
    """
    The Fourier Graph Network (FGN) applies frequency-domain graph convolutions across series to map structure.
    """

    def __init__(
        self,
        pre_length,
        embed_size,
        seq_length,
        hidden_size,
        hard_thresholding_fraction=1,
        hidden_size_factor=1,
        sparsity_threshold=0.01,
    ):
        if debugOption: print(f"[Debug_Output]: Function 'FGN.__init__' called with pre_length={pre_length}, embed_size={embed_size}, seq_length={seq_length}, hidden_size={hidden_size}, hard_thresholding_fraction={hard_thresholding_fraction}, hidden_size_factor={hidden_size_factor}, sparsity_threshold={sparsity_threshold}")
        """
        Initializes FGN and its trainable scaling factors, embeddings, dimensions and Linear sequential filters.

        Args:
            pre_length (int): Target projection length. Example: 12
            embed_size (int): Size per timestamp mapping. Example: 128
            seq_length (int): Received context length. Example: 12
            hidden_size (int): Capacity dimension. Example: 256
            hard_thresholding_fraction (float): Fractional multiplier defining FFT density thresholds. Example: 1.0
            hidden_size_factor (int): Amplifier applied for frequency representations. Example: 1
            sparsity_threshold (float): Filter drop rate logic scaling factor. Example: 0.01
        """
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.number_frequency = 1
        self.pre_length = pre_length
        self.seq_length = seq_length
        self.frequency_size = self.embed_size // self.number_frequency
        self.hidden_size_factor = hidden_size_factor
        self.sparsity_threshold = sparsity_threshold
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.scale = 0.02
        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.w1 = nn.Parameter(self.scale *
                               torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale *
                               torch.randn(2, self.frequency_size * self.hidden_size_factor, self.frequency_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size))
        self.w3 = nn.Parameter(self.scale *
                               torch.randn(2, self.frequency_size, self.frequency_size * self.hidden_size_factor))
        self.b3 = nn.Parameter(self.scale * torch.randn(2, self.frequency_size * self.hidden_size_factor))
        self.embeddings_10 = nn.Parameter(torch.randn(self.seq_length, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pre_length),
        )
        self.to(device)

    def tokenEmb(self, x):
        if debugOption: print(f"[Debug_Output]: Function 'tokenEmb' called with x.shape={x.shape}")
        """
        Multiplies input data spatially against uniform node-embedding characteristics.
        
        Args:
            x (Tensor): Incoming tensor structure pre-mapped sequence representations.
        
        Returns:
            Tensor: Expanded dimensional properties.
        """
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    # FourierGNN
    def fourierGC(self, x, B, N, L):
        if debugOption: print(f"[Debug_Output]: Function 'fourierGC' called with B={B}, N={N}, L={L}")
        """
        Calculates complex multiplication sequences substituting matrix Graph manipulations 
        within isolated frequency boundaries spanning across 3 cascading Linear tiers relying upon Softshrink filters.
        
        Args:
            x (Tensor): FFT-induced frequency domains.
            B (int): Batch magnitude scalar.
            N (int): Node quantity references.
            L (int): Loop/Length variables indicating observation sequences.
            
        Returns:
            Tensor: Translated Inverse complex variable domain outputs.
        """
        o1_real = torch.zeros(
            [B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
            device=x.device,
        )
        o1_imag = torch.zeros(
            [B, (N * L) // 2 + 1, self.frequency_size * self.hidden_size_factor],
            device=x.device,
        )
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        o3_real = torch.zeros(x.shape, device=x.device)
        o3_imag = torch.zeros(x.shape, device=x.device)

        o1_real = F.relu(
            torch.einsum("bli,ii->bli", x.real, self.w1[0]) - torch.einsum("bli,ii->bli", x.imag, self.w1[1]) +
            self.b1[0])

        o1_imag = F.relu(
            torch.einsum("bli,ii->bli", x.imag, self.w1[0]) + torch.einsum("bli,ii->bli", x.real, self.w1[1]) +
            self.b1[1])

        # 1 layer
        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)

        o2_real = F.relu(
            torch.einsum("bli,ii->bli", o1_real, self.w2[0]) - torch.einsum("bli,ii->bli", o1_imag, self.w2[1]) +
            self.b2[0])

        o2_imag = F.relu(
            torch.einsum("bli,ii->bli", o1_imag, self.w2[0]) + torch.einsum("bli,ii->bli", o1_real, self.w2[1]) +
            self.b2[1])

        # 2 layer
        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = x + y

        o3_real = F.relu(
            torch.einsum("bli,ii->bli", o2_real, self.w3[0]) - torch.einsum("bli,ii->bli", o2_imag, self.w3[1]) +
            self.b3[0])

        o3_imag = F.relu(
            torch.einsum("bli,ii->bli", o2_imag, self.w3[0]) + torch.einsum("bli,ii->bli", o2_real, self.w3[1]) +
            self.b3[1])

        # 3 layer
        z = torch.stack([o3_real, o3_imag], dim=-1)
        z = F.softshrink(z, lambd=self.sparsity_threshold)
        z = z + x
        z = torch.view_as_complex(z)
        return z

    def forward(self, x):
        if debugOption: print(f"[Debug_Output]: Function 'forward' called with x.shape={x.shape}")
        """
        Performs network sequence execution chaining `tokenEmb`, RFFT transformations,
        `fourierGC` filtration and returning predictions scaled matching `pre_length`.
        
        Args:
            x (Tensor): Node sequence tensor mapped by B * N * L.
        
        Returns:
            Tensor: Predicted continuous representations mapped matching prediction targets.
        """
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm="ortho")

        x = x.reshape(B, (N * L) // 2 + 1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fourierGC(x, B, N, L)

        x = x + bias

        x = x.reshape(B, (N * L) // 2 + 1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N * L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)

        return x
