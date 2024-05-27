import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDTransformer(nn.Module):
    """Wrapper class for integrating a transformer block with Mixture-of-Depths routing.

    Attributes:
        transformer_block (nn.Module): Transformer block to be wrapped.
        router_mlp (nn.Linear): MLP layer for calculating router weights.
        aux_mlp (nn.Linear): MLP layer for calculating auxiliary routing decision.
        capacity (float): Capacity of the mixture-of-depths routing. Default is 0.125.
        aux_loss (torch.Tensor): Auxiliary loss for training auxiliary MLP.

    Example:
        >>> # Example Usage:
        >>> transformer_block = SimpleTransformerBlock(hidden_size=512)
        >>> mod_transformer = MoDTransformer(transformer_block, hidden_size=512)
        >>> # Batch of 32 sequences, each with length 128 and hidden size 512
        >>> x = torch.randn(32, 128, 512)
        >>> output = mod_transformer(x)

        >>> # Training Example:
        >>> optimizer = torch.optim.Adam(mod_transformer.parameters(), lr=1e-4)
        >>> criterion = nn.MSELoss()
        >>> for epoch in range(10):
        >>>     mod_transformer.train()
        >>>     optimizer.zero_grad()
        >>>     output = mod_transformer(x)
        >>>     main_loss = criterion(output, x)  # This is a placeholder; use your actual task loss
        >>>     total_loss = main_loss + mod_transformer.aux_loss
        >>>     total_loss.backward()
        >>>     optimizer.step()
        >>>     print(f"Epoch {epoch+1}")
        >>>     print(
        >>>         f"Main Loss: {main_loss.item()}, "
        >>>         f"Aux Loss: {mod_transformer.aux_loss.item()}, "
        >>>         f"Total Loss: {total_loss.item()}"
        >>>     )

    Notes:
        MoD Paper Link: https://arxiv.org/pdf/2404.02258
    """

    def __init__(
        self,
        transformer_block,
        hidden_size: int,
        capacity: float = 0.125,
    ):
        """Initialize the MoD wrapped transformer block.

        Args:
            transformer_block (...): Transformer block to be wrapped.
            hidden_size (int): Hidden size of the transformer block.
            capacity (float, optional): Capacity of the mixture-of-depths routing.
                Defaults to 0.125.

        Raises:
            ValueError: If the capacity is not in the range (0, 1].

        Note:
            The default capacity of 0.125 is according to the original paper.
        """
        super(MoDTransformer, self).__init__()
        self.transformer_block = transformer_block
        self.router_mlp: nn.Linear = nn.Linear(hidden_size, 1)
        self.aux_mlp: nn.Linear = nn.Linear(hidden_size, 1)

        if capacity <= 0 or capacity > 1:
            raise ValueError(
                f"Capacity must be in the range (0, 1]. Got: {capacity}"
            )
        self.capacity = capacity

    def forward(self, x: torch.Tensor):
        """Forward pass through the MoD wrapped transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        B, sequence_length, _ = x.shape

        # Calculate scalar router weights logits.
        # [Shape] router_weights: (batch_size, sequence_length)
        router_weights: torch.Tensor = self.router_mlp(x).squeeze(-1)

        if self.training:
            # ┌────────────────────────────────────────────────────────┐
            # │                  > Training STAGE <                    │
            # └────────────────────────────────────────────────────────┘

            # Calculate top-k indices based on router weights.
            k = int(sequence_length * self.capacity)

            # [Shape] topk_indices: (batch_size, k)
            topk_indices = torch.topk(router_weights, k, dim=-1).indices

            # Generate binary labels for auxiliary MLP training.
            # [Shape] aux_targets: (batch_size, sequence_length)
            aux_targets = torch.zeros_like(router_weights)
            aux_targets.scatter_(1, topk_indices, 1.0)

            # Calculate auxiliary logits for training auxiliary MLP.
            # Stop gradient flow to the auxiliary decision. (means `detach()`)
            # [Shape] aux_logits: (batch_size, sequence_length)
            aux_logits = self.aux_mlp(x.detach()).squeeze(-1)

            # Calculate auxiliary routing decision (binary 0/1 index).
            # [Shape] aux_decision: (batch_size, sequence_length)
            aux_decision = (torch.sigmoid(aux_logits) > 0.5).float()

            # Calculate auxiliary loss (Binary Cross Entropy) and save for backward pass.
            self.aux_loss = F.binary_cross_entropy_with_logits(
                aux_logits, aux_targets
            )
        else:
            # ┌────────────────────────────────────────────────────────┐
            # │                 > Inference STAGE <                    │
            # └────────────────────────────────────────────────────────┘

            # Calculate auxiliary logits for training auxiliary MLP.
            # [Shape] aux_logits: (batch_size, sequence_length)
            aux_logits = self.aux_mlp(x.detach()).squeeze(-1)

            # Calculate auxiliary routing decision (binary 0/1 index).
            # [Shape] aux_decision: (batch_size, sequence_length)
            aux_decision = (torch.sigmoid(aux_logits) > 0.5).float()

        # Tokens not routed for specialized computation will skip it via the residual connection.
        # [Shape] output: (batch_size, sequence_length, hidden_size)
        output = x.clone()

        # Assure that the auxiliary decision is a boolean tensor.
        aux_decision = aux_decision.bool()

        for b in range(B):
            # Extract tokens and router that need to go through the transformer block.
            # [Shape] selected_tokens_emb: (selected_tokens_count, hidden_size)
            selected_tokens_emb = x[b, aux_decision[b]]
            # [Shape] selected_router_weights: (selected_tokens_count, 1)
            selected_router_weights = router_weights[
                b, aux_decision[b]
            ].unsqueeze(-1)

            if selected_tokens_emb.shape[0] > 0:
                # Apply the transformer block to the selected tokens.
                transformer_tokens_emb = (
                    self.transformer_block(selected_tokens_emb)
                    * selected_router_weights
                )

                # Scatter the tokens into output according to the auxiliary decision.
                output[b, aux_decision[b]] = transformer_tokens_emb

        return output
