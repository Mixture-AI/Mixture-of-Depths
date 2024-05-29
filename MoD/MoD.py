import torch
import torch.nn as nn
import torch.nn.functional as F


class MoDTransformerBlock(nn.Module):
    """Wrapper class for integrating a transformer block with Mixture-of-Depths routing.

    Attributes:
        transformer_block (...): Transformer block to be wrapped.
        router_mlp (nn.Linear): MLP layer for calculating router weights.
        aux_mlp (nn.Linear): MLP layer for calculating auxiliary routing decision.
        capacity (float): Capacity of the mixture-of-depths routing. Default is 0.125.
        aux_loss (torch.Tensor): Auxiliary loss for training auxiliary MLP.

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
        super(MoDTransformerBlock, self).__init__()
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

        Note:
            Since we just to demonstrate the MoD concept, we don't consider the extra parameters
            for the transformer block.
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

        # In the train stage, we use the real top-k decision.
        # In the eval stage, we use the auxiliary router prediction decision.
        topk_decision = (
            aux_targets.bool() if self.training else aux_decision.bool()
        )

        # TODO[keli]: How to enable batch processing for the following loop?
        for b in range(B):
            # Extract tokens and router that need to go through the transformer block.
            # `unsqueeze(0)` is used to add the batch dimension back.
            # [Shape] selected_tokens_emb: (1, selected_tokens_count, hidden_size)
            selected_tokens_emb = (x[b, topk_decision[b]]).unsqueeze(0)
            # [Shape] selected_router_weights: (selected_tokens_count, 1)
            selected_router_weights = router_weights[
                b, topk_decision[b]
            ].unsqueeze(-1)

            if selected_tokens_emb.shape[1] > 0:
                # Apply the transformer block to the selected tokens.
                # [Shape] transformer_tokens_emb: (selected_tokens_count, hidden_size)
                transformer_tokens_emb = (
                    self.transformer_block(selected_tokens_emb)
                    * selected_router_weights
                ).squeeze(0)

                # Scatter the tokens into output according to the auxiliary decision.
                output[b, topk_decision[b]] = transformer_tokens_emb

        return output


if __name__ == "__main__":
    # Set the seed for reproducibility.
    torch.manual_seed(42)

    # Define the transformer block.
    transformer_block = nn.TransformerEncoderLayer(d_model=512, nhead=1)

    # Wrap the transformer block with MoD.
    mod_transformer_block = MoDTransformerBlock(
        transformer_block, hidden_size=512, capacity=0.125
    )

    # Input tensor.
    # [Shape] x: (batch_size, sequence_length, hidden_size)
    x = torch.rand(2, 20, 512)

    # Forward pass.
    output = mod_transformer_block(x)

    print(output.shape)
