"""
DropBlock: A regularization method for convolutional networks
Paper: DropBlock: A regularization method for convolutional networks (NeurIPS 2018)
https://arxiv.org/abs/1810.12890
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """
    DropBlock is a form of structured dropout that drops contiguous regions
    from feature maps instead of individual activations.

    This is particularly effective for convolutional networks and segmentation tasks.

    Args:
        drop_prob: Probability of dropping a unit (0.0-1.0)
        block_size: Size of the block to drop
    """

    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Output tensor with dropblock applied
        """
        # Only apply during training
        if not self.training or self.drop_prob == 0:
            return x

        # Get dimensions
        batch_size, channels, height, width = x.size()

        # Calculate gamma (sampling probability for the mask)
        # gamma is the probability that a seed unit is chosen to be dropped
        gamma = self._compute_gamma(x)

        # Sample mask with shape (B, C, H, W)
        # First, sample from Bernoulli distribution
        mask = (torch.rand(batch_size, channels, height, width, device=x.device) < gamma).float()

        # Apply max pooling to create blocks
        # This expands each sampled point to a block
        block_mask = self._compute_block_mask(mask)

        # Normalize the output to maintain expected value
        # Count valid elements (not dropped)
        normalize_factor = block_mask.numel() / block_mask.sum()

        # Apply mask and normalize
        out = x * block_mask * normalize_factor

        return out

    def _compute_gamma(self, x):
        """
        Calculate gamma for sampling probability.

        gamma = (drop_prob / block_size^2) * (feat_size^2 / (feat_size - block_size + 1)^2)

        This ensures that the effective drop probability equals drop_prob.
        """
        _, _, height, width = x.size()

        # Calculate the valid area (area where block centers can be placed)
        valid_height = height - self.block_size + 1
        valid_width = width - self.block_size + 1

        # Calculate gamma
        gamma = (self.drop_prob / (self.block_size ** 2)) * \
                ((height * width) / (valid_height * valid_width))

        return gamma

    def _compute_block_mask(self, mask):
        """
        Expand sampled points to blocks using max pooling.

        Args:
            mask: Binary mask of shape (B, C, H, W)

        Returns:
            Block mask of shape (B, C, H, W)
        """
        batch_size, channels, height, width = mask.size()

        # Use max pooling to expand points to blocks
        # Padding ensures output size matches input size
        padding = self.block_size // 2

        # Use max_pool2d to expand the mask
        # A value of 1 in mask will create a block_size x block_size region of 1s
        block_mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=padding
        )

        # Invert: 0s become 1s (keep), 1s become 0s (drop)
        block_mask = 1 - block_mask

        return block_mask


class LinearScheduler:
    """
    Linear scheduler for DropBlock probability.
    Gradually increases drop_prob from 0 to target value.

    This helps with training stability.
    """

    def __init__(self, dropblock, start_value=0.0, stop_value=0.1, nr_steps=5000):
        """
        Args:
            dropblock: DropBlock2D module
            start_value: Initial drop probability
            stop_value: Final drop probability
            nr_steps: Number of steps to reach stop_value
        """
        self.dropblock = dropblock
        self.start_value = start_value
        self.stop_value = stop_value
        self.nr_steps = nr_steps
        self.current_step = 0

    def step(self):
        """Update drop probability"""
        if self.current_step < self.nr_steps:
            # Linear interpolation
            self.dropblock.drop_prob = self.start_value + \
                (self.stop_value - self.start_value) * (self.current_step / self.nr_steps)
            self.current_step += 1
        else:
            self.dropblock.drop_prob = self.stop_value


if __name__ == '__main__':
    # Test DropBlock
    print("Testing DropBlock2D...")

    # Create sample input
    batch_size, channels, height, width = 2, 64, 32, 32
    x = torch.randn(batch_size, channels, height, width)

    # Create DropBlock module
    dropblock = DropBlock2D(drop_prob=0.1, block_size=7)
    dropblock.train()

    # Apply DropBlock
    out = dropblock(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Input mean: {x.mean().item():.4f}")
    print(f"Output mean: {out.mean().item():.4f}")
    print(f"Fraction of zeros: {(out == 0).float().mean().item():.4f}")

    # Test in eval mode
    dropblock.eval()
    out_eval = dropblock(x)
    print(f"\nEval mode - Input equals output: {torch.allclose(x, out_eval)}")

    # Test scheduler
    print("\nTesting LinearScheduler...")
    dropblock = DropBlock2D(drop_prob=0.0, block_size=7)
    scheduler = LinearScheduler(dropblock, start_value=0.0, stop_value=0.1, nr_steps=100)

    print(f"Initial drop_prob: {dropblock.drop_prob:.4f}")
    for _ in range(50):
        scheduler.step()
    print(f"After 50 steps: {dropblock.drop_prob:.4f}")
    for _ in range(50):
        scheduler.step()
    print(f"After 100 steps: {dropblock.drop_prob:.4f}")
    for _ in range(50):
        scheduler.step()
    print(f"After 150 steps: {dropblock.drop_prob:.4f}")
