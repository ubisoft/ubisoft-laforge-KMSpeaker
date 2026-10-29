# Standard Library
import logging
from pathlib import Path

# External Modules
import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int, milestone_interval: int = 100000):
        """
        Initializes the CheckpointManager with two-tier checkpoint strategy.

        Args:
            checkpoint_dir (str): Directory where checkpoints are saved.
            max_checkpoints (int): Maximum number of regular checkpoints to keep.
            milestone_interval (int): Interval for milestone checkpoints (e.g., 100k).
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.milestone_interval = milestone_interval
        self.regular_checkpoints = []  # Temporary checkpoints
        self.milestone_checkpoints = []  # Permanent milestone checkpoints

        # Ensure the checkpoint directory exists
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _is_milestone(self, global_step: int) -> bool:
        """Check if the current step is a milestone (multiple of milestone_interval)."""
        return global_step % self.milestone_interval == 0

    def save_checkpoint(self, global_step: int, data: dict):
        """
        Saves a new checkpoint with two-tier strategy:
        - Regular checkpoints are saved every time but only keep the latest few
        - Milestone checkpoints (multiples of milestone_interval) are kept permanently

        Args:
            global_step (int): Current training step.
            data (dict): Dictionary containing the data to save in the checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{global_step}.pth"
        torch.save(data, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        if self._is_milestone(global_step):
            # This is a milestone checkpoint - keep it permanently
            self.milestone_checkpoints.append(checkpoint_path)
            logger.info(f"Milestone checkpoint saved: {checkpoint_path}")
        else:
            # This is a regular checkpoint - add to regular list
            self.regular_checkpoints.append(checkpoint_path)

            # Clean up old regular checkpoints if we exceed the limit
            if len(self.regular_checkpoints) > self.max_checkpoints:
                oldest_checkpoint = self.regular_checkpoints.pop(0)
                try:
                    oldest_checkpoint.unlink()
                    logger.info(f"Deleted old regular checkpoint: {oldest_checkpoint}")
                except Exception as e:
                    logger.error(f"Failed to delete checkpoint {oldest_checkpoint}: {e}")
