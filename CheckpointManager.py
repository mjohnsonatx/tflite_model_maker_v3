import os
import json
import shutil
from typing import Tuple, Optional, Dict
import tensorflow as tf
from datetime import datetime


class CheckpointManager:
    """Manages model checkpoints for training resumption and crash recovery."""

    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.metadata_file = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._init_metadata()

    def _init_metadata(self):
        """Initialize metadata file if it doesn't exist."""
        if not os.path.exists(self.metadata_file):
            self._write_metadata({
                'checkpoints': [],
                'best_epoch': 0,
                'best_loss': float('inf'),
                'last_epoch_trained': 0
            })

    def _write_metadata(self, metadata: Dict):
        """Write metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _read_metadata(self) -> Dict:
        """Read metadata from file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'checkpoints': [], 'best_epoch': 0, 'best_loss': float('inf')}

    def save_checkpoint(
            self,
            model: tf.keras.Model,
            epoch: int,
            metrics: Optional[Dict] = None,
            is_best: bool = False
    ) -> str:
        """
        Save model checkpoint with metadata.

        Args:
            model: Keras model to save
            epoch: Current epoch number
            metrics: Dictionary of training metrics
            is_best: Whether this is the best model so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f'checkpoint_epoch_{epoch:03d}'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        # Save model weights
        model.save_weights(checkpoint_path)

        # Update metadata
        metadata = self._read_metadata()
        checkpoint_info = {
            'epoch': epoch,
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {}
        }

        metadata['checkpoints'].append(checkpoint_info)
        metadata['last_epoch_trained'] = epoch

        if is_best or (metrics and 'loss' in metrics and
                       metrics['loss'] < metadata.get('best_loss', float('inf'))):
            metadata['best_epoch'] = epoch
            metadata['best_loss'] = metrics.get('loss', float('inf')) if metrics else 0

        self._write_metadata(metadata)

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

        print(f"✓ Checkpoint saved: {checkpoint_path} (Epoch {epoch})")
        return checkpoint_path

    def load_checkpoint(
            self,
            model: tf.keras.Model,
            checkpoint_epoch: Optional[int] = None
    ) -> int:
        """
        Load model from checkpoint.

        Args:
            model: Keras model to load weights into
            checkpoint_epoch: Specific epoch to load. If None, loads latest.

        Returns:
            Epoch number of loaded checkpoint, or 0 if no checkpoint found
        """
        metadata = self._read_metadata()

        if not metadata['checkpoints']:
            print("No checkpoints found.")
            return 0

        # Find checkpoint to load
        checkpoint_to_load = None
        if checkpoint_epoch is not None:
            for cp in metadata['checkpoints']:
                if cp['epoch'] == checkpoint_epoch:
                    checkpoint_to_load = cp
                    break
        else:
            # Load latest checkpoint
            checkpoint_to_load = metadata['checkpoints'][-1]

        if checkpoint_to_load is None:
            print(f"Checkpoint for epoch {checkpoint_epoch} not found.")
            return 0

        checkpoint_path = checkpoint_to_load['checkpoint_path']

        try:
            if os.path.exists(checkpoint_path):
                model.load_weights(checkpoint_path)
                print(f"✓ Loaded checkpoint from epoch {checkpoint_to_load['epoch']}")
                print(f"  Path: {checkpoint_path}")
                if checkpoint_to_load['metrics']:
                    print(f"  Metrics: {checkpoint_to_load['metrics']}")
                return checkpoint_to_load['epoch']
            else:
                print(f"Checkpoint file not found: {checkpoint_path}")
                return 0
        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            return 0

    def load_best_checkpoint(self, model: tf.keras.Model) -> int:
        """
        Load the best checkpoint (lowest validation loss).

        Args:
            model: Keras model to load weights into

        Returns:
            Epoch number of best checkpoint
        """
        metadata = self._read_metadata()
        best_epoch = metadata.get('best_epoch', 0)

        if best_epoch == 0:
            print("No best checkpoint found.")
            return 0

        return self.load_checkpoint(model, best_epoch)

    def list_checkpoints(self) -> None:
        """Print list of available checkpoints."""
        metadata = self._read_metadata()

        if not metadata['checkpoints']:
            print("No checkpoints available.")
            return

        print("\nAvailable checkpoints:")
        print("-" * 70)
        for cp in metadata['checkpoints']:
            print(f"Epoch {cp['epoch']:3d} | {cp['timestamp']} | "
                  f"Loss: {cp['metrics'].get('loss', 'N/A')}")
        print("-" * 70)

    def _cleanup_old_checkpoints(self):
        """Keep only the most recent checkpoints."""
        metadata = self._read_metadata()

        if len(metadata['checkpoints']) <= self.max_checkpoints:
            return

        # Sort by epoch (descending) and keep only the latest
        checkpoints = sorted(metadata['checkpoints'],
                             key=lambda x: x['epoch'], reverse=True)
        checkpoints_to_keep = checkpoints[:self.max_checkpoints]
        checkpoints_to_remove = checkpoints[self.max_checkpoints:]

        for cp in checkpoints_to_remove:
            try:
                path = cp['checkpoint_path']
                if os.path.exists(path):
                    os.remove(path)
                    print(f"  Cleaned up old checkpoint: {path}")
            except Exception as e:
                print(f"  Warning: Could not remove {cp['checkpoint_path']}: {e}")

        metadata['checkpoints'] = checkpoints_to_keep
        self._write_metadata(metadata)

    def get_training_status(self) -> Dict:
        """Get current training status."""
        metadata = self._read_metadata()
        return {
            'last_epoch_trained': metadata['last_epoch_trained'],
            'best_epoch': metadata['best_epoch'],
            'best_loss': metadata['best_loss'],
            'num_checkpoints': len(metadata['checkpoints'])
        }

    def clear_checkpoints(self):
        """Delete all checkpoints."""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            self._init_metadata()
            print("✓ All checkpoints cleared.")