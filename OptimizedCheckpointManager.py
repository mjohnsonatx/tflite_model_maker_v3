import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import tensorflow as tf


class OptimizedCheckpointManager:
    """
    Production-grade checkpoint manager with:
    - Full optimizer state preservation
    - Proper learning rate schedule continuation
    - Callbacks integration
    - Overfitting detection
    - Best model preservation
    - Bulletproof metadata handling
    """

    def __init__(
            self,
            checkpoint_dir: str,
            max_checkpoints: int = 5,
            early_stopping_patience: int = 10,
            early_stopping_min_delta: float = 0.001
    ):
        self.checkpoint_dir = checkpoint_dir
        self.max_checkpoints = max_checkpoints
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.metadata_file = os.path.join(checkpoint_dir, 'checkpoint_metadata.json')
        self.best_model_dir = os.path.join(checkpoint_dir, 'best_model')
        self.best_optimizer_dir = os.path.join(checkpoint_dir, 'best_optimizer')

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.best_optimizer_dir, exist_ok=True)

        self._init_metadata()

    def _get_default_metadata(self) -> Dict:
        """Get default metadata structure."""
        return {
            'checkpoints': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'last_epoch_trained': 0,
            'epochs_since_improvement': 0,
            'training_history': [],
            'overfitting_detected': False,
            'learning_rate_at_best': 0.0,
        }

    def _validate_metadata(self, metadata: Dict) -> Dict:
        """
        Validate and fix metadata, ensuring all required keys exist.
        """
        default = self._get_default_metadata()

        # Add any missing keys with default values
        for key, value in default.items():
            if key not in metadata:
                metadata[key] = value

        return metadata

    def _init_metadata(self):
        """Initialize metadata file, creating or fixing as needed."""
        if not os.path.exists(self.metadata_file):
            # Create new metadata file
            self._write_metadata(self._get_default_metadata())
        else:
            # Read existing and validate
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)

                # Validate and fix
                metadata = self._validate_metadata(metadata)
                self._write_metadata(metadata)
            except Exception as e:
                print(f"⚠️  Metadata file corrupted or invalid: {e}")
                print("   Reinitializing metadata...")
                self._write_metadata(self._get_default_metadata())

    def _write_metadata(self, metadata: Dict):
        """Write metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"✗ Error writing metadata: {e}")

    def _read_metadata(self) -> Dict:
        """
        Read metadata from file with full error handling and validation.
        """
        default = self._get_default_metadata()

        if not os.path.exists(self.metadata_file):
            return default

        try:
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)

            # Validate all keys exist
            metadata = self._validate_metadata(metadata)
            return metadata

        except Exception as e:
            print(f"✗ Error reading metadata: {e}")
            print("   Using defaults...")
            return default

    def save_checkpoint(
            self,
            model: tf.keras.Model,
            epoch: int,
            metrics: Optional[Dict] = None
    ) -> str:
        """
        Save full model state including optimizer.

        Args:
            model: Keras model with optimizer
            epoch: Current epoch number
            metrics: Dictionary of metrics (from callbacks logs)

        Returns:
            Path to saved checkpoint
        """
        metrics = metrics or {}
        checkpoint_name = f'checkpoint_epoch_{epoch:03d}'
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        try:
            # Save model weights
            weights_path = os.path.join(checkpoint_path, 'model_weights')
            os.makedirs(weights_path, exist_ok=True)
            model.save_weights(weights_path)

            # Save optimizer state using tf.train_NEEDS_ANOTHER_PASS.Checkpoint
            if model.optimizer is not None:
                optimizer_path = os.path.join(checkpoint_path, 'optimizer.ckpt')
                ckpt = tf.train.Checkpoint(
                    model=model,
                    optimizer=model.optimizer,
                    step=tf.Variable(epoch, dtype=tf.int64)
                )
                ckpt.save(optimizer_path)

            # Read and update metadata
            metadata = self._read_metadata()

            train_loss = metrics.get('loss', None)
            val_loss = metrics.get('val_loss', None)
            learning_rate = metrics.get('learning_rate', None)

            checkpoint_info = {
                'epoch': epoch,
                'checkpoint_path': checkpoint_path,
                'timestamp': datetime.now().isoformat(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': learning_rate,
                'metrics': metrics
            }

            # Update training history
            metadata['training_history'].append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': learning_rate
            })

            # Check if this is the best model
            improved = False
            if val_loss is not None:
                best_loss = metadata.get('best_val_loss', float('inf'))
                if best_loss == float('inf') or val_loss < best_loss - self.early_stopping_min_delta:
                    improved = True
                    metadata['best_val_loss'] = val_loss
                    metadata['best_epoch'] = epoch
                    metadata['epochs_since_improvement'] = 0
                    metadata['learning_rate_at_best'] = learning_rate or 0.0
                    print(f"  ★ IMPROVED: Val Loss {val_loss:.6f}")
                else:
                    metadata['epochs_since_improvement'] = metadata.get('epochs_since_improvement', 0) + 1

            # Save best model separately
            if improved:
                print(f"  ★ Saving best model to {self.best_model_dir}")
                model.save_weights(self.best_model_dir)

                # Save best optimizer state
                if model.optimizer is not None:
                    best_ckpt = tf.train.Checkpoint(
                        model=model,
                        optimizer=model.optimizer
                    )
                    best_ckpt.save(os.path.join(self.best_optimizer_dir, 'optimizer.ckpt'))

                metadata['best_checkpoint_info'] = checkpoint_info

            metadata['checkpoints'].append(checkpoint_info)
            metadata['last_epoch_trained'] = epoch

            # Detect overfitting
            if len(metadata['training_history']) >= 3:
                recent = metadata['training_history'][-3:]
                train_losses = [h['train_loss'] for h in recent if h['train_loss']]
                val_losses = [h['val_loss'] for h in recent if h['val_loss']]

                if train_losses and val_losses:
                    avg_train = np.mean(train_losses)
                    avg_val = np.mean(val_losses)
                    gap = avg_val - avg_train

                    if gap > 0.1 and metadata['best_epoch'] > 0:
                        metadata['overfitting_detected'] = True
                        print(f"  ⚠️ Overfitting detected (gap: {gap:.6f})")

            self._write_metadata(metadata)
            self._cleanup_old_checkpoints(metadata)

            print(f"✓ Checkpoint saved: epoch {epoch} | "
                  f"train_loss={train_loss:.4f if train_loss else 'N/A'} | "
                  f"val_loss={val_loss:.4f if val_loss else 'N/A'}")

            return checkpoint_path

        except Exception as e:
            print(f"✗ Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def should_stop_early(self) -> bool:
        """Check if training should stop."""
        metadata = self._read_metadata()
        if metadata.get('epochs_since_improvement', 0) >= self.early_stopping_patience:
            print(f"\n⚠️ Early stopping: No improvement for {self.early_stopping_patience} epochs")
            return True
        return False

    def load_checkpoint(
            self,
            model: tf.keras.Model,
            checkpoint_epoch: Optional[int] = None,
            load_optimizer: bool = True
    ) -> int:
        """
        Load model and optimizer from checkpoint.

        Args:
            model: Keras model to load into
            checkpoint_epoch: Specific epoch. If None, loads latest.
            load_optimizer: Whether to restore optimizer state

        Returns:
            Epoch number of loaded checkpoint
        """
        metadata = self._read_metadata()
        checkpoints = metadata.get('checkpoints', [])

        if not checkpoints:
            print("No checkpoints found.")
            return 0

        # Find checkpoint to load
        checkpoint_to_load = None
        if checkpoint_epoch is not None:
            checkpoint_to_load = next(
                (cp for cp in checkpoints if cp['epoch'] == checkpoint_epoch),
                None
            )
        else:
            checkpoint_to_load = checkpoints[-1]  # Latest

        if checkpoint_to_load is None:
            print(f"Checkpoint for epoch {checkpoint_epoch} not found.")
            return 0

        checkpoint_path = checkpoint_to_load['checkpoint_path']

        try:
            # Load model weights
            weights_path = os.path.join(checkpoint_path, 'model_weights')
            if os.path.exists(weights_path):
                model.load_weights(weights_path)
                print(f"✓ Loaded model weights from epoch {checkpoint_to_load['epoch']}")

            # Load optimizer state
            if load_optimizer and model.optimizer is not None:
                optimizer_path = os.path.join(checkpoint_path, 'optimizer.ckpt')
                if os.path.exists(optimizer_path + '.index'):
                    ckpt = tf.train.Checkpoint(
                        model=model,
                        optimizer=model.optimizer
                    )
                    ckpt.restore(optimizer_path).expect_partial()
                    print(f"✓ Restored optimizer state")

            val_loss = checkpoint_to_load.get('val_loss')
            if val_loss:
                print(f"  Val Loss: {val_loss:.6f}")
            return checkpoint_to_load['epoch']

        except Exception as e:
            print(f"✗ Error loading checkpoint: {e}")
            return 0

    def load_best_checkpoint(
            self,
            model: tf.keras.Model,
            load_optimizer: bool = True
    ) -> int:
        """Load the best checkpoint."""
        metadata = self._read_metadata()
        best_epoch = metadata.get('best_epoch', 0)

        if best_epoch == 0:
            print("No best checkpoint found.")
            return 0

        # Load best model weights
        if os.path.exists(self.best_model_dir):
            try:
                model.load_weights(self.best_model_dir)
                print(f"✓ Loaded BEST model from epoch {best_epoch}")

                best_val_loss = metadata.get('best_val_loss', float('inf'))
                if best_val_loss != float('inf'):
                    print(f"  Best Val Loss: {best_val_loss:.6f}")

                # Load best optimizer if requested
                if load_optimizer and model.optimizer is not None:
                    best_ckpt_path = os.path.join(self.best_optimizer_dir, 'optimizer.ckpt')
                    if os.path.exists(best_ckpt_path + '.index'):
                        ckpt = tf.train.Checkpoint(
                            model=model,
                            optimizer=model.optimizer
                        )
                        ckpt.restore(best_ckpt_path).expect_partial()
                        print(f"✓ Restored best optimizer state")

                return best_epoch
            except Exception as e:
                print(f"✗ Error loading best model: {e}")

        return 0

    def get_training_status(self) -> Dict:
        """Get current training status with full defensive checks."""
        metadata = self._read_metadata()

        return {
            'last_epoch_trained': metadata.get('last_epoch_trained', 0),
            'best_epoch': metadata.get('best_epoch', 0),
            'best_val_loss': metadata.get('best_val_loss', float('inf')),
            'num_checkpoints': len(metadata.get('checkpoints', [])),
            'epochs_since_improvement': metadata.get('epochs_since_improvement', 0),
            'overfitting_detected': metadata.get('overfitting_detected', False),
            'learning_rate_at_best': metadata.get('learning_rate_at_best', 0.0),
        }

    def print_training_summary(self) -> None:
        """Print training summary."""
        metadata = self._read_metadata()

        training_history = metadata.get('training_history', [])
        if not training_history:
            print("No training history available.")
            return

        print("\n" + "=" * 100)
        print("TRAINING SUMMARY")
        print("=" * 100)

        last_epoch = training_history[-1].get('epoch', 0) if training_history else 0
        best_epoch = metadata.get('best_epoch', 0)
        best_val_loss = metadata.get('best_val_loss', float('inf'))

        print(f"\nEpochs trained: {last_epoch}")
        print(f"Best epoch: {best_epoch}" +
              (f" (Val Loss: {best_val_loss:.6f})" if best_val_loss != float('inf') else ""))
        print(f"Epochs since improvement: {metadata.get('epochs_since_improvement', 0)}/{self.early_stopping_patience}")

        if metadata.get('overfitting_detected'):
            print(f"⚠️ Overfitting detected")

        print(f"\nLast 5 epochs:")
        print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'LR':<15}")
        print("-" * 60)

        for h in training_history[-5:]:
            epoch_num = h.get('epoch', 0)
            train_str = f"{h.get('train_loss', 0):.6f}" if h.get('train_loss') else "N/A"
            val_str = f"{h.get('val_loss', 0):.6f}" if h.get('val_loss') else "N/A"
            lr_str = f"{h.get('learning_rate', 0):.2e}" if h.get('learning_rate') else "N/A"
            marker = " ★" if epoch_num == best_epoch else ""
            print(f"{epoch_num:<8} {train_str:<15} {val_str:<15} {lr_str:<15}{marker}")

        print("=" * 100 + "\n")

    def _cleanup_old_checkpoints(self, metadata: Dict):
        """Keep only recent checkpoints, preserve best model."""
        checkpoints = metadata.get('checkpoints', [])
        if len(checkpoints) <= self.max_checkpoints:
            return

        checkpoints = sorted(checkpoints,
                             key=lambda x: x['epoch'], reverse=True)

        best_epoch = metadata.get('best_epoch', 0)
        checkpoints_to_keep = []
        checkpoints_to_remove = []

        for i, cp in enumerate(checkpoints):
            if i < self.max_checkpoints or cp['epoch'] == best_epoch:
                checkpoints_to_keep.append(cp)
            else:
                checkpoints_to_remove.append(cp)

        for cp in checkpoints_to_remove:
            try:
                path = cp['checkpoint_path']
                if os.path.exists(path):
                    shutil.rmtree(path)
                    print(f"  Cleaned up: {Path(path).name}")
            except Exception as e:
                print(f"  Warning: {e}")

        metadata['checkpoints'] = sorted(checkpoints_to_keep,
                                         key=lambda x: x['epoch'])
        self._write_metadata(metadata)

    def list_checkpoints(self) -> None:
        """List all available checkpoints."""
        metadata = self._read_metadata()
        checkpoints = metadata.get('checkpoints', [])

        if not checkpoints:
            print("\nNo checkpoints available.")
            return

        print("\n" + "=" * 100)
        print("AVAILABLE CHECKPOINTS")
        print("=" * 100)
        print(f"{'Epoch':<8} {'Train Loss':<15} {'Val Loss':<15} {'Learning Rate':<15}")
        print("-" * 100)

        for cp in checkpoints:
            train_str = f"{cp.get('train_loss', 0):.6f}" if cp.get('train_loss') else "N/A"
            val_str = f"{cp.get('val_loss', 0):.6f}" if cp.get('val_loss') else "N/A"
            lr_str = f"{cp.get('learning_rate', 0):.2e}" if cp.get('learning_rate') else "N/A"
            marker = " ★ BEST" if cp['epoch'] == metadata.get('best_epoch', 0) else ""
            print(f"{cp['epoch']:<8} {train_str:<15} {val_str:<15} {lr_str:<15}{marker}")

        print("=" * 100 + "\n")

    def clear_checkpoints(self):
        """Delete all checkpoints."""
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.best_optimizer_dir, exist_ok=True)
        self._init_metadata()
        print("✓ All checkpoints cleared.")


