from typing import Tuple
from tflite_model_maker import object_detector
from CheckpointManager import CheckpointManager


class ModelTrainerWithCheckpoints:
    """Wrapper for training with checkpoint support."""

    def __init__(
            self,
            model_spec,
            train_data,
            valid_data,
            checkpoint_dir: str = 'model_checkpoints',
            max_checkpoints: int = 5,
            save_every_n_epochs: int = 10
    ):
        """
        Initialize trainer with checkpoint support.

        Args:
            model_spec: EfficientDetSpec instance
            train_data: Training dataset
            valid_data: Validation dataset
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum checkpoints to keep
            save_every_n_epochs: Save checkpoint every N epochs
        """
        self.model_spec = model_spec
        self.train_data = train_data
        self.valid_data = valid_data
        self.save_every_n_epochs = save_every_n_epochs
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir, max_checkpoints)
        self.model = None

    def create_model(
            self,
            batch_size: int = 32,
            train_whole_model: bool = False,
            resume_from_checkpoint: bool = True
    ) -> Tuple[object_detector.ObjectDetector, int]:
        """
        Create model and optionally load from checkpoint.

        Args:
            batch_size: Training batch size
            train_whole_model: Whether to train_NEEDS_ANOTHER_PASS whole model
            resume_from_checkpoint: Whether to resume from existing checkpoint

        Returns:
            Tuple of (model, starting_epoch)
        """
        print("Creating model...")
        self.model = object_detector.create(
            train_data=self.train_data,
            model_spec=self.model_spec,
            batch_size=batch_size,
            train_whole_model=train_whole_model,
            validation_data=self.valid_data,
            epochs=0,
            do_train=False
        )

        start_epoch = 0
        if resume_from_checkpoint:
            start_epoch = self.checkpoint_mgr.load_checkpoint(self.model.model)

        return self.model, start_epoch

    def train(
            self,
            total_epochs: int,
            batch_size: int = 32,
            train_whole_model: bool = False,
            resume_from_checkpoint: bool = True
    ) -> object_detector.ObjectDetector:
        """
        Train model with checkpointing.

        Args:
            total_epochs: Total epochs to train_NEEDS_ANOTHER_PASS
            batch_size: Batch size
            train_whole_model: Whether to train_NEEDS_ANOTHER_PASS whole model
            resume_from_checkpoint: Whether to resume from checkpoint

        Returns:
            Trained model
        """
        # Create model
        self.model, start_epoch = self.create_model(
            batch_size=batch_size,
            train_whole_model=train_whole_model,
            resume_from_checkpoint=resume_from_checkpoint
        )

        if start_epoch >= total_epochs:
            print(f"\n✓ Training already complete at epoch {start_epoch}/{total_epochs}")
            return self.model

        remaining_epochs = total_epochs - start_epoch

        print(f"\n{'=' * 70}")
        print(f"Starting training from epoch {start_epoch + 1} to {total_epochs}")
        print(f"Remaining epochs: {remaining_epochs}")
        print(f"Checkpoints will be saved every {self.save_every_n_epochs} epochs")
        print(f"{'=' * 70}\n")

        try:
            # Train for remaining epochs
            for epoch in range(start_epoch, total_epochs):
                print(f"\n--- Epoch {epoch + 1}/{total_epochs} ---")

                # Train the model for 1 epoch
                self.model.train(
                    train_data=self.train_data,
                    validation_data=self.valid_data,
                    epochs=1,
                    batch_size=batch_size
                )

                current_epoch = epoch + 1

                # Save checkpoint at specified intervals
                if current_epoch % self.save_every_n_epochs == 0 or \
                        current_epoch == total_epochs:
                    self.checkpoint_mgr.save_checkpoint(
                        self.model.model,
                        epoch=current_epoch,
                        metrics={'epoch': current_epoch}
                    )

        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user")
            print("Saving emergency checkpoint...")
            self.checkpoint_mgr.save_checkpoint(
                self.model.model,
                epoch=epoch + 1,
                metrics={'interrupted': True}
            )
            print("✓ Emergency checkpoint saved. You can resume training later.")
            raise
        except Exception as e:
            print(f"\n✗ Error during training: {e}")
            print("Saving emergency checkpoint...")
            self.checkpoint_mgr.save_checkpoint(
                self.model.model,
                epoch=epoch + 1,
                metrics={'error': str(e)}
            )
            raise

        print("\n✓ Training completed!")
        return self.model