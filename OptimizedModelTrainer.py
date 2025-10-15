from typing import Tuple

from tflite_model_maker import object_detector

from OptimizedCheckpointManager import OptimizedCheckpointManager
from SaveCheckpointCallback import SaveCheckpointCallback



class OptimizedModelTrainer:
    """
    Training wrapper that properly handles:
    - Checkpoint callbacks
    - Optimizer state restoration
    - Learning rate schedule continuation
    - Early stopping
    """

    def __init__(
            self,
            model_spec,
            train_data,
            valid_data,
            checkpoint_dir: str = 'model_checkpoints',
            max_checkpoints: int = 5,
            save_every_n_epochs: int = 10,
            early_stopping_patience: int = 15
    ):
        self.model_spec = model_spec
        self.train_data = train_data
        self.valid_data = valid_data
        self.save_every_n_epochs = save_every_n_epochs
        self.checkpoint_mgr = OptimizedCheckpointManager(
            checkpoint_dir,
            max_checkpoints=max_checkpoints,
            early_stopping_patience=early_stopping_patience
        )
        self.model = None

    def create_model(
            self,
            batch_size: int = 32,
            train_whole_model: bool = False,
            resume_from_checkpoint: bool = True
    ) -> Tuple[object_detector.ObjectDetector, int]:
        """Create model and optionally load from checkpoint."""
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
            start_epoch = self.checkpoint_mgr.load_checkpoint(
                self.model.model,
                load_optimizer=True
            )

        return self.model, start_epoch

    def train(
            self,
            total_epochs: int,
            batch_size: int = 32,
            train_whole_model: bool = False,
            resume_from_checkpoint: bool = True,
            use_best_model: bool = False
    ) -> object_detector.ObjectDetector:
        """
        Train with proper epoch continuation and checkpoint management.

        This directly calls model.fit() instead of model_spec.train()
        so we can inject our custom checkpoint callback.
        """
        # Create model
        self.model, start_epoch = self.create_model(
            batch_size=batch_size,
            train_whole_model=train_whole_model,
            resume_from_checkpoint=resume_from_checkpoint
        )

        if start_epoch >= total_epochs:
            print(f"\n✓ Training already complete at epoch {start_epoch}/{total_epochs}")
            if use_best_model:
                self.checkpoint_mgr.load_best_checkpoint(self.model.model)
            return self.model

        remaining_epochs = total_epochs - start_epoch

        print(f"\n{'=' * 100}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'=' * 100}")
        print(f"Total epochs: {total_epochs}")
        print(f"Starting from: Epoch {start_epoch}")
        print(f"Remaining epochs: {remaining_epochs}")
        print(f"Save checkpoint every: {self.save_every_n_epochs} epochs")
        print(f"Early stopping patience: {self.checkpoint_mgr.early_stopping_patience} epochs")
        print(f"{'=' * 100}\n")

        # Create checkpoint callback
        checkpoint_callback = SaveCheckpointCallback(
            self.checkpoint_mgr,
            save_every_n_epochs=self.save_every_n_epochs,
            start_epoch=start_epoch
        )

        try:
            # Generate datasets
            train_ds = self.train_data.gen_dataset(
                self.model_spec,
                batch_size=batch_size,
                is_training=True
            )
            val_ds = self.valid_data.gen_dataset(
                self.model_spec,
                batch_size=batch_size,
                is_training=False
            )

            # Calculate steps
            train_steps = len(self.train_data) // batch_size
            val_steps = len(self.valid_data) // batch_size

            # Setup model as train_lib.train() does
            from tensorflow_examples.lite.model_maker.core.task.object_detector import train
            train.setup_model(self.model.model, self.model_spec.config)
            train.init_experimental(self.model_spec.config)

            # Get default callbacks from train_lib
            from tensorflow_examples.lite.model_maker.core.task.object_detector import train as train_lib
            default_callbacks = train_lib.get_callbacks(
                self.model_spec.config.as_dict(),
                val_ds
            )

            # Add our checkpoint callback to the list
            callbacks = default_callbacks + [checkpoint_callback]

            print(f"Callbacks: {[type(cb).__name__ for cb in callbacks]}\n")

            # Train using model.fit() directly with all callbacks
            self.model.model.fit(
                train_ds,
                epochs=total_epochs,
                steps_per_epoch=train_steps,
                callbacks=callbacks,
                validation_data=val_ds,
                validation_steps=val_steps,
                initial_epoch=start_epoch  # ← KEY: Continue from start_epoch
            )

        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user")
            print("Saving emergency checkpoint...")
            self.checkpoint_mgr.save_checkpoint(
                self.model.model,
                epoch=start_epoch + 1,
                metrics={}
            )
            print("✓ Resume training by running again.")
            raise

        except Exception as e:
            print(f"\n✗ Error during training: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Load best model if requested
        if use_best_model:
            print("\nLoading best model...")
            self.checkpoint_mgr.load_best_checkpoint(self.model.model)

        self.checkpoint_mgr.print_training_summary()
        print("✓ Training completed!")
        return self.model

