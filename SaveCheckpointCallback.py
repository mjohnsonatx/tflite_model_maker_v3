import tensorflow as tf

import OptimizedCheckpointManager




class SaveCheckpointCallback(tf.keras.callbacks.Callback):
    """
    Keras callback to save checkpoints at specified intervals.

    Integrates with OptimizedCheckpointManager to capture metrics
    from training logs and preserve optimizer state.
    """

    def __init__(
            self,
            checkpoint_manager: OptimizedCheckpointManager,
            save_every_n_epochs: int = 10,
            start_epoch: int = 0
    ):
        super().__init__()
        self.checkpoint_manager = checkpoint_manager
        self.save_every_n_epochs = save_every_n_epochs
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint at end of epoch."""
        logs = logs or {}
        current_epoch = self.start_epoch + epoch + 1

        # Save every N epochs or if early stopping triggers
        if current_epoch % self.save_every_n_epochs == 0:
            self.checkpoint_manager.save_checkpoint(
                self.model,
                epoch=current_epoch,
                metrics={k: float(v) for k, v in logs.items()}
            )

            # Check early stopping
            if self.checkpoint_manager.should_stop_early():
                self.model.stop_training = True
