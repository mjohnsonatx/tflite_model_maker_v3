import os
import sys


class DualLogger:
    """Captures stdout/stderr and writes to both console and file."""

    def __init__(self, filepath, mode='w'):
        self.terminal = sys.stdout if mode == 'w' else sys.stderr
        self.log = open(filepath, mode, encoding='utf-8')
        self.is_stderr = (mode != 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure immediate write to file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def save_specifications(
    export_dir,
    batch_size,
    epochs,
    hparams,
    train_whole_model,
    backbone,
    architecture,
    mixed_precision_policy=None
):
    """Save training specifications to a formatted text file."""

    specs_filepath = os.path.join(export_dir, "training_specs.txt")

    with open(specs_filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRAINING SPECIFICATIONS\n")
        f.write("=" * 80 + "\n\n")

        # Basic training configuration
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Batch Size:               {batch_size}\n")
        f.write(f"Epochs:                  {epochs}\n")
        f.write(f"Train Whole Model:       {train_whole_model}\n")
        f.write(f"Architecture:            {architecture}\n")
        f.write(f"Backbone:                {backbone}\n")
        if mixed_precision_policy is not None:
            f.write(f"Mixed Precision Policy:  {mixed_precision_policy}\n")
        else:
            f.write(f"Mixed Precision Policy:  (not specified)\n")
        f.write("\n")

        # Model specification
        f.write("MODEL SPECIFICATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model Name:              efficientdet-lite0\n")
        f.write(f"URI:                     https://tfhub.dev/tensorflow/efficientdet/lite0/feature-vector/1\n")
        f.write("\n")

        # Hyperparameters - organized by category
        f.write("HYPERPARAMETERS\n")
        f.write("-" * 80 + "\n\n")

        # Model Architecture
        f.write("MODEL ARCHITECTURE:\n")
        f.write(f"  backbone_name:         {hparams['backbone_name']}\n")
        f.write(f"  num_classes:           {hparams['num_classes']}\n")
        f.write(f"  image_size:            {hparams['image_size']}\n")
        f.write("\n")

        # NMS Configurations
        f.write("NMS CONFIGURATIONS:\n")
        for key, value in hparams['nms_configs'].items():
            f.write(f"  {key}:".ljust(28) + f"{value}\n")
        f.write("\n")

        # Loss and Optimization Parameters
        f.write("LOSS AND OPTIMIZATION PARAMETERS:\n")
        f.write(f"  alpha:                 {hparams['alpha']}\n")
        f.write(f"  gamma:                 {hparams['gamma']}\n")
        f.write(f"  delta:                 {hparams['delta']}\n")
        f.write(f"  box_loss_weight:       {hparams['box_loss_weight']}\n")
        f.write(f"  label_smoothing:       {hparams['label_smoothing']}\n")
        f.write("\n")

        # Learning Rate Schedule
        f.write("LEARNING RATE SCHEDULE:\n")
        f.write(f"  learning_rate:         {hparams['learning_rate']}\n")
        f.write(f"  lr_warmup_init:        {hparams['lr_warmup_init']}\n")
        f.write(f"  lr_warmup_epoch:       {hparams['lr_warmup_epoch']}\n")
        f.write(f"  lr_decay_method:       {hparams['lr_decay_method']}\n")
        f.write(f"  first_lr_drop_epoch:   {hparams['first_lr_drop_epoch']}\n")
        f.write(f"  second_lr_drop_epoch:  {hparams['second_lr_drop_epoch']}\n")
        f.write(f"  poly_lr_power:         {hparams['poly_lr_power']}\n")
        f.write("\n")

        # Optimization
        f.write("OPTIMIZATION:\n")
        f.write(f"  momentum:              {hparams['momentum']}\n")
        f.write(f"  optimizer:             {hparams['optimizer']}\n")
        f.write(f"  weight_decay:          {hparams['weight_decay']}\n")
        f.write(f"  clip_gradients_norm:   {hparams['clip_gradients_norm']}\n")
        f.write(f"  moving_average_decay:  {hparams['moving_average_decay']}\n")
        f.write("\n")

        # Augmentation
        f.write("AUGMENTATION:\n")
        f.write(f"  input_rand_hflip:      {hparams['input_rand_hflip']}\n")
        f.write(f"  jitter_min:            {hparams['jitter_min']}\n")
        f.write(f"  jitter_max:            {hparams['jitter_max']}\n")
        f.write(f"  autoaugment_policy:    {hparams['autoaugment_policy']}\n")
        f.write(f"  grid_mask:             {hparams['grid_mask']}\n")
        f.write("\n")

        # Anchors / Detection Architecture
        f.write("ANCHORS / DETECTION ARCHITECTURE:\n")
        f.write(f"  anchor_scale:          {hparams['anchor_scale']}\n")
        f.write(f"  aspect_ratios:         {hparams['aspect_ratios']}\n")
        f.write(f"  num_scales:            {hparams['num_scales']}\n")
        f.write("\n")

        # Training Settings
        f.write("TRAINING SETTINGS:\n")
        f.write(f"  skip_crowd_during_training: {hparams['skip_crowd_during_training']}\n")
        f.write(f"  max_instances_per_image:    {hparams['max_instances_per_image']}\n")
        f.write(f"  mixed_precision:            {hparams['mixed_precision']}\n")
        f.write(f"  drop_remainder:             {hparams['drop_remainder']}\n")
        f.write(f"  map_freq:                   {hparams['map_freq']}\n")
        f.write(f"  num_epochs:                 {hparams['num_epochs']}\n")
        f.write("\n")

        f.write("=" * 80 + "\n")

    return specs_filepath
