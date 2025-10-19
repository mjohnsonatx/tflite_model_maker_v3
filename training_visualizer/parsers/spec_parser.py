"""Model specification parsing utilities."""
import re


def parse_model_specs(spec_file):
    """Parse ALL model specifications from training_specs.txt"""
    specs = {
        'main': {},
        'hyperparameters': {}
    }

    try:
        with open(spec_file, 'r', encoding='utf-8') as f:
            content = f.read()

        main_patterns = {
            'batch_size': r'Batch Size:\s+(\d+)',
            'epochs': r'Epochs:\s+(\d+)',
            'architecture': r'Architecture:\s+(\S+)',
            'model_name': r'Model Name:\s+(.+)',
            'train_whole_model': r'Train Whole Model:\s+(\w+)'
        }

        for key, pattern in main_patterns.items():
            match = re.search(pattern, content)
            if match:
                specs['main'][key] = match.group(1).strip()

        hyperparameter_patterns = {
            'backbone_name': r'backbone_name:\s+(\S+)',
            'num_classes': r'num_classes:\s+(\d+)',
            'image_size': r'image_size:\s+(\d+)',
            'nms_method': r'method:\s+(\S+)',
            'iou_thresh': r'iou_thresh:\s+([\d.]+)',
            'score_thresh': r'score_thresh:\s+([\d.]+)',
            'sigma': r'sigma:\s+([\d.]+)',
            'pyfunc': r'pyfunc:\s+(\w+)',
            'max_nms_inputs': r'max_nms_inputs:\s+(\d+)',
            'max_output_size': r'max_output_size:\s+(\d+)',
            'alpha': r'alpha:\s+([\d.]+)',
            'gamma': r'gamma:\s+([\d.]+)',
            'delta': r'delta:\s+([\d.]+)',
            'box_loss_weight': r'box_loss_weight:\s+([\d.]+)',
            'label_smoothing': r'label_smoothing:\s+([\d.]+)',
            'learning_rate': r'learning_rate:\s+([\d.]+)',
            'lr_warmup_init': r'lr_warmup_init:\s+([\d.]+)',
            'lr_warmup_epoch': r'lr_warmup_epoch:\s+([\d.]+)',
            'lr_decay_method': r'lr_decay_method:\s+(\S+)',
            'first_lr_drop_epoch': r'first_lr_drop_epoch:\s+([\d.]+)',
            'second_lr_drop_epoch': r'second_lr_drop_epoch:\s+([\d.]+)',
            'poly_lr_power': r'poly_lr_power:\s+([\d.]+)',
            'momentum': r'momentum:\s+([\d.]+)',
            'optimizer': r'optimizer:\s+(\S+)',
            'weight_decay': r'weight_decay:\s+([\d.e-]+)',
            'clip_gradients_norm': r'clip_gradients_norm:\s+([\d.]+)',
            'moving_average_decay': r'moving_average_decay:\s+([\d.]+)',
            'input_rand_hflip': r'input_rand_hflip:\s+(\w+)',
            'jitter_min': r'jitter_min:\s+([\d.]+)',
            'jitter_max': r'jitter_max:\s+([\d.]+)',
            'autoaugment_policy': r'autoaugment_policy:\s+(\S+)',
            'grid_mask': r'grid_mask:\s+(\w+)',
            'anchor_scale': r'anchor_scale:\s+([\d.]+)',
            'aspect_ratios': r'aspect_ratios:\s+(\[[\d.,\s]+\])',
            'num_scales': r'num_scales:\s+(\d+)',
            'skip_crowd_during_training': r'skip_crowd_during_training:\s+(\w+)',
            'max_instances_per_image': r'max_instances_per_image:\s+(\d+)',
            'mixed_precision': r'mixed_precision:\s+(\w+)',
            'drop_remainder': r'drop_remainder:\s+(\w+)',
            'map_freq': r'map_freq:\s+(\d+)',
        }

        for key, pattern in hyperparameter_patterns.items():
            match = re.search(pattern, content)
            if match:
                specs['hyperparameters'][key] = match.group(1).strip()

    except FileNotFoundError:
        print("⚠️  Model specifications file not found")

    return specs