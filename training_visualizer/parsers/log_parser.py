"""Training log parsing utilities."""
import re
import ast


def parse_training_log(log_file):
    """Extract training metrics from log file - handles all progress formats."""

    metrics = {
        'epoch': [],
        'det_loss': [],
        'cls_loss': [],
        'box_loss': [],
        'reg_l2_loss': [],
        'loss': [],
        'learning_rate': [],
        'gradient_norm': [],
        'val_det_loss': [],
        'val_cls_loss': [],
        'val_box_loss': [],
        'val_reg_l2_loss': [],
        'val_loss': []
    }

    eval_metrics = {
        'validation': {},
        'test_CLEAN': {}
    }

    epoch_pattern = r'Epoch (\d+)/(\d+)'
    metrics_pattern = r'(\w+):\s+([\d.e+-]+)'
    val_metrics_pattern = r'Validation metrics:\s*({.*?})'
    test_metrics_pattern = r'Test metrics:\s*({.*?})'

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    current_epoch = None
    epoch_data_buffer = {}

    # Parse validation and test_CLEAN metrics
    val_match = re.search(val_metrics_pattern, content, re.DOTALL)
    if val_match:
        try:
            eval_metrics['validation'] = ast.literal_eval(val_match.group(1))
        except:
            pass

    test_match = re.search(test_metrics_pattern, content, re.DOTALL)
    if test_match:
        try:
            eval_metrics['test_CLEAN'] = ast.literal_eval(test_match.group(1))
        except:
            pass

    for i, line in enumerate(lines):
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            if current_epoch and epoch_data_buffer:
                if 'loss' in epoch_data_buffer and 'val_loss' in epoch_data_buffer:
                    for key in metrics.keys():
                        if key in epoch_data_buffer:
                            metrics[key].append(epoch_data_buffer[key])
                        elif key == 'epoch':
                            metrics[key].append(current_epoch)

            current_epoch = int(epoch_match.group(1))
            epoch_data_buffer = {'epoch': current_epoch}

        if (current_epoch and
                '[==============================]' in line and
                'val_loss:' in line and
                's/step' in line):

            found_metrics = re.findall(metrics_pattern, line)
            for metric_name, value in found_metrics:
                try:
                    epoch_data_buffer[metric_name] = float(value)
                except ValueError:
                    continue

    if current_epoch and epoch_data_buffer:
        if 'loss' in epoch_data_buffer and 'val_loss' in epoch_data_buffer:
            for key in metrics.keys():
                if key in epoch_data_buffer:
                    metrics[key].append(epoch_data_buffer[key])
                elif key == 'epoch':
                    metrics[key].append(current_epoch)

    return metrics, eval_metrics
