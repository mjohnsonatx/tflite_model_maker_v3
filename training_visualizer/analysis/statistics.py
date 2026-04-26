"""Statistical analysis utilities."""
import numpy as np


def calculate_statistics(metrics, eval_metrics):
    """Calculate detailed statistics for the training."""
    stats = {}

    if len(metrics['loss']) > 0:
        stats['total_epochs'] = len(metrics['epoch'])
        stats['final_train_loss'] = metrics['loss'][-1]
        stats['final_val_loss'] = metrics['val_loss'][-1]
        stats['best_val_loss'] = min(metrics['val_loss'])
        stats['best_epoch'] = metrics['epoch'][metrics['val_loss'].index(min(metrics['val_loss']))]

        stats['train_loss_reduction'] = (metrics['loss'][0] - metrics['loss'][-1]) / metrics['loss'][0] * 100
        stats['val_loss_reduction'] = (metrics['val_loss'][0] - metrics['val_loss'][-1]) / metrics['val_loss'][0] * 100

        stats['final_overfit_gap'] = metrics['loss'][-1] - metrics['val_loss'][-1]
        stats['avg_overfit_gap'] = np.mean([t - v for t, v in zip(metrics['loss'], metrics['val_loss'])])

        if len(metrics['loss']) > 5:
            last_5_train = metrics['loss'][-5:]
            last_5_val = metrics['val_loss'][-5:]
            stats['train_stability'] = np.std(last_5_train)
            stats['val_stability'] = np.std(last_5_val)

        if 'learning_rate' in metrics and len(metrics['learning_rate']) > 0:
            stats['initial_lr'] = metrics['learning_rate'][0]
            stats['final_lr'] = metrics['learning_rate'][-1]
            stats['lr_reduction'] = (stats['initial_lr'] - stats['final_lr']) / stats['initial_lr'] * 100

    if eval_metrics['validation']:
        stats['val_ap'] = eval_metrics['validation'].get('AP', 0)
        stats['val_ap50'] = eval_metrics['validation'].get('AP50', 0)
        stats['val_ap75'] = eval_metrics['validation'].get('AP75', 0)

    if eval_metrics['test_CLEAN']:
        stats['test_ap'] = eval_metrics['test_CLEAN'].get('AP', 0)
        stats['test_ap50'] = eval_metrics['test_CLEAN'].get('AP50', 0)
        stats['test_ap75'] = eval_metrics['test_CLEAN'].get('AP75', 0)

    return stats