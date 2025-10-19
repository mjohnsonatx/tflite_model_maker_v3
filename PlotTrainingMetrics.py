import re
import ast
import os
import tempfile
import webbrowser
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


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

    # Additional metrics for evaluation
    eval_metrics = {
        'validation': {},
        'test': {}
    }

    # Pattern to match the epoch number
    epoch_pattern = r'Epoch (\d+)/(\d+)'
    # Pattern to match all metrics
    metrics_pattern = r'(\w+):\s+([\d.e+-]+)'
    # Pattern to match validation/test metrics dictionaries
    val_metrics_pattern = r'Validation metrics:\s*({.*?})'
    test_metrics_pattern = r'Test metrics:\s*({.*?})'

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    current_epoch = None
    epoch_data_buffer = {}

    # Parse validation and test metrics
    val_match = re.search(val_metrics_pattern, content, re.DOTALL)
    if val_match:
        try:
            eval_metrics['validation'] = ast.literal_eval(val_match.group(1))
        except:
            pass

    test_match = re.search(test_metrics_pattern, content, re.DOTALL)
    if test_match:
        try:
            eval_metrics['test'] = ast.literal_eval(test_match.group(1))
        except:
            pass

    for i, line in enumerate(lines):
        # Check for epoch start
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            # Save previous epoch data if exists
            if current_epoch and epoch_data_buffer:
                if 'loss' in epoch_data_buffer and 'val_loss' in epoch_data_buffer:
                    for key in metrics.keys():
                        if key in epoch_data_buffer:
                            metrics[key].append(epoch_data_buffer[key])
                        elif key == 'epoch':
                            metrics[key].append(current_epoch)

            # Start new epoch
            current_epoch = int(epoch_match.group(1))
            epoch_data_buffer = {'epoch': current_epoch}

        # Look for completed progress bar with validation metrics
        if (current_epoch and
                '[==============================]' in line and
                'val_loss:' in line and
                's/step' in line):

            # Extract all metrics from this line
            found_metrics = re.findall(metrics_pattern, line)

            for metric_name, value in found_metrics:
                try:
                    epoch_data_buffer[metric_name] = float(value)
                except ValueError:
                    continue

    # Don't forget the last epoch
    if current_epoch and epoch_data_buffer:
        if 'loss' in epoch_data_buffer and 'val_loss' in epoch_data_buffer:
            for key in metrics.keys():
                if key in epoch_data_buffer:
                    metrics[key].append(epoch_data_buffer[key])
                elif key == 'epoch':
                    metrics[key].append(current_epoch)

    return metrics, eval_metrics


def parse_model_specs(spec_file):
    """Parse ALL model specifications from training_specs.txt"""
    specs = {
        'main': {},
        'hyperparameters': {}
    }

    try:
        with open(spec_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Main specifications
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

        # All hyperparameters - more comprehensive parsing
        hyperparameter_patterns = {
            # Model Architecture
            'backbone_name': r'backbone_name:\s+(\S+)',
            'num_classes': r'num_classes:\s+(\d+)',
            'image_size': r'image_size:\s+(\d+)',

            # NMS Configurations
            'nms_method': r'method:\s+(\S+)',
            'iou_thresh': r'iou_thresh:\s+([\d.]+)',
            'score_thresh': r'score_thresh:\s+([\d.]+)',
            'sigma': r'sigma:\s+([\d.]+)',
            'pyfunc': r'pyfunc:\s+(\w+)',
            'max_nms_inputs': r'max_nms_inputs:\s+(\d+)',
            'max_output_size': r'max_output_size:\s+(\d+)',

            # Loss and Optimization
            'alpha': r'alpha:\s+([\d.]+)',
            'gamma': r'gamma:\s+([\d.]+)',
            'delta': r'delta:\s+([\d.]+)',
            'box_loss_weight': r'box_loss_weight:\s+([\d.]+)',
            'label_smoothing': r'label_smoothing:\s+([\d.]+)',

            # Learning Rate Schedule
            'learning_rate': r'learning_rate:\s+([\d.]+)',
            'lr_warmup_init': r'lr_warmup_init:\s+([\d.]+)',
            'lr_warmup_epoch': r'lr_warmup_epoch:\s+([\d.]+)',
            'lr_decay_method': r'lr_decay_method:\s+(\S+)',
            'first_lr_drop_epoch': r'first_lr_drop_epoch:\s+([\d.]+)',
            'second_lr_drop_epoch': r'second_lr_drop_epoch:\s+([\d.]+)',
            'poly_lr_power': r'poly_lr_power:\s+([\d.]+)',

            # Optimization
            'momentum': r'momentum:\s+([\d.]+)',
            'optimizer': r'optimizer:\s+(\S+)',
            'weight_decay': r'weight_decay:\s+([\d.e-]+)',
            'clip_gradients_norm': r'clip_gradients_norm:\s+([\d.]+)',
            'moving_average_decay': r'moving_average_decay:\s+([\d.]+)',

            # Augmentation
            'input_rand_hflip': r'input_rand_hflip:\s+(\w+)',
            'jitter_min': r'jitter_min:\s+([\d.]+)',
            'jitter_max': r'jitter_max:\s+([\d.]+)',
            'autoaugment_policy': r'autoaugment_policy:\s+(\S+)',
            'grid_mask': r'grid_mask:\s+(\w+)',

            # Anchors/Detection
            'anchor_scale': r'anchor_scale:\s+([\d.]+)',
            'aspect_ratios': r'aspect_ratios:\s+(\[[\d.,\s]+\])',
            'num_scales': r'num_scales:\s+(\d+)',

            # Training Settings
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


def create_evaluation_metrics_plot(eval_metrics, full_size=False):
    """Create a line plot for AP/AR evaluation metrics."""

    fig = go.Figure()

    # Prepare data for both validation and test sets
    categories = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                  'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']

    val_values = []
    test_values = []

    for cat in categories:
        val_values.append(eval_metrics['validation'].get(cat, 0))
        test_values.append(eval_metrics['test'].get(cat, 0))

    # Create line plots with markers
    fig.add_trace(go.Scatter(
        name='Validation',
        x=categories,
        y=val_values,
        mode='lines+markers',
        line=dict(color='rgb(31, 119, 180)', width=3 if full_size else 2),
        marker=dict(size=10 if full_size else 8, symbol='circle'),
        text=[f'{v:.3f}' for v in val_values],
        textposition='top center',
        textfont=dict(size=10 if full_size else 9)
    ))

    fig.add_trace(go.Scatter(
        name='Test',
        x=categories,
        y=test_values,
        mode='lines+markers',
        line=dict(color='rgb(255, 127, 14)', width=3 if full_size else 2),
        marker=dict(size=10 if full_size else 8, symbol='diamond'),
        text=[f'{v:.3f}' for v in test_values],
        textposition='bottom center',
        textfont=dict(size=10 if full_size else 9)
    ))

    # Add a subtle difference area between the lines
    fig.add_trace(go.Scatter(
        x=categories + categories[::-1],
        y=val_values + test_values[::-1],
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.1)',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        title={
            'text': 'Model Evaluation Metrics Comparison (Validation vs Test)',
            'font': {'size': 24 if full_size else 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_tickangle=-45,
        height=600 if full_size else 500,
        yaxis_title='Score',
        xaxis_title='Metric',
        yaxis=dict(
            gridcolor='rgba(128, 128, 128, 0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(128, 128, 128, 0.3)',
            range=[0, 1.05]
        ),
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.3)',
        paper_bgcolor='white',
        font=dict(size=12 if full_size else 11)
    )

    return fig


def create_model_specs_card(specs):
    """Create an HTML card displaying model specifications with collapsible hyperparameters."""

    html = """
    <div style="background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h2 style="color: #333; margin-bottom: 20px; border-bottom: 2px solid #667eea; 
                   padding-bottom: 10px;">🔧 Model Configuration</h2>

        <!-- Main Parameters -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 15px; margin-bottom: 20px;">
    """

    # Main parameters with icons
    main_items = [
        ('Model Name', specs['main'].get('model_name', 'N/A'), '🏷️'),
        ('Architecture', specs['main'].get('architecture', 'N/A'), '🏗️'),
        ('Batch Size', specs['main'].get('batch_size', 'N/A'), '📦'),
        ('Epochs', specs['main'].get('epochs', 'N/A'), '🔄'),
        ('Train Whole Model', specs['main'].get('train_whole_model', 'N/A'), '🎯')
    ]

    for label, value, icon in main_items:
        html += f"""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
                    padding: 12px; border-radius: 8px; 
                    border-left: 3px solid #667eea;">
            <div style="display: flex; align-items: center; margin-bottom: 3px;">
                <span style="font-size: 1.2em; margin-right: 8px;">{icon}</span>
                <strong style="color: #495057; font-size: 0.9em;">{label}</strong>
            </div>
            <div style="font-size: 1.05em; color: #212529; font-family: monospace; margin-left: 28px;">
                {value}
            </div>
        </div>
        """

    html += """
        </div>

        <!-- Collapsible Hyperparameters Section -->
        <details style="margin-top: 20px; border: 1px solid #dee2e6; border-radius: 10px; 
                        overflow: hidden; background: white;">
            <summary style="cursor: pointer; font-weight: bold; color: #667eea; 
                          font-size: 1.1em; padding: 15px; background: #f8f9fa;
                          border-bottom: 1px solid #dee2e6;">
                ⚙️ Advanced Hyperparameters (Click to expand)
            </summary>

            <div style="padding: 20px; background: white;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                           gap: 30px;">
    """

    # Group hyperparameters by category
    hyperparam_groups = [
        ('🎨 Model Architecture', ['backbone_name', 'num_classes', 'image_size']),
        ('🎯 NMS Configuration', ['nms_method', 'iou_thresh', 'score_thresh', 'sigma',
                                 'pyfunc', 'max_nms_inputs', 'max_output_size']),
        ('📉 Loss Parameters', ['alpha', 'gamma', 'delta', 'box_loss_weight', 'label_smoothing']),
        ('📈 Learning Rate', ['learning_rate', 'lr_warmup_init', 'lr_warmup_epoch',
                             'lr_decay_method', 'first_lr_drop_epoch', 'second_lr_drop_epoch',
                             'poly_lr_power']),
        ('⚡ Optimization', ['momentum', 'optimizer', 'weight_decay', 'clip_gradients_norm',
                            'moving_average_decay']),
        ('🔄 Augmentation', ['input_rand_hflip', 'jitter_min', 'jitter_max',
                            'autoaugment_policy', 'grid_mask']),
        ('📐 Anchors/Detection', ['anchor_scale', 'aspect_ratios', 'num_scales']),
        ('🎲 Training Settings', ['skip_crowd_during_training', 'max_instances_per_image',
                                 'mixed_precision', 'drop_remainder', 'map_freq'])
    ]

    for group_name, params in hyperparam_groups:
        html += f"""
            <div style="background: #f8f9fa; border-radius: 8px; padding: 15px; 
                       border: 1px solid #e9ecef;">
                <h4 style="color: #495057; margin: 0 0 15px 0; padding-bottom: 10px; 
                          border-bottom: 2px solid #667eea; font-size: 0.95em;">
                    {group_name}
                </h4>
                <table style="width: 100%; border-collapse: collapse;">
        """

        for param in params:
            if param in specs['hyperparameters']:
                param_label = param.replace('_', ' ').title()
                param_value = specs['hyperparameters'][param]

                # Highlight certain important values
                value_style = "color: #212529;"
                if param_value in ['True', 'False']:
                    value_color = '#28a745' if param_value == 'True' else '#dc3545'
                    value_style = f"color: {value_color}; font-weight: 600;"
                elif param in ['learning_rate', 'batch_size', 'epochs']:
                    value_style = "color: #667eea; font-weight: 600;"

                html += f"""
                <tr style="border-bottom: 1px solid #e9ecef;">
                    <td style="padding: 8px 12px; color: #6c757d; font-size: 0.85em; 
                              width: 60%; vertical-align: middle;">
                        {param_label}
                    </td>
                    <td style="padding: 8px 12px; font-family: 'Courier New', monospace; 
                              font-size: 0.85em; text-align: right; {value_style}">
                        {param_value}
                    </td>
                </tr>
                """

        html += """
                </table>
            </div>
        """

    html += """
                </div>
            </div>
        </details>
    </div>

    <style>
        details summary::-webkit-details-marker {
            display: none;
        }

        details summary::before {
            content: '▶';
            display: inline-block;
            margin-right: 8px;
            transition: transform 0.3s;
        }

        details[open] summary::before {
            transform: rotate(90deg);
        }

        details summary:hover {
            background: #e9ecef !important;
        }

        table tr:hover {
            background: rgba(102, 126, 234, 0.05);
        }

        table tr:last-child {
            border-bottom: none !important;
        }
    </style>
    """

    return html


def create_interactive_plot(metrics, full_size=False):
    """Create highly detailed interactive plot with Plotly - with better grouping."""

    epochs = metrics['epoch']

    # Adjust height based on whether it's full size
    height = 900 if full_size else 700

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    # Group plot configurations by type for better visual organization
    plot_groups = {
        'Main Losses': [
            ('loss', 'Training Loss', 'rgb(31, 119, 180)', 'solid', True, 3),
            ('val_loss', 'Validation Loss', 'rgb(255, 127, 14)', 'solid', True, 3),
        ],
        'Component Losses': [
            ('det_loss', 'Detection Loss (Train)', 'rgb(44, 160, 44)', 'dash', True, 2),
            ('val_det_loss', 'Detection Loss (Val)', 'rgb(152, 223, 138)', 'dash', True, 2),
            ('cls_loss', 'Classification Loss (Train)', 'rgb(214, 39, 40)', 'dot', True, 2),
            ('val_cls_loss', 'Classification Loss (Val)', 'rgb(255, 152, 150)', 'dot', True, 2),
            ('box_loss', 'Box Loss (Train)', 'rgb(148, 103, 189)', 'dashdot', True, 2),
            ('val_box_loss', 'Box Loss (Val)', 'rgb(197, 176, 213)', 'dashdot', True, 2),
        ],
        'Regularization': [
            ('reg_l2_loss', 'L2 Regularization (Train)', 'rgb(140, 86, 75)', 'longdash', True, 1.5),
            ('val_reg_l2_loss', 'L2 Regularization (Val)', 'rgb(196, 156, 148)', 'longdash', True, 1.5),
        ],
        'Learning Dynamics': [
            ('learning_rate', 'Learning Rate', 'rgb(23, 190, 207)', 'solid', False, 2.5),
            ('gradient_norm', 'Gradient Norm', 'rgb(127, 127, 127)', 'longdashdot', False, 2),
        ]
    }

    # Add traces with group information in legend
    for group_name, group_configs in plot_groups.items():
        for metric_key, label, color, dash, use_primary_y, width in group_configs:
            if metric_key in metrics and len(metrics[metric_key]) > 0:
                # Make the main losses visible by default
                visible = True if metric_key in ['loss', 'val_loss'] else 'legendonly'

                # Add group name to legend for better organization
                legend_group = group_name

                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=metrics[metric_key],
                        name=label,
                        mode='lines+markers',
                        line=dict(color=color, dash=dash, width=width),
                        marker=dict(size=4 if not full_size else 6),
                        hovertemplate='<b>' + label + '</b><br>' +
                                      'Epoch: %{x}<br>' +
                                      'Value: %{y:.6f}<br>' +
                                      '<extra></extra>',
                        visible=visible,
                        legendgroup=legend_group,
                        legendgrouptitle_text=legend_group if metric_key == group_configs[0][0] else None
                    ),
                    secondary_y=not use_primary_y
                )

    # Calculate additional statistics for annotations
    if len(metrics['val_loss']) > 0:
        best_epoch = metrics['epoch'][metrics['val_loss'].index(min(metrics['val_loss']))]
        best_val_loss = min(metrics['val_loss'])

        # Add annotation for best validation loss
        fig.add_annotation(
            x=best_epoch,
            y=best_val_loss,
            text=f"Best Val Loss<br>{best_val_loss:.4f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=-50,
            ay=-40,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="red",
            borderwidth=2,
            font=dict(size=12 if not full_size else 14)
        )

    # Update layout
    fig.update_xaxes(
        title_text="Epoch",
        gridcolor='rgba(128, 128, 128, 0.2)',
        showgrid=True,
        zeroline=True,
        title_font=dict(size=14 if not full_size else 16)
    )

    fig.update_yaxes(
        title_text="Loss Value",
        secondary_y=False,
        gridcolor='rgba(128, 128, 128, 0.2)',
        showgrid=True,
        title_font=dict(size=14 if not full_size else 16)
    )

    fig.update_yaxes(
        title_text="Learning Rate / Gradient Norm",
        secondary_y=True,
        gridcolor='rgba(128, 128, 128, 0.1)',
        showgrid=True,
        title_font=dict(size=14 if not full_size else 16)
    )

    fig.update_layout(
        title={
            'text': 'Detailed Training Metrics Visualization',
            'font': {'size': 24 if full_size else 20, 'color': 'rgb(50, 50, 50)'},
            'x': 0.5,
            'xanchor': 'center'
        },
        hovermode='x unified',
        height=height,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255, 255, 255, 0.95)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            font=dict(size=11 if not full_size else 13),
            groupclick="toggleitem",
            tracegroupgap=25,
            title=dict(text="Metric Groups", font=dict(size=12 if not full_size else 14))
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=11 if not full_size else 13),
        margin=dict(r=250)  # Add right margin for legend
    )

    # Add range slider
    fig.update_xaxes(rangeslider_visible=True)

    return fig


def create_detailed_subplots(metrics, full_size=False):
    """Create detailed subplot analysis with multiple perspectives."""

    epochs = metrics['epoch']

    # Adjust dimensions based on whether it's full size
    height = 1400 if full_size else 1000
    title_size = 18 if full_size else 14

    # Create subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Overall Loss Comparison',
            'Loss Components Breakdown',
            'Training vs Validation Gap',
            'Learning Dynamics',
            'Loss Ratios and Trends',
            'Convergence Analysis'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )

    # Update subplot title font sizes
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=title_size)

    # 1. Overall Loss Comparison
    if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['loss'], name='Training Loss',
                       line=dict(color='blue', width=3 if full_size else 2),
                       mode='lines+markers', marker=dict(size=4)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['val_loss'], name='Validation Loss',
                       line=dict(color='red', width=3 if full_size else 2),
                       mode='lines+markers', marker=dict(size=4)),
            row=1, col=1
        )

        # Add smoothed trend lines
        if len(epochs) > 5:
            window = min(5, len(epochs) // 4)
            train_smooth = np.convolve(metrics['loss'], np.ones(window) / window, mode='valid')
            val_smooth = np.convolve(metrics['val_loss'], np.ones(window) / window, mode='valid')
            epochs_smooth = epochs[window // 2:-(window // 2) if window > 1 else None]

            fig.add_trace(
                go.Scatter(x=epochs_smooth, y=train_smooth, name='Training (Smoothed)',
                           line=dict(color='lightblue', width=2 if full_size else 1, dash='dash'),
                           showlegend=True),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_smooth, y=val_smooth, name='Validation (Smoothed)',
                           line=dict(color='pink', width=2 if full_size else 1, dash='dash'),
                           showlegend=True),
                row=1, col=1
            )

    # 2. Loss Components Breakdown (Stacked Area)
    if len(metrics['det_loss']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['det_loss'], name='Detection',
                       stackgroup='one', fillcolor='rgba(44, 160, 44, 0.5)'),
            row=1, col=2
        )
    if len(metrics['cls_loss']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['cls_loss'], name='Classification',
                       stackgroup='one', fillcolor='rgba(214, 39, 40, 0.5)'),
            row=1, col=2
        )
    if len(metrics['box_loss']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['box_loss'], name='Box',
                       stackgroup='one', fillcolor='rgba(148, 103, 189, 0.5)'),
            row=1, col=2
        )

    # 3. Training vs Validation Gap
    if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
        gaps = [t - v for t, v in zip(metrics['loss'], metrics['val_loss'])]
        fig.add_trace(
            go.Scatter(x=epochs, y=gaps, name='Train-Val Gap',
                       fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.3)',
                       line=dict(color='orange', width=3 if full_size else 2)),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # 4. Learning Dynamics (dual axis)
    if len(metrics['learning_rate']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['learning_rate'], name='Learning Rate',
                       line=dict(color='cyan', width=3 if full_size else 2)),
            row=2, col=2, secondary_y=False
        )
    if len(metrics['gradient_norm']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['gradient_norm'], name='Gradient Norm',
                       line=dict(color='gray', width=3 if full_size else 2)),
            row=2, col=2, secondary_y=True
        )

    # 5. Loss Ratios and Trends
    if len(metrics['loss']) > 1:
        loss_change_rate = [0] + [(metrics['loss'][i] - metrics['loss'][i - 1]) / metrics['loss'][i - 1] * 100
                                  if metrics['loss'][i - 1] != 0 else 0
                                  for i in range(1, len(metrics['loss']))]
        val_loss_change_rate = [0] + [
            (metrics['val_loss'][i] - metrics['val_loss'][i - 1]) / metrics['val_loss'][i - 1] * 100
            if metrics['val_loss'][i - 1] != 0 else 0
            for i in range(1, len(metrics['val_loss']))]

        fig.add_trace(
            go.Bar(x=epochs, y=loss_change_rate, name='Train Loss Change %',
                   marker_color='blue', opacity=0.6),
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=epochs, y=val_loss_change_rate, name='Val Loss Change %',
                   marker_color='red', opacity=0.6),
            row=3, col=1
        )

    # 6. Convergence Analysis (Log scale)
    if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
        # Filter out zero or negative values for log scale
        positive_loss = [l if l > 0 else None for l in metrics['loss']]
        positive_val_loss = [l if l > 0 else None for l in metrics['val_loss']]

        fig.add_trace(
            go.Scatter(x=epochs, y=positive_loss, name='Training (log)',
                       line=dict(color='blue', width=3 if full_size else 2),
                       connectgaps=False),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=positive_val_loss, name='Validation (log)',
                       line=dict(color='red', width=3 if full_size else 2),
                       connectgaps=False),
            row=3, col=2
        )

    # Update axes with larger fonts for full size
    axis_font_size = 14 if full_size else 11

    fig.update_xaxes(title_text="Epoch", row=3, col=1, title_font=dict(size=axis_font_size))
    fig.update_xaxes(title_text="Epoch", row=3, col=2, title_font=dict(size=axis_font_size))
    fig.update_yaxes(title_text="Loss", row=1, col=1, title_font=dict(size=axis_font_size))
    fig.update_yaxes(title_text="Loss", row=1, col=2, title_font=dict(size=axis_font_size))
    fig.update_yaxes(title_text="Gap", row=2, col=1, title_font=dict(size=axis_font_size))
    fig.update_yaxes(title_text="Learning Rate", row=2, col=2, secondary_y=False, title_font=dict(size=axis_font_size))
    fig.update_yaxes(title_text="Gradient Norm", row=2, col=2, secondary_y=True, title_font=dict(size=axis_font_size))
    fig.update_yaxes(title_text="Change %", row=3, col=1, title_font=dict(size=axis_font_size))
    fig.update_yaxes(title_text="Loss (log scale)", type="log", row=3, col=2, title_font=dict(size=axis_font_size))

    # Update layout
    fig.update_layout(
        title_text="Comprehensive Training Analysis",
        title_font=dict(size=24 if full_size else 20),
        title_x=0.5,
        title_xanchor='center',
        height=height,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12 if full_size else 11)
    )

    return fig


def create_individual_plot_html(metrics, plot_type, plot_num, temp_dir):
    """Create HTML for individual plot with full-size version."""

    epochs = metrics['epoch']

    if plot_type == "overall_loss":
        fig = go.Figure()

        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            fig.add_trace(go.Scatter(x=epochs, y=metrics['loss'], name='Training Loss',
                                     line=dict(color='blue', width=3),
                                     mode='lines+markers', marker=dict(size=6)))
            fig.add_trace(go.Scatter(x=epochs, y=metrics['val_loss'], name='Validation Loss',
                                     line=dict(color='red', width=3),
                                     mode='lines+markers', marker=dict(size=6)))

            # Add smoothed lines if enough data
            if len(epochs) > 5:
                window = min(5, len(epochs) // 4)
                train_smooth = np.convolve(metrics['loss'], np.ones(window) / window, mode='valid')
                val_smooth = np.convolve(metrics['val_loss'], np.ones(window) / window, mode='valid')
                epochs_smooth = epochs[window // 2:-(window // 2) if window > 1 else None]

                fig.add_trace(go.Scatter(x=epochs_smooth, y=train_smooth, name='Training (Smoothed)',
                                         line=dict(color='lightblue', width=2, dash='dash')))
                fig.add_trace(go.Scatter(x=epochs_smooth, y=val_smooth, name='Validation (Smoothed)',
                                         line=dict(color='pink', width=2, dash='dash')))

        fig.update_layout(
            title="Overall Loss Comparison",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=800,
            hovermode='x unified',
            showlegend=True
        )

    elif plot_type == "components":
        fig = go.Figure()

        if len(metrics['det_loss']) > 0:
            fig.add_trace(go.Scatter(x=epochs, y=metrics['det_loss'], name='Detection',
                                     stackgroup='one', fillcolor='rgba(44, 160, 44, 0.5)'))
        if len(metrics['cls_loss']) > 0:
            fig.add_trace(go.Scatter(x=epochs, y=metrics['cls_loss'], name='Classification',
                                     stackgroup='one', fillcolor='rgba(214, 39, 40, 0.5)'))
        if len(metrics['box_loss']) > 0:
            fig.add_trace(go.Scatter(x=epochs, y=metrics['box_loss'], name='Box',
                                     stackgroup='one', fillcolor='rgba(148, 103, 189, 0.5)'))

        fig.update_layout(
            title="Loss Components Breakdown",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=800,
            hovermode='x unified'
        )

    elif plot_type == "gap":
        fig = go.Figure()

        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            gaps = [t - v for t, v in zip(metrics['loss'], metrics['val_loss'])]

            fig.add_trace(go.Scatter(x=epochs, y=gaps, name='Train-Val Gap',
                                     fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.3)',
                                     line=dict(color='orange', width=3)))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")

        fig.update_layout(
            title="Training vs Validation Gap",
            xaxis_title="Epoch",
            yaxis_title="Gap",
            height=800,
            hovermode='x unified'
        )

    elif plot_type == "learning":
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        if len(metrics['learning_rate']) > 0:
            fig.add_trace(go.Scatter(x=epochs, y=metrics['learning_rate'], name='Learning Rate',
                                     line=dict(color='cyan', width=3)), secondary_y=False)
        if len(metrics['gradient_norm']) > 0:
            fig.add_trace(go.Scatter(x=epochs, y=metrics['gradient_norm'], name='Gradient Norm',
                                     line=dict(color='gray', width=3)), secondary_y=True)

        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="Learning Rate", secondary_y=False)
        fig.update_yaxes(title_text="Gradient Norm", secondary_y=True)
        fig.update_layout(title="Learning Dynamics", height=800, hovermode='x unified')

    elif plot_type == "ratios":
        fig = go.Figure()
        if len(metrics['loss']) > 1:
            loss_change_rate = [0] + [(metrics['loss'][i] - metrics['loss'][i - 1]) / metrics['loss'][i - 1] * 100
                                      if metrics['loss'][i - 1] != 0 else 0
                                      for i in range(1, len(metrics['loss']))]
            val_loss_change_rate = [0] + [
                (metrics['val_loss'][i] - metrics['val_loss'][i - 1]) / metrics['val_loss'][i - 1] * 100
                if metrics['val_loss'][i - 1] != 0 else 0
                for i in range(1, len(metrics['val_loss']))]

            fig.add_trace(go.Bar(x=epochs, y=loss_change_rate, name='Train Loss Change %',
                                 marker_color='blue', opacity=0.6))
            fig.add_trace(go.Bar(x=epochs, y=val_loss_change_rate, name='Val Loss Change %',
                                 marker_color='red', opacity=0.6))

        fig.update_layout(
            title="Loss Change Ratios",
            xaxis_title="Epoch",
            yaxis_title="Change %",
            height=800,
            hovermode='x unified'
        )

    elif plot_type == "convergence":
        fig = go.Figure()

        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            # Filter out zero or negative values for log scale
            positive_loss = [l if l > 0 else None for l in metrics['loss']]
            positive_val_loss = [l if l > 0 else None for l in metrics['val_loss']]

            fig.add_trace(go.Scatter(x=epochs, y=positive_loss, name='Training (log)',
                                     line=dict(color='blue', width=3),
                                     connectgaps=False))
            fig.add_trace(go.Scatter(x=epochs, y=positive_val_loss, name='Validation (log)',
                                     line=dict(color='red', width=3),
                                     connectgaps=False))

        fig.update_yaxes(type="log")
        fig.update_layout(
            title="Convergence Analysis (Log Scale)",
            xaxis_title="Epoch",
            yaxis_title="Loss (log scale)",
            height=800,
            hovermode='x unified'
        )
    else:
        return None

    # Save the individual plot
    filename = f"plot_{plot_num}_{plot_type}.html"
    filepath = os.path.join(temp_dir, filename)
    fig.write_html(filepath, auto_open=False)

    return filename


def create_dashboard_with_enlargement(metrics, eval_metrics, model_specs, temp_dir):
    """Create a dashboard HTML with enlargement functionality."""

    # Create evaluation metrics plot
    eval_fig = create_evaluation_metrics_plot(eval_metrics, full_size=False)
    eval_fig_full = create_evaluation_metrics_plot(eval_metrics, full_size=True)

    # Save full-size evaluation plot
    eval_file_full = os.path.join(temp_dir, "evaluation_metrics_full.html")
    eval_fig_full.write_html(eval_file_full, auto_open=False)

    # Create full-size versions
    fig1_full = create_interactive_plot(metrics, full_size=True)
    fig2_full = create_detailed_subplots(metrics, full_size=True)

    # Create regular-size versions
    fig1 = create_interactive_plot(metrics, full_size=False)
    fig2 = create_detailed_subplots(metrics, full_size=False)

    # Save full-size versions
    file1_full = os.path.join(temp_dir, "training_interactive_full.html")
    fig1_full.write_html(file1_full, auto_open=False)

    file2_full = os.path.join(temp_dir, "training_detailed_full.html")
    fig2_full.write_html(file2_full, auto_open=False)

    # Create individual plot files for detailed view
    individual_plots = []
    for plot_type, num in [("overall_loss", 1), ("components", 2), ("gap", 3),
                           ("learning", 4), ("ratios", 5), ("convergence", 6)]:
        filename = create_individual_plot_html(metrics, plot_type, num, temp_dir)
        if filename:
            individual_plots.append(filename)
        else:
            individual_plots.append("")

    # Convert figures to HTML strings with CDN
    fig1_html = fig1.to_html(include_plotlyjs='cdn', div_id="plot1")
    fig2_html = fig2.to_html(include_plotlyjs='cdn', div_id="plot2")
    eval_html = eval_fig.to_html(include_plotlyjs='cdn', div_id="plot_eval")

    # Create model specs card
    specs_html = create_model_specs_card(model_specs)

    # Create the main dashboard HTML
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Training Visualization Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}

            .dashboard-header {{
                text-align: center;
                color: white;
                margin-bottom: 30px;
            }}

            .dashboard-header h1 {{
                font-size: 2.5em;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }}

            .dashboard-header p {{
                font-size: 1.2em;
                margin-top: 10px;
                opacity: 0.9;
            }}

            .plot-container {{
                background: white;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                position: relative;
                transition: all 0.3s ease;
            }}

            .plot-container:hover {{
                box-shadow: 0 15px 40px rgba(0,0,0,0.3);
                transform: translateY(-2px);
            }}

            .enlarge-button {{
                position: absolute;
                top: 10px;
                right: 10px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                font-size: 20px;
                cursor: pointer;
                opacity: 0.7;
                transition: all 0.3s ease;
                z-index: 1000;
                display: flex;
                align-items: center;
                justify-content: center;
            }}

            .enlarge-button:hover {{
                opacity: 1;
                transform: scale(1.1);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }}

            .plot-title {{
                font-size: 1.5em;
                font-weight: bold;
                color: #333;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #667eea;
            }}

            .subplot-grid {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-top: 20px;
            }}

            .subplot-item {{
                background: #f8f9fa;
                border-radius: 10px;
                padding: 15px;
                position: relative;
                transition: all 0.3s ease;
            }}

            .subplot-item:hover {{
                background: #e9ecef;
                transform: scale(1.02);
            }}

            .subplot-enlarge {{
                position: absolute;
                top: 5px;
                right: 5px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
                cursor: pointer;
                opacity: 0.7;
                transition: opacity 0.3s ease;
            }}

            .subplot-enlarge:hover {{
                opacity: 1;
            }}

            .tooltip {{
                position: absolute;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 5px 10px;
                border-radius: 5px;
                font-size: 12px;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s ease;
                z-index: 1001;
            }}

            .enlarge-button:hover + .tooltip {{
                opacity: 1;
            }}

            .eval-summary {{
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 20px;
                margin-bottom: 20px;
            }}

            .eval-card {{
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }}

            .eval-card h3 {{
                margin-top: 0;
                color: #495057;
            }}

            .metric-value {{
                font-size: 2em;
                font-weight: bold;
                color: #667eea;
                margin: 10px 0;
            }}

            .metric-label {{
                color: #6c757d;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-header">
            <h1>📊 Training Visualization Dashboard</h1>
            <p>Interactive analysis of your model training metrics</p>
        </div>

        {specs_html}

        <div class="plot-container">
            <div class="plot-title">🎯 Model Evaluation Metrics</div>
            <div class="eval-summary">
                <div class="eval-card">
                    <h3>📋 Validation Performance</h3>
                    <div class="metric-value">{eval_metrics['validation'].get('AP', 0):.3f}</div>
                    <div class="metric-label">Average Precision (AP)</div>
                    <div style="margin-top: 10px;">
                        <strong>AP50:</strong> {eval_metrics['validation'].get('AP50', 0):.3f} | 
                        <strong>AP75:</strong> {eval_metrics['validation'].get('AP75', 0):.3f}
                    </div>
                </div>
                <div class="eval-card">
                    <h3>🧪 Test Performance</h3>
                    <div class="metric-value">{eval_metrics['test'].get('AP', 0):.3f}</div>
                    <div class="metric-label">Average Precision (AP)</div>
                    <div style="margin-top: 10px;">
                        <strong>AP50:</strong> {eval_metrics['test'].get('AP50', 0):.3f} | 
                        <strong>AP75:</strong> {eval_metrics['test'].get('AP75', 0):.3f}
                    </div>
                </div>
            </div>
            <button class="enlarge-button" onclick="openFullSize('evaluation_metrics_full.html')" title="Open in new window">⤢</button>
            <div class="tooltip">Click to enlarge in new window</div>
            {eval_html}
        </div>

        <div class="plot-container">
            <button class="enlarge-button" onclick="openFullSize('training_interactive_full.html')" title="Open in new window">⤢</button>
            <div class="tooltip">Click to enlarge in new window</div>
            <div class="plot-title">📈 Interactive Training Metrics (Grouped)</div>
            {fig1_html}
        </div>

        <div class="plot-container">
            <button class="enlarge-button" onclick="openFullSize('training_detailed_full.html')" title="Open in new window">⤢</button>
            <div class="tooltip">Click to enlarge in new window</div>
            <div class="plot-title">📊 Detailed Analysis Grid</div>
            {fig2_html}

            <div class="subplot-grid">
                <div class="subplot-item">
                    <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[0]}')">Enlarge</button>
                    <strong>1. Overall Loss</strong>
                </div>
                <div class="subplot-item">
                    <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[1]}')">Enlarge</button>
                    <strong>2. Components</strong>
                </div>
                <div class="subplot-item">
                    <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[2]}')">Enlarge</button>
                    <strong>3. Train-Val Gap</strong>
                </div>
                <div class="subplot-item">
                    <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[3]}')">Enlarge</button>
                    <strong>4. Learning Dynamics</strong>
                </div>
                <div class="subplot-item">
                    <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[4]}')">Enlarge</button>
                    <strong>5. Loss Ratios</strong>
                </div>
                <div class="subplot-item">
                    <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[5]}')">Enlarge</button>
                    <strong>6. Convergence</strong>
                </div>
            </div>
        </div>

        <script>
            function openFullSize(filename) {{
                window.open(filename, '_blank', 'width=1400,height=900,scrollbars=yes,resizable=yes');
            }}

            function openIndividualPlot(filename) {{
                if (filename) {{
                    window.open(filename, '_blank', 'width=1200,height=800,scrollbars=yes,resizable=yes');
                }}
            }}

            // Add hover effect for enlarge buttons
            document.querySelectorAll('.enlarge-button').forEach(button => {{
                button.addEventListener('mouseenter', function() {{
                    this.innerHTML = '🔍';
                }});
                button.addEventListener('mouseleave', function() {{
                    this.innerHTML = '⤢';
                }});
            }});
        </script>
    </body>
    </html>
    """

    return dashboard_html


def calculate_statistics(metrics, eval_metrics):
    """Calculate detailed statistics for the training."""
    stats = {}

    if len(metrics['loss']) > 0:
        # Basic statistics
        stats['total_epochs'] = len(metrics['epoch'])
        stats['final_train_loss'] = metrics['loss'][-1]
        stats['final_val_loss'] = metrics['val_loss'][-1]
        stats['best_val_loss'] = min(metrics['val_loss'])
        stats['best_epoch'] = metrics['epoch'][metrics['val_loss'].index(min(metrics['val_loss']))]

        # Convergence metrics
        stats['train_loss_reduction'] = (metrics['loss'][0] - metrics['loss'][-1]) / metrics['loss'][0] * 100
        stats['val_loss_reduction'] = (metrics['val_loss'][0] - metrics['val_loss'][-1]) / metrics['val_loss'][0] * 100

        # Overfitting metrics
        stats['final_overfit_gap'] = metrics['loss'][-1] - metrics['val_loss'][-1]
        stats['avg_overfit_gap'] = np.mean([t - v for t, v in zip(metrics['loss'], metrics['val_loss'])])

        # Stability metrics
        if len(metrics['loss']) > 5:
            last_5_train = metrics['loss'][-5:]
            last_5_val = metrics['val_loss'][-5:]
            stats['train_stability'] = np.std(last_5_train)
            stats['val_stability'] = np.std(last_5_val)

        # Learning rate stats
        if 'learning_rate' in metrics and len(metrics['learning_rate']) > 0:
            stats['initial_lr'] = metrics['learning_rate'][0]
            stats['final_lr'] = metrics['learning_rate'][-1]
            stats['lr_reduction'] = (stats['initial_lr'] - stats['final_lr']) / stats['initial_lr'] * 100

    # Add evaluation metrics
    if eval_metrics['validation']:
        stats['val_ap'] = eval_metrics['validation'].get('AP', 0)
        stats['val_ap50'] = eval_metrics['validation'].get('AP50', 0)
        stats['val_ap75'] = eval_metrics['validation'].get('AP75', 0)

    if eval_metrics['test']:
        stats['test_ap'] = eval_metrics['test'].get('AP', 0)
        stats['test_ap50'] = eval_metrics['test'].get('AP50', 0)
        stats['test_ap75'] = eval_metrics['test'].get('AP75', 0)

    return stats


def create_statistics_table(stats):
    """Create an enhanced HTML table with training statistics."""

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Training Statistics</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                padding: 40px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                margin: 0;
            }

            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }

            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }

            h1 {
                margin: 0;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }

            .subtitle {
                margin-top: 10px;
                opacity: 0.9;
                font-size: 1.1em;
            }

            table {
                width: 100%;
                border-collapse: collapse;
            }

            tr {
                transition: background-color 0.3s ease;
            }

            tr:hover {
                background-color: #f8f9fa;
            }

            td {
                padding: 20px 30px;
                border-bottom: 1px solid #e0e0e0;
            }

            .metric-name {
                font-weight: 600;
                color: #495057;
                font-size: 1.1em;
            }

            .metric-value {
                color: #212529;
                font-size: 1.1em;
                text-align: right;
                font-family: 'Courier New', monospace;
            }

            .good {
                color: #28a745;
                font-weight: bold;
            }

            .warning {
                color: #ffc107;
                font-weight: bold;
            }

            .bad {
                color: #dc3545;
                font-weight: bold;
            }

            .footer {
                padding: 20px;
                text-align: center;
                background: #f8f9fa;
                color: #6c757d;
                font-size: 0.9em;
            }

            .stat-icon {
                margin-right: 10px;
                font-size: 1.2em;
            }

            .section-header {
                background: #f8f9fa;
                font-weight: bold;
                color: #667eea;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 1px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📊 Training Statistics Summary</h1>
                <div class="subtitle">Comprehensive analysis of your model's training performance</div>
            </div>
            <table>
    """

    icons = {
        'total_epochs': '📈',
        'final_train_loss': '🎯',
        'final_val_loss': '🎯',
        'best_val_loss': '🏆',
        'best_epoch': '⭐',
        'train_loss_reduction': '📉',
        'val_loss_reduction': '📉',
        'final_overfit_gap': '⚖️',
        'avg_overfit_gap': '⚖️',
        'train_stability': '🔧',
        'val_stability': '🔧',
        'initial_lr': '🚀',
        'final_lr': '🏁',
        'lr_reduction': '🔻',
        'val_ap': '🎯',
        'val_ap50': '🎯',
        'val_ap75': '🎯',
        'test_ap': '🧪',
        'test_ap50': '🧪',
        'test_ap75': '🧪'
    }

    # Add section for training metrics
    html += """
        <tr>
            <td colspan="2" class="section-header metric-name">Training Metrics</td>
        </tr>
    """

    # Define sections
    training_metrics = ['total_epochs', 'final_train_loss', 'final_val_loss', 'best_val_loss',
                        'best_epoch', 'train_loss_reduction', 'val_loss_reduction',
                        'final_overfit_gap', 'avg_overfit_gap', 'train_stability',
                        'val_stability', 'initial_lr', 'final_lr', 'lr_reduction']

    eval_metrics = ['val_ap', 'val_ap50', 'val_ap75', 'test_ap', 'test_ap50', 'test_ap75']

    for key in training_metrics:
        if key in stats:
            value = stats[key]
            display_name = key.replace('_', ' ').title()
            icon = icons.get(key, '📌')

            if isinstance(value, float):
                formatted_value = f"{value:.4f}"

                # Add color coding for certain metrics
                if 'reduction' in key and value > 50:
                    formatted_value = f'<span class="good">{formatted_value}%</span>'
                elif 'overfit' in key and abs(value) > 0.1:
                    formatted_value = f'<span class="warning">{formatted_value}</span>'
                elif 'stability' in key and value > 0.1:
                    formatted_value = f'<span class="bad">{formatted_value}</span>'
            else:
                formatted_value = str(value)

            html += f"""
                <tr>
                    <td class="metric-name">
                        <span class="stat-icon">{icon}</span>
                        {display_name}
                    </td>
                    <td class="metric-value">{formatted_value}</td>
                </tr>
            """

    # Add evaluation metrics section
    html += """
        <tr>
            <td colspan="2" class="section-header metric-name">Evaluation Metrics</td>
        </tr>
    """

    for key in eval_metrics:
        if key in stats:
            value = stats[key]
            display_name = key.replace('_', ' ').title()
            icon = icons.get(key, '📌')

            formatted_value = f"{value:.3f}"

            # Add color coding for AP metrics
            if value > 0.8:
                formatted_value = f'<span class="good">{formatted_value}</span>'
            elif value > 0.6:
                formatted_value = f'{formatted_value}'
            else:
                formatted_value = f'<span class="warning">{formatted_value}</span>'

            html += f"""
                <tr>
                    <td class="metric-name">
                        <span class="stat-icon">{icon}</span>
                        {display_name}
                    </td>
                    <td class="metric-value">{formatted_value}</td>
                </tr>
            """

    html += """
            </table>
            <div class="footer">
                Generated with Training Visualizer | Use the dashboard for interactive exploration
            </div>
        </div>
    </body>
    </html>
    """

    return html


def create_comparison_evaluation_plot(datasets, full_size=False):
    """Create side-by-side evaluation metrics plots for multiple datasets."""

    num_datasets = len(datasets)
    if num_datasets == 1:
        # Single dataset - use original layout
        return create_evaluation_metrics_plot(datasets[0]['eval_metrics'], full_size)

    # Multiple datasets - create side-by-side with larger size
    fig = make_subplots(
        rows=1, cols=num_datasets,
        subplot_titles=[f"<b>{d['name']}</b>" for d in datasets],
        horizontal_spacing=0.12
    )

    categories = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                  'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']

    colors_val = ['rgb(31, 119, 180)', 'rgb(44, 160, 44)', 'rgb(148, 103, 189)']
    colors_test = ['rgb(255, 127, 14)', 'rgb(214, 39, 40)', 'rgb(227, 119, 194)']

    for idx, dataset in enumerate(datasets, 1):
        eval_metrics = dataset['eval_metrics']

        val_values = []
        test_values = []

        for cat in categories:
            val_values.append(eval_metrics['validation'].get(cat, 0))
            test_values.append(eval_metrics['test'].get(cat, 0))

        # Validation line with thicker lines
        fig.add_trace(
            go.Scatter(
                name=f'Validation',
                x=categories,
                y=val_values,
                mode='lines+markers',
                line=dict(color=colors_val[idx - 1], width=4 if full_size else 3),
                marker=dict(size=12 if full_size else 10, symbol='circle'),
                text=[f'Val: {v:.3f}' for v in val_values],
                textposition='top center',
                textfont=dict(size=12 if full_size else 11),
                legendgroup=f'dataset{idx}',
                showlegend=(idx == 1)
            ),
            row=1, col=idx
        )

        # Test line with thicker lines
        fig.add_trace(
            go.Scatter(
                name=f'Test',
                x=categories,
                y=test_values,
                mode='lines+markers',
                line=dict(color=colors_test[idx - 1], width=4 if full_size else 3, dash='dash'),
                marker=dict(size=12 if full_size else 10, symbol='diamond'),
                text=[f'Test: {v:.3f}' for v in test_values],
                textposition='bottom center',
                textfont=dict(size=12 if full_size else 11),
                legendgroup=f'dataset{idx}',
                showlegend=(idx == 1)
            ),
            row=1, col=idx
        )

        # Add highlighted background for key metrics
        key_metrics_idx = [categories.index(m) for m in ['AP', 'AP50', 'AP75'] if m in categories]
        if key_metrics_idx:
            fig.add_vrect(
                x0=-0.5, x1=max(key_metrics_idx) + 0.5,
                fillcolor="rgba(102, 126, 234, 0.05)",
                layer="below",
                line_width=0,
                row=1, col=idx
            )

    # Update layout with larger size
    fig.update_layout(
        title={
            'text': 'Model Evaluation Metrics Comparison',
            'font': {'size': 28 if full_size else 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=800 if full_size else 700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.3)',
        paper_bgcolor='white',
        font=dict(size=13)
    )

    # Update all axes with better formatting
    for i in range(1, num_datasets + 1):
        fig.update_xaxes(
            tickangle=-45,
            row=1, col=i,
            tickfont=dict(size=12),
            title_font=dict(size=14)
        )
        fig.update_yaxes(
            range=[0, 1.1],
            row=1, col=i,
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        if i == 1:
            fig.update_yaxes(title_text='Score', row=1, col=i)

    return fig


def create_comparison_interactive_plot(datasets, full_size=False):
    """Create side-by-side interactive training metrics plots."""

    num_datasets = len(datasets)

    # Create subplots with secondary y-axis
    fig = make_subplots(
        rows=1, cols=num_datasets,
        subplot_titles=[f"<b>{d['name']}</b>" for d in datasets],
        horizontal_spacing=0.12,
        specs=[[{"secondary_y": True} for _ in range(num_datasets)]]
    )

    # Color schemes for different datasets
    colors = {
        'loss': ['rgb(31, 119, 180)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)'],
        'val_loss': ['rgb(255, 127, 14)', 'rgb(152, 223, 138)', 'rgb(255, 152, 150)'],
        'det_loss': ['rgba(44, 160, 44, 0.7)', 'rgba(76, 190, 76, 0.7)', 'rgba(100, 220, 100, 0.7)'],
        'cls_loss': ['rgba(214, 39, 40, 0.7)', 'rgba(244, 69, 70, 0.7)', 'rgba(255, 99, 100, 0.7)'],
        'lr': ['rgb(23, 190, 207)', 'rgb(53, 220, 237)', 'rgb(83, 250, 255)'],
        'grad': ['rgb(127, 127, 127)', 'rgb(157, 157, 157)', 'rgb(187, 187, 187)']
    }

    for idx, dataset in enumerate(datasets, 1):
        metrics = dataset['metrics']
        epochs = metrics['epoch']

        # Main losses (always visible)
        if len(metrics['loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['loss'],
                    name='Training Loss',
                    line=dict(color=colors['loss'][idx - 1], width=3),
                    mode='lines+markers',
                    marker=dict(size=5),
                    legendgroup=f'dataset{idx}',
                    showlegend=(idx == 1)
                ),
                row=1, col=idx, secondary_y=False
            )

        if len(metrics['val_loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['val_loss'],
                    name='Validation Loss',
                    line=dict(color=colors['val_loss'][idx - 1], width=3),
                    mode='lines+markers',
                    marker=dict(size=5),
                    legendgroup=f'dataset{idx}',
                    showlegend=(idx == 1)
                ),
                row=1, col=idx, secondary_y=False
            )

            # Mark best validation loss
            best_idx = metrics['val_loss'].index(min(metrics['val_loss']))
            best_epoch = metrics['epoch'][best_idx]
            best_val_loss = min(metrics['val_loss'])

            fig.add_annotation(
                x=best_epoch,
                y=best_val_loss,
                text=f"Best: {best_val_loss:.4f}",
                showarrow=True,
                arrowhead=2,
                arrowcolor=colors['val_loss'][idx - 1],
                font=dict(size=11),
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor=colors['val_loss'][idx - 1],
                borderwidth=2,
                row=1, col=idx
            )

        # Component losses (legend only)
        if len(metrics['det_loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['det_loss'],
                    name='Detection Loss',
                    line=dict(color=colors['det_loss'][idx - 1], width=2, dash='dash'),
                    mode='lines',
                    visible='legendonly',
                    legendgroup=f'dataset{idx}',
                    showlegend=(idx == 1)
                ),
                row=1, col=idx, secondary_y=False
            )

        if len(metrics['cls_loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['cls_loss'],
                    name='Classification Loss',
                    line=dict(color=colors['cls_loss'][idx - 1], width=2, dash='dot'),
                    mode='lines',
                    visible='legendonly',
                    legendgroup=f'dataset{idx}',
                    showlegend=(idx == 1)
                ),
                row=1, col=idx, secondary_y=False
            )

        # Learning rate on secondary axis
        if len(metrics['learning_rate']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['learning_rate'],
                    name='Learning Rate',
                    line=dict(color=colors['lr'][idx - 1], width=2),
                    mode='lines',
                    visible='legendonly',
                    legendgroup=f'dataset{idx}',
                    showlegend=(idx == 1)
                ),
                row=1, col=idx, secondary_y=True
            )

    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Training Metrics Comparison',
            'font': {'size': 26},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=700,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=13)
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12)
    )

    # Update axes
    for i in range(1, num_datasets + 1):
        fig.update_xaxes(
            title_text="Epoch",
            row=1, col=i,
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        fig.update_yaxes(
            title_text="Loss" if i == 1 else "",
            row=1, col=i,
            secondary_y=False,
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor='rgba(128, 128, 128, 0.2)'
        )
        fig.update_yaxes(
            title_text="Learning Rate" if i == num_datasets else "",
            row=1, col=i,
            secondary_y=True,
            tickfont=dict(size=12),
            title_font=dict(size=14),
            gridcolor='rgba(128, 128, 128, 0.1)'
        )

    # Update subplot titles
    for annotation in fig['layout']['annotations']:
        if 'text' in annotation and any(d['name'] in annotation['text'] for d in datasets):
            annotation['font'] = dict(size=16)

    return fig


def create_comparison_training_plot(datasets, plot_type='loss', full_size=False):
    """Create side-by-side training plots for multiple datasets."""

    num_datasets = len(datasets)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=num_datasets,
        subplot_titles=[d['name'] for d in datasets],
        horizontal_spacing=0.1,
        specs=[[{"secondary_y": True} for _ in range(num_datasets)]]
    )

    colors = {
        'train': ['rgb(31, 119, 180)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)'],
        'val': ['rgb(255, 127, 14)', 'rgb(152, 223, 138)', 'rgb(255, 152, 150)']
    }

    for idx, dataset in enumerate(datasets, 1):
        metrics = dataset['metrics']
        epochs = metrics['epoch']

        if plot_type == 'loss':
            # Training and validation loss
            if len(metrics['loss']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=metrics['loss'],
                        name=f'Train ({dataset["name"]})',
                        line=dict(color=colors['train'][idx - 1], width=3 if full_size else 2),
                        mode='lines+markers',
                        marker=dict(size=4),
                        legendgroup=f'dataset{idx}'
                    ),
                    row=1, col=idx, secondary_y=False
                )

            if len(metrics['val_loss']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=metrics['val_loss'],
                        name=f'Val ({dataset["name"]})',
                        line=dict(color=colors['val'][idx - 1], width=3 if full_size else 2),
                        mode='lines+markers',
                        marker=dict(size=4),
                        legendgroup=f'dataset{idx}'
                    ),
                    row=1, col=idx, secondary_y=False
                )

                # Mark best validation loss
                best_idx = metrics['val_loss'].index(min(metrics['val_loss']))
                best_epoch = metrics['epoch'][best_idx]
                best_val_loss = min(metrics['val_loss'])

                fig.add_annotation(
                    x=best_epoch,
                    y=best_val_loss,
                    text=f"Best: {best_val_loss:.4f}",
                    showarrow=True,
                    arrowhead=2,
                    row=1, col=idx
                )

        elif plot_type == 'learning':
            # Learning rate and gradient norm
            if len(metrics['learning_rate']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=metrics['learning_rate'],
                        name=f'LR ({dataset["name"]})',
                        line=dict(color='cyan', width=2),
                        legendgroup=f'dataset{idx}'
                    ),
                    row=1, col=idx, secondary_y=False
                )

            if len(metrics['gradient_norm']) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=metrics['gradient_norm'],
                        name=f'Grad ({dataset["name"]})',
                        line=dict(color='gray', width=2),
                        legendgroup=f'dataset{idx}'
                    ),
                    row=1, col=idx, secondary_y=True
                )

    # Update layout
    title = 'Loss Comparison' if plot_type == 'loss' else 'Learning Dynamics Comparison'
    fig.update_layout(
        title={
            'text': title,
            'font': {'size': 24 if full_size else 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=600 if full_size else 500,
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(240, 240, 240, 0.3)',
        paper_bgcolor='white'
    )

    # Update axes
    for i in range(1, num_datasets + 1):
        fig.update_xaxes(title_text="Epoch", row=1, col=i)
        if plot_type == 'loss':
            fig.update_yaxes(title_text="Loss" if i == 1 else "", row=1, col=i, secondary_y=False)
        else:
            fig.update_yaxes(title_text="Learning Rate" if i == 1 else "", row=1, col=i, secondary_y=False)
            fig.update_yaxes(title_text="Gradient Norm" if i == num_datasets else "", row=1, col=i, secondary_y=True)

    return fig


def create_comparison_dashboard(datasets, temp_dir):
    """Create a comparison dashboard for multiple datasets with all visualizations."""

    # Create all comparison plots
    eval_fig = create_comparison_evaluation_plot(datasets, full_size=False)
    detailed_fig = create_comparison_detailed_subplots(datasets, full_size=False)
    convergence_fig = create_comparison_convergence_plot(datasets, full_size=False)
    learning_fig = create_comparison_learning_dynamics_plot(datasets, full_size=False)

    # Convert to HTML
    eval_html = eval_fig.to_html(include_plotlyjs='cdn', div_id="eval_compare")
    detailed_html = detailed_fig.to_html(include_plotlyjs='cdn', div_id="detailed_compare")
    convergence_html = convergence_fig.to_html(include_plotlyjs='cdn', div_id="convergence_compare")
    learning_html = learning_fig.to_html(include_plotlyjs='cdn', div_id="learning_compare")

    # Create comparison statistics table with hyperparameters
    stats_html = create_comparison_statistics_table(datasets)

    # Create summary cards for key metrics
    summary_cards_html = """
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
    """

    for idx, dataset in enumerate(datasets):
        color = ['#1f77b4', '#2ca02c', '#d62728'][idx % 3]
        val_ap = dataset['stats'].get('val_ap', 0)
        test_ap = dataset['stats'].get('test_ap', 0)
        best_loss = dataset['stats'].get('best_val_loss', 0)
        best_epoch = dataset['stats'].get('best_epoch', 0)

        summary_cards_html += f"""
        <div style="background: white; border-radius: 10px; padding: 25px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-left: 5px solid {color};">
            <h3 style="margin: 0 0 20px 0; color: #333; font-size: 1.4em;">{dataset['name']}</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px;">
                    <div style="color: #6c757d; font-size: 0.9em; margin-bottom: 5px;">Validation AP</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: {color};">{val_ap:.3f}</div>
                </div>
                <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px;">
                    <div style="color: #6c757d; font-size: 0.9em; margin-bottom: 5px;">Test AP</div>
                    <div style="font-size: 1.8em; font-weight: bold; color: {color};">{test_ap:.3f}</div>
                </div>
            </div>
            <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 5px;">
                    <div style="color: #6c757d; font-size: 0.85em;">Best Val Loss</div>
                    <div style="font-size: 1.3em; font-weight: 600; color: #333;">{best_loss:.4f}</div>
                </div>
                <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 5px;">
                    <div style="color: #6c757d; font-size: 0.85em;">Best Epoch</div>
                    <div style="font-size: 1.3em; font-weight: 600; color: #333;">{best_epoch}</div>
                </div>
            </div>
        </div>
        """

    summary_cards_html += "</div>"

    # Build dashboard HTML
    dashboard_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Training Comparison Dashboard</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}

            .dashboard-header {{
                text-align: center;
                color: white;
                margin-bottom: 40px;
            }}

            .dashboard-header h1 {{
                font-size: 3em;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                font-weight: 600;
            }}

            .dashboard-header p {{
                font-size: 1.2em;
                margin-top: 10px;
                opacity: 0.95;
            }}

            .plot-container {{
                background: white;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                transition: transform 0.3s ease;
            }}

            .plot-container:hover {{
                transform: translateY(-2px);
                box-shadow: 0 12px 35px rgba(0,0,0,0.25);
            }}

            .plot-title {{
                font-size: 1.8em;
                font-weight: 600;
                color: #333;
                margin-bottom: 20px;
                padding-bottom: 12px;
                border-bottom: 3px solid #667eea;
                display: flex;
                align-items: center;
            }}

            .plot-description {{
                font-size: 0.95em;
                color: #6c757d;
                margin-bottom: 20px;
                line-height: 1.5;
            }}

            .section-divider {{
                height: 2px;
                background: linear-gradient(90deg, transparent, #667eea, transparent);
                margin: 40px 0;
            }}
        </style>
    </head>
    <body>
        <div class="dashboard-header">
            <h1>📊 Model Training Comparison Dashboard</h1>
            <p>Comprehensive side-by-side analysis of {len(datasets)} training runs</p>
        </div>

        {summary_cards_html}

        {stats_html}

        <div class="section-divider"></div>

        <div class="plot-container">
            <div class="plot-title">🎯 Evaluation Metrics Comparison</div>
            <div class="plot-description">
                Compare validation and test performance across all metrics. Solid lines show validation, dashed lines show test results.
            </div>
            {eval_html}
        </div>

        <div class="plot-container">
            <div class="plot-title">📈 Detailed Training Analysis</div>
            <div class="plot-description">
                Top row: Training vs validation loss curves with best epoch marked.<br>
                Middle row: Loss component breakdown showing contribution of each loss type.<br>
                Bottom row: Training-validation gap analysis (smaller is better).
            </div>
            {detailed_html}
        </div>

        <div class="plot-container">
            <div class="plot-title">📉 Convergence & Stability Analysis</div>
            <div class="plot-description">
                Top row: Loss convergence on logarithmic scale for better visualization of later epochs.<br>
                Bottom row: Epoch-to-epoch loss change rate showing training stability.
            </div>
            {convergence_html}
        </div>

        <div class="plot-container">
            <div class="plot-title">⚡ Learning Dynamics</div>
            <div class="plot-description">
                Top row: Learning rate schedule with marked LR drops.<br>
                Bottom row: Gradient norm evolution showing optimization dynamics (smoothed trend in red).
            </div>
            {learning_html}
        </div>

        <div style="text-align: center; color: white; padding: 30px; opacity: 0.9;">
            <p style="font-size: 1.1em;">💡 Tip: Hover over plots for detailed values | Plots are interactive - zoom, pan, and explore!</p>
        </div>
    </body>
    </html>
    """

    return dashboard_html


def create_comparison_statistics_table(datasets):
    """Create a comparison table for multiple datasets with hyperparameters."""

    html = """
    <div style="background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h2 style="color: #333; margin-bottom: 20px; border-bottom: 2px solid #667eea; 
                   padding-bottom: 10px; font-size: 1.8em;">📊 Performance Comparison</h2>

        <!-- Performance Metrics Table -->
        <h3 style="color: #495057; margin-top: 20px; margin-bottom: 15px; font-size: 1.4em;">📈 Training & Evaluation Metrics</h3>
        <table style="width: 100%; border-collapse: collapse; font-size: 1.1em;">
            <thead>
                <tr style="background: #f8f9fa;">
                    <th style="padding: 15px; text-align: left; border-bottom: 2px solid #667eea; font-size: 1.2em;">Metric</th>
    """

    # Add column headers for each dataset
    for dataset in datasets:
        html += f'<th style="padding: 15px; text-align: center; border-bottom: 2px solid #667eea; font-size: 1.2em;">{dataset["name"]}</th>'

    html += """
            </tr>
        </thead>
        <tbody>
    """

    # Define metrics to compare
    comparison_metrics = [
        ('Total Epochs', 'total_epochs', '{:.0f}'),
        ('Best Val Loss', 'best_val_loss', '{:.4f}'),
        ('Final Train Loss', 'final_train_loss', '{:.4f}'),
        ('Final Val Loss', 'final_val_loss', '{:.4f}'),
        ('Train Loss Reduction %', 'train_loss_reduction', '{:.2f}'),
        ('Val Loss Reduction %', 'val_loss_reduction', '{:.2f}'),
        ('Val AP', 'val_ap', '{:.3f}'),
        ('Val AP50', 'val_ap50', '{:.3f}'),
        ('Val AP75', 'val_ap75', '{:.3f}'),
        ('Test AP', 'test_ap', '{:.3f}'),
        ('Test AP50', 'test_ap50', '{:.3f}'),
        ('Test AP75', 'test_ap75', '{:.3f}'),
    ]

    for metric_name, metric_key, format_str in comparison_metrics:
        html += f"""
            <tr style="border-bottom: 1px solid #e0e0e0;">
                <td style="padding: 14px; font-weight: 600; color: #495057; font-size: 1.05em;">{metric_name}</td>
        """

        values = []
        for dataset in datasets:
            if metric_key in dataset['stats']:
                value = dataset['stats'][metric_key]
                values.append(value)
                formatted_value = format_str.format(value)
            else:
                values.append(None)
                formatted_value = 'N/A'

            # Highlight best value
            style = "padding: 14px; text-align: center; font-family: monospace; font-size: 1.1em;"
            if values and value is not None and len([v for v in values if v is not None]) > 1:
                if 'loss' in metric_key.lower() and 'reduction' not in metric_key:
                    # For loss, lower is better
                    if value == min(v for v in values if v is not None):
                        style += " color: #28a745; font-weight: bold; background-color: rgba(40, 167, 69, 0.1);"
                else:
                    # For AP metrics and reduction percentages, higher is better
                    if value == max(v for v in values if v is not None):
                        style += " color: #28a745; font-weight: bold; background-color: rgba(40, 167, 69, 0.1);"

            html += f'<td style="{style}">{formatted_value}</td>'

        html += '</tr>'

    html += """
        </tbody>
    </table>
    """

    # Add Hyperparameters Comparison Table
    html += """
        <h3 style="color: #495057; margin-top: 30px; margin-bottom: 15px; font-size: 1.4em;">⚙️ Hyperparameters Comparison</h3>
        <div style="max-height: 600px; overflow-y: auto; border: 1px solid #dee2e6; border-radius: 8px;">
        <table style="width: 100%; border-collapse: collapse; font-size: 1.05em;">
            <thead style="position: sticky; top: 0; background: white; z-index: 10;">
                <tr style="background: #f8f9fa;">
                    <th style="padding: 12px; text-align: left; border-bottom: 2px solid #667eea; font-size: 1.1em;">Parameter</th>
    """

    # Add column headers for hyperparameters
    for dataset in datasets:
        html += f'<th style="padding: 12px; text-align: center; border-bottom: 2px solid #667eea; font-size: 1.1em;">{dataset["name"]}</th>'

    html += """
            </tr>
        </thead>
        <tbody>
    """

    # Collect all unique hyperparameters from all datasets
    all_hyperparams = set()
    for dataset in datasets:
        all_hyperparams.update(dataset['specs']['hyperparameters'].keys())

    # Group hyperparameters by category
    hyperparam_groups = [
        ('Model Architecture', ['backbone_name', 'num_classes', 'image_size']),
        ('NMS Configuration', ['nms_method', 'iou_thresh', 'score_thresh', 'sigma',
                               'pyfunc', 'max_nms_inputs', 'max_output_size']),
        ('Loss Parameters', ['alpha', 'gamma', 'delta', 'box_loss_weight', 'label_smoothing']),
        ('Learning Rate', ['learning_rate', 'lr_warmup_init', 'lr_warmup_epoch',
                           'lr_decay_method', 'first_lr_drop_epoch', 'second_lr_drop_epoch',
                           'poly_lr_power']),
        ('Optimization', ['momentum', 'optimizer', 'weight_decay', 'clip_gradients_norm',
                          'moving_average_decay']),
        ('Augmentation', ['input_rand_hflip', 'jitter_min', 'jitter_max',
                          'autoaugment_policy', 'grid_mask']),
        ('Anchors/Detection', ['anchor_scale', 'aspect_ratios', 'num_scales']),
        ('Training Settings', ['skip_crowd_during_training', 'max_instances_per_image',
                               'mixed_precision', 'drop_remainder', 'map_freq'])
    ]

    for group_name, params in hyperparam_groups:
        # Only show group if at least one param exists
        group_params = [p for p in params if p in all_hyperparams]
        if not group_params:
            continue

        # Add group header
        html += f"""
            <tr style="background: #e9ecef;">
                <td colspan="{len(datasets) + 1}" style="padding: 10px; font-weight: bold; 
                    color: #667eea; font-size: 1.05em; border-top: 1px solid #dee2e6;">
                    {group_name}
                </td>
            </tr>
        """

        for param in group_params:
            param_label = param.replace('_', ' ').title()
            html += f"""
                <tr style="border-bottom: 1px solid #e0e0e0;">
                    <td style="padding: 10px 12px; color: #6c757d; font-size: 0.95em;">
                        {param_label}
                    </td>
            """

            values = []
            for dataset in datasets:
                value = dataset['specs']['hyperparameters'].get(param, 'N/A')
                values.append(value)

                # Check if values differ
                style = "padding: 10px 12px; text-align: center; font-family: monospace; font-size: 1.05em;"
                if len(set(values)) > 1 and value != 'N/A':
                    # Highlight differences
                    style += " background-color: rgba(255, 193, 7, 0.15); font-weight: 600;"

                # Special formatting for certain values
                if value in ['True', 'False']:
                    color = '#28a745' if value == 'True' else '#dc3545'
                    style += f" color: {color}; font-weight: bold;"

                html += f'<td style="{style}">{value}</td>'

            html += '</tr>'

    # Add any remaining parameters not in groups
    remaining_params = all_hyperparams - set(sum([params for _, params in hyperparam_groups], []))
    if remaining_params:
        html += f"""
            <tr style="background: #e9ecef;">
                <td colspan="{len(datasets) + 1}" style="padding: 10px; font-weight: bold; 
                    color: #667eea; font-size: 1.05em; border-top: 1px solid #dee2e6;">
                    Other Parameters
                </td>
            </tr>
        """

        for param in sorted(remaining_params):
            param_label = param.replace('_', ' ').title()
            html += f"""
                <tr style="border-bottom: 1px solid #e0e0e0;">
                    <td style="padding: 10px 12px; color: #6c757d; font-size: 0.95em;">
                        {param_label}
                    </td>
            """

            values = []
            for dataset in datasets:
                value = dataset['specs']['hyperparameters'].get(param, 'N/A')
                values.append(value)

                style = "padding: 10px 12px; text-align: center; font-family: monospace; font-size: 1.05em;"
                if len(set(values)) > 1 and value != 'N/A':
                    style += " background-color: rgba(255, 193, 7, 0.15); font-weight: 600;"

                html += f'<td style="{style}">{value}</td>'

            html += '</tr>'

    html += """
        </tbody>
    </table>
    </div>

    <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
        <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
            <strong>Legend:</strong> 
            <span style="background-color: rgba(40, 167, 69, 0.2); padding: 2px 6px; border-radius: 3px;">Best value</span>
            <span style="background-color: rgba(255, 193, 7, 0.2); padding: 2px 6px; border-radius: 3px; margin-left: 10px;">Differs between models</span>
        </p>
    </div>
    </div>
    """

    return html


class DatasetSelector:
    """GUI for selecting multiple datasets to compare."""

    def __init__(self):
        self.datasets = []
        self.root = None

    def select_datasets(self):
        """Open GUI for dataset selection."""
        self.root = tk.Tk()
        self.root.title("Select Training Logs to Compare")
        self.root.geometry("600x400")

        # Header
        header = tk.Label(
            self.root,
            text="Select Training Logs for Comparison",
            font=("Arial", 16, "bold")
        )
        header.pack(pady=10)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="Add up to 3 training log files to compare side-by-side",
            font=("Arial", 10)
        )
        instructions.pack()

        # Dataset list frame
        list_frame = tk.Frame(self.root)
        list_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Listbox for datasets
        self.listbox = tk.Listbox(list_frame, height=10)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Add dataset button
        add_btn = tk.Button(
            button_frame,
            text="Add Dataset",
            command=self.add_dataset,
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=5
        )
        add_btn.pack(side=tk.LEFT, padx=5)

        # Remove dataset button
        remove_btn = tk.Button(
            button_frame,
            text="Remove Selected",
            command=self.remove_dataset,
            bg="#f44336",
            fg="white",
            padx=20,
            pady=5
        )
        remove_btn.pack(side=tk.LEFT, padx=5)

        # Clear all button
        clear_btn = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_datasets,
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=5
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Visualize button
        self.visualize_btn = tk.Button(
            button_frame,
            text="Visualize",
            command=self.visualize,
            bg="#2196F3",
            fg="white",
            padx=30,
            pady=5,
            state=tk.DISABLED
        )
        self.visualize_btn.pack(side=tk.LEFT, padx=20)

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="No datasets selected",
            font=("Arial", 10),
            fg="gray"
        )
        self.status_label.pack(pady=5)

        self.root.mainloop()

    def add_dataset(self):
        """Add a dataset via file dialog."""
        if len(self.datasets) >= 3:
            messagebox.showwarning("Limit Reached", "Maximum of 3 datasets can be compared")
            return

        filepath = filedialog.askopenfilename(
            title="Select Training Log File",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filepath:
            # Get a custom name for the dataset
            name = simpledialog.askstring(
                "Dataset Name",
                f"Enter a name for this dataset:",
                initialvalue=Path(filepath).stem
            )

            if name:
                self.datasets.append({'path': filepath, 'name': name})
                self.listbox.insert(tk.END, f"{name} - {Path(filepath).name}")
                self.update_status()

    def remove_dataset(self):
        """Remove selected dataset."""
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            self.listbox.delete(idx)
            del self.datasets[idx]
            self.update_status()

    def clear_datasets(self):
        """Clear all datasets."""
        self.listbox.delete(0, tk.END)
        self.datasets = []
        self.update_status()

    def update_status(self):
        """Update status label and button states."""
        count = len(self.datasets)
        if count == 0:
            self.status_label.config(text="No datasets selected", fg="gray")
            self.visualize_btn.config(state=tk.DISABLED)
        else:
            self.status_label.config(text=f"{count} dataset(s) selected", fg="green")
            self.visualize_btn.config(state=tk.NORMAL)

    def visualize(self):
        """Start visualization."""
        self.root.quit()
        self.root.destroy()


def visualize_training_comparison(dataset_configs):
    """Visualize multiple training logs for comparison."""

    datasets = []

    for config in dataset_configs:
        try:
            # Parse metrics and specs for each dataset
            metrics, eval_metrics = parse_training_log(config['path'])

            # Try to find spec file
            log_dir = Path(config['path']).parent
            spec_filepath = log_dir / "training_specs.txt"
            model_specs = parse_model_specs(spec_filepath) if spec_filepath.exists() else {'main': {},
                                                                                           'hyperparameters': {}}

            # Calculate statistics
            stats = calculate_statistics(metrics, eval_metrics)

            datasets.append({
                'name': config['name'],
                'metrics': metrics,
                'eval_metrics': eval_metrics,
                'specs': model_specs,
                'stats': stats
            })

            print(f"✅ Loaded dataset: {config['name']}")

        except Exception as e:
            print(f"⚠️ Error loading {config['name']}: {e}")

    if not datasets:
        print("❌ No valid datasets loaded")
        return False

    print(f"\n🔄 Generating comparison visualizations for {len(datasets)} dataset(s)...")

    # Create temp directory
    temp_dir = tempfile.gettempdir()

    if len(datasets) == 1:
        # Single dataset - use original visualization
        dashboard_html = create_dashboard_with_enlargement(
            datasets[0]['metrics'],
            datasets[0]['eval_metrics'],
            datasets[0]['specs'],
            temp_dir
        )
    else:
        # Multiple datasets - use comparison view
        dashboard_html = create_comparison_dashboard(datasets, temp_dir)

    # Save and open dashboard
    dashboard_file = os.path.join(temp_dir, "comparison_dashboard.html")
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(dashboard_html)

    print("\n📊 Opening comparison dashboard in browser...")
    webbrowser.open(f"file://{dashboard_file}", new=2)

    print(f"\n✅ Comparison dashboard opened: {dashboard_file}")

    return True

def create_comparison_convergence_plot(datasets, full_size=False):
    """Create convergence analysis plots for comparison."""

    num_datasets = len(datasets)

    # Create subplot titles in correct row-major order
    row1_titles = [f"<b>{d['name']}</b><br>Convergence (Log Scale)" for d in datasets]
    row2_titles = [f"<b>{d['name']}</b><br>Loss Change Rate %" for d in datasets]
    subplot_titles = row1_titles + row2_titles

    fig = make_subplots(
        rows=2, cols=num_datasets,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False} for _ in range(num_datasets)],
               [{"secondary_y": False} for _ in range(num_datasets)]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    colors_set = [
        {'train': 'rgb(31, 119, 180)', 'val': 'rgb(255, 127, 14)'},
        {'train': 'rgb(44, 160, 44)', 'val': 'rgb(214, 39, 40)'},
        {'train': 'rgb(148, 103, 189)', 'val': 'rgb(227, 119, 194)'}
    ]

    for idx, dataset in enumerate(datasets):
        metrics = dataset['metrics']
        epochs = metrics['epoch']
        colors = colors_set[idx % 3]
        col = idx + 1

        # Row 1: Convergence Analysis (Log scale)
        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            # Filter out zero or negative values for log scale
            positive_loss = [l if l > 0 else None for l in metrics['loss']]
            positive_val_loss = [l if l > 0 else None for l in metrics['val_loss']]

            fig.add_trace(
                go.Scatter(
                    x=epochs, y=positive_loss,
                    name='Training (log)',
                    line=dict(color=colors['train'], width=3),
                    connectgaps=False,
                    legendgroup=f'log{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=positive_val_loss,
                    name='Validation (log)',
                    line=dict(color=colors['val'], width=3),
                    connectgaps=False,
                    legendgroup=f'log{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

        # Row 2: Loss Change Rate
        if len(metrics['loss']) > 1:
            loss_change_rate = [0] + [
                (metrics['loss'][i] - metrics['loss'][i - 1]) / metrics['loss'][i - 1] * 100
                if metrics['loss'][i - 1] != 0 else 0
                for i in range(1, len(metrics['loss']))
            ]
            val_loss_change_rate = [0] + [
                (metrics['val_loss'][i] - metrics['val_loss'][i - 1]) / metrics['val_loss'][i - 1] * 100
                if metrics['val_loss'][i - 1] != 0 else 0
                for i in range(1, len(metrics['val_loss']))
            ]

            fig.add_trace(
                go.Bar(
                    x=epochs, y=loss_change_rate,
                    name='Train Loss Change %',
                    marker_color=colors['train'],
                    opacity=0.6,
                    legendgroup=f'change{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )
            fig.add_trace(
                go.Bar(
                    x=epochs, y=val_loss_change_rate,
                    name='Val Loss Change %',
                    marker_color=colors['val'],
                    opacity=0.6,
                    legendgroup=f'change{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=col)

    # Update axes
    for i in range(1, num_datasets + 1):
        # Row 1 - Log scale
        fig.update_yaxes(
            type="log",
            title_text="Loss (log scale)" if i == 1 else "",
            row=1, col=i,
            tickfont=dict(size=11),
            title_font=dict(size=12)
        )
        fig.update_xaxes(title_text="", row=1, col=i, tickfont=dict(size=11))

        # Row 2 - Change rate
        fig.update_yaxes(
            title_text="Change %" if i == 1 else "",
            row=2, col=i,
            tickfont=dict(size=11),
            title_font=dict(size=12)
        )
        fig.update_xaxes(title_text="Epoch", row=2, col=i, tickfont=dict(size=11))

    # Update layout
    fig.update_layout(
        title_text="Convergence and Stability Analysis",
        title_font=dict(size=26 if full_size else 22),
        title_x=0.5,
        title_xanchor='center',
        height=1000 if full_size else 900,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12)
    )

    # Update subplot title font sizes
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=title_size if full_size else 14)

    return fig

def create_comparison_learning_dynamics_plot(datasets, full_size=False):
    """Create enhanced learning dynamics comparison with better visibility."""

    num_datasets = len(datasets)

    # Create subplot titles in correct row-major order
    row1_titles = [f"<b>{d['name']}</b><br>Learning Rate" for d in datasets]
    row2_titles = [f"<b>{d['name']}</b><br>Gradient Norm" for d in datasets]
    subplot_titles = row1_titles + row2_titles

    fig = make_subplots(
        rows=2, cols=num_datasets,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False} for _ in range(num_datasets)],
               [{"secondary_y": False} for _ in range(num_datasets)]],
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    colors_set = ['rgb(23, 190, 207)', 'rgb(188, 189, 34)', 'rgb(219, 64, 82)']

    for idx, dataset in enumerate(datasets):
        metrics = dataset['metrics']
        epochs = metrics['epoch']
        color = colors_set[idx % 3]
        col = idx + 1

        # Row 1: Learning Rate
        if len(metrics['learning_rate']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['learning_rate'],
                    name='Learning Rate',
                    line=dict(color=color, width=3),
                    mode='lines+markers',
                    marker=dict(size=6),
                    legendgroup=f'lr{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

            # Add vertical lines for LR drops instead of crowded annotations
            if len(metrics['learning_rate']) > 1:
                for i in range(1, len(metrics['learning_rate'])):
                    if metrics['learning_rate'][i] < metrics['learning_rate'][i - 1] * 0.5:
                        fig.add_vline(
                            x=epochs[i],
                            line_width=2,
                            line_dash="dash",
                            line_color="red",
                            opacity=0.5,
                            row=1, col=col
                        )

        # Row 2: Gradient Norm
        if len(metrics['gradient_norm']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['gradient_norm'],
                    name='Gradient Norm',
                    line=dict(color='gray', width=2),
                    mode='lines',
                    opacity=0.5,
                    legendgroup=f'grad{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )

            # Add rolling average for clearer trend
            if len(metrics['gradient_norm']) > 5:
                window = min(5, len(metrics['gradient_norm']) // 4)
                grad_smooth = np.convolve(metrics['gradient_norm'],
                                          np.ones(window) / window, mode='valid')
                epochs_smooth = epochs[window // 2:-(window // 2) if window > 1 else None]

                fig.add_trace(
                    go.Scatter(
                        x=epochs_smooth, y=grad_smooth,
                        name='Gradient (Smoothed)',
                        line=dict(color='red', width=3),
                        legendgroup=f'grad_smooth{col}',
                        showlegend=(col == 1)
                    ),
                    row=2, col=col
                )

    # Update axes
    for i in range(1, num_datasets + 1):
        # Row 1 - Learning Rate
        fig.update_yaxes(
            title_text="Learning Rate" if i == 1 else "",
            row=1, col=i,
            tickfont=dict(size=11),
            title_font=dict(size=12)
        )
        fig.update_xaxes(title_text="", row=1, col=i, tickfont=dict(size=11))

        # Row 2 - Gradient Norm
        fig.update_yaxes(
            title_text="Gradient Norm" if i == 1 else "",
            row=2, col=i,
            tickfont=dict(size=11),
            title_font=dict(size=12)
        )
        fig.update_xaxes(title_text="Epoch", row=2, col=i, tickfont=dict(size=11))

    # Update layout
    fig.update_layout(
        title_text="Learning Dynamics Analysis",
        title_font=dict(size=26 if full_size else 22),
        title_x=0.5,
        title_xanchor='center',
        height=1000 if full_size else 900,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12)
    )

    # Update subplot title font sizes
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=16 if full_size else 14)

    return fig

def create_comparison_detailed_subplots(datasets, full_size=False):
    """Create detailed subplot analysis for comparison - similar to single mode."""

    num_datasets = len(datasets)

    # Adjust dimensions
    height = 1600 if full_size else 1400
    title_size = 16 if full_size else 14

    # Create subplot titles in correct row-major order
    row1_titles = [f"<b>{d['name']}</b><br>Loss Comparison" for d in datasets]
    row2_titles = [f"<b>{d['name']}</b><br>Components Breakdown" for d in datasets]
    row3_titles = [f"<b>{d['name']}</b><br>Train-Val Gap" for d in datasets]
    subplot_titles = row1_titles + row2_titles + row3_titles

    # Create subplots grid - 3 rows, num_datasets columns
    fig = make_subplots(
        rows=3, cols=num_datasets,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False} for _ in range(num_datasets)],
               [{"secondary_y": False} for _ in range(num_datasets)],
               [{"secondary_y": False} for _ in range(num_datasets)]],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    colors_set = [
        {'train': 'rgb(31, 119, 180)', 'val': 'rgb(255, 127, 14)'},
        {'train': 'rgb(44, 160, 44)', 'val': 'rgb(214, 39, 40)'},
        {'train': 'rgb(148, 103, 189)', 'val': 'rgb(227, 119, 194)'}
    ]

    for idx, dataset in enumerate(datasets):
        metrics = dataset['metrics']
        epochs = metrics['epoch']
        colors = colors_set[idx % 3]
        col = idx + 1

        # Row 1: Overall Loss Comparison
        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['loss'],
                    name=f'Training',
                    line=dict(color=colors['train'], width=3),
                    mode='lines+markers',
                    marker=dict(size=6),
                    legendgroup=f'dataset{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['val_loss'],
                    name=f'Validation',
                    line=dict(color=colors['val'], width=3),
                    mode='lines+markers',
                    marker=dict(size=6),
                    legendgroup=f'dataset{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

            # Add best point annotation
            if len(metrics['val_loss']) > 0:
                best_idx = metrics['val_loss'].index(min(metrics['val_loss']))
                best_epoch = epochs[best_idx]
                best_val = min(metrics['val_loss'])

                fig.add_annotation(
                    x=best_epoch,
                    y=best_val,
                    text=f"Best<br>{best_val:.4f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=colors['val'],
                    bgcolor="white",
                    bordercolor=colors['val'],
                    row=1, col=col
                )

        # Row 2: Loss Components Breakdown
        component_colors = {
            'det': 'rgba(44, 160, 44, 0.7)',
            'cls': 'rgba(214, 39, 40, 0.7)',
            'box': 'rgba(148, 103, 189, 0.7)',
            'reg': 'rgba(140, 86, 75, 0.7)'
        }

        if len(metrics['det_loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['det_loss'],
                    name='Detection',
                    stackgroup='components',
                    fillcolor=component_colors['det'],
                    line=dict(width=0.5),
                    legendgroup=f'components{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )
        if len(metrics['cls_loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['cls_loss'],
                    name='Classification',
                    stackgroup='components',
                    fillcolor=component_colors['cls'],
                    line=dict(width=0.5),
                    legendgroup=f'components{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )
        if len(metrics['box_loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['box_loss'],
                    name='Box',
                    stackgroup='components',
                    fillcolor=component_colors['box'],
                    line=dict(width=0.5),
                    legendgroup=f'components{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )

        # Row 3: Training vs Validation Gap
        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            gaps = [t - v for t, v in zip(metrics['loss'], metrics['val_loss'])]

            # Color based on gap (green = small gap, red = large gap)
            gap_colors = ['green' if abs(g) < 0.05 else 'orange' if abs(g) < 0.1 else 'red' for g in gaps]

            fig.add_trace(
                go.Bar(
                    x=epochs, y=gaps,
                    name='Train-Val Gap',
                    marker_color=gap_colors,
                    opacity=0.7,
                    legendgroup=f'gap{col}',
                    showlegend=(col == 1)
                ),
                row=3, col=col
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=col)

    # Update axes labels
    for i in range(1, num_datasets + 1):
        # Row 1
        fig.update_xaxes(title_text="", row=1, col=i, tickfont=dict(size=11))
        fig.update_yaxes(
            title_text="Loss" if i == 1 else "",
            row=1, col=i,
            tickfont=dict(size=11),
            title_font=dict(size=12)
        )

        # Row 2
        fig.update_xaxes(title_text="", row=2, col=i, tickfont=dict(size=11))
        fig.update_yaxes(
            title_text="Loss Components" if i == 1 else "",
            row=2, col=i,
            tickfont=dict(size=11),
            title_font=dict(size=12)
        )

        # Row 3
        fig.update_xaxes(title_text="Epoch", row=3, col=i, tickfont=dict(size=11))
        fig.update_yaxes(
            title_text="Gap (Train - Val)" if i == 1 else "",
            row=3, col=i,
            tickfont=dict(size=11),
            title_font=dict(size=12)
        )

    # Update layout
    fig.update_layout(
        title_text="Detailed Training Analysis Comparison",
        title_font=dict(size=26 if full_size else 22),
        title_x=0.5,
        title_xanchor='center',
        height=height,
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12)
    )

    # Update subplot title font sizes
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=title_size)

    return fig


def visualize_training(log_filepath=None, spec_filepath=None, compare_mode=False):
    """Main function to visualize training results."""

    if compare_mode or log_filepath is None:
        # Open dataset selector GUI
        selector = DatasetSelector()
        selector.select_datasets()

        if not selector.datasets:
            print("No datasets selected")
            return False

        return visualize_training_comparison(selector.datasets)

    else:
        # Single dataset mode (original functionality)
        try:
            metrics, eval_metrics = parse_training_log(log_filepath)

            # Try to find spec file if not provided
            if spec_filepath is None:
                log_dir = Path(log_filepath).parent
                spec_filepath = log_dir / "training_specs.txt"

            model_specs = parse_model_specs(spec_filepath) if spec_filepath else {'main': {}, 'hyperparameters': {}}

            if len(metrics['epoch']) == 0:
                print("⚠️  No training data found in the log file!")
                return False

            print(f"✅ Found {len(metrics['epoch'])} epochs of training data")
            if eval_metrics['validation']:
                print(f"✅ Found validation metrics")
            if eval_metrics['test']:
                print(f"✅ Found test metrics")
            print("\n🔄 Generating visualizations...")

            # Create temp directory
            temp_dir = tempfile.gettempdir()

            # Create dashboard with enlargement functionality
            dashboard_html = create_dashboard_with_enlargement(metrics, eval_metrics, model_specs, temp_dir)

            # Calculate statistics
            stats = calculate_statistics(metrics, eval_metrics)
            stats_html = create_statistics_table(stats)

            # Save dashboard with UTF-8 encoding
            dashboard_file = os.path.join(temp_dir, "training_dashboard.html")
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)

            # Save statistics with UTF-8 encoding
            stats_file = os.path.join(temp_dir, "training_statistics.html")
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(stats_html)

            # Open both in browser
            print("\n📊 Opening visualizations in browser...")
            webbrowser.open(f"file://{dashboard_file}", new=2)
            webbrowser.open(f"file://{stats_file}", new=2)

            print("\n✅ Visualizations opened in browser:")
            print(f"  1. Dashboard: {dashboard_file}")
            print(f"  2. Statistics: {stats_file}")

            return True

        except Exception as e:
            print(f"\n⚠️  Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    import sys

    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Comparison mode
            visualize_training(compare_mode=True)
        else:
            # Single file mode
            log_file = sys.argv[1].strip('"').strip("'")
            visualize_training(log_file)
    else:
        # Interactive mode
        print("Training Log Visualizer")
        print("-" * 40)
        print("1. Visualize single training log")
        print("2. Compare multiple training logs")
        print("-" * 40)

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "2":
            visualize_training(compare_mode=True)
        else:
            log_file = input("Enter the path to your log file: ").strip()
            log_file = log_file.strip('"').strip("'")
            visualize_training(log_file)

        input("\nPress Enter to exit...")
