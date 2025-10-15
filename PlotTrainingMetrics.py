import ast
import os
import re
import tempfile
import webbrowser
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
    """Parse model specifications from training_specs.txt"""
    specs = {}

    try:
        with open(spec_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract key information using regex
        patterns = {
            'batch_size': r'Batch Size:\s+(\d+)',
            'epochs': r'Epochs:\s+(\d+)',
            'architecture': r'Architecture:\s+(\S+)',
            'backbone': r'backbone_name:\s+(\S+)',
            'num_classes': r'num_classes:\s+(\d+)',
            'image_size': r'image_size:\s+(\d+)',
            'learning_rate': r'learning_rate:\s+([\d.]+)',
            'optimizer': r'optimizer:\s+(\S+)',
            'model_name': r'Model Name:\s+(.+)',
            'train_whole_model': r'Train Whole Model:\s+(\w+)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                specs[key] = match.group(1)

    except FileNotFoundError:
        print("⚠️  Model specifications file not found")

    return specs


def create_evaluation_metrics_plot(eval_metrics, full_size=False):
    """Create a plot for AP/AR evaluation metrics."""

    fig = go.Figure()

    # Prepare data for both validation and test sets
    categories = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                  'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']

    val_values = []
    test_values = []

    for cat in categories:
        val_values.append(eval_metrics['validation'].get(cat, 0))
        test_values.append(eval_metrics['test'].get(cat, 0))

    # Create grouped bar chart
    fig.add_trace(go.Bar(
        name='Validation',
        x=categories,
        y=val_values,
        marker_color='rgb(31, 119, 180)',
        text=[f'{v:.3f}' for v in val_values],
        textposition='outside'
    ))

    fig.add_trace(go.Bar(
        name='Test',
        x=categories,
        y=test_values,
        marker_color='rgb(255, 127, 14)',
        text=[f'{v:.3f}' for v in test_values],
        textposition='outside'
    ))

    fig.update_layout(
        title={
            'text': 'Model Evaluation Metrics (AP/AR)',
            'font': {'size': 24 if full_size else 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_tickangle=-45,
        barmode='group',
        height=600 if full_size else 500,
        yaxis_title='Score',
        xaxis_title='Metric',
        hovermode='x unified',
        showlegend=True,
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=12 if full_size else 11)
    )

    # Add value annotations for better readability
    fig.update_traces(texttemplate='%{text}', textfont_size=10 if not full_size else 12)

    return fig


def create_model_specs_card(specs):
    """Create an HTML card displaying model specifications."""

    html = """
    <div style="background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h2 style="color: #333; margin-bottom: 20px; border-bottom: 2px solid #667eea; 
                   padding-bottom: 10px;">🔧 Model Specifications</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 20px;">
    """

    spec_items = [
        ('Model Name', specs.get('model_name', 'N/A'), '🏷️'),
        ('Architecture', specs.get('architecture', 'N/A'), '🏗️'),
        ('Backbone', specs.get('backbone', 'N/A'), '🦴'),
        ('Batch Size', specs.get('batch_size', 'N/A'), '📦'),
        ('Epochs', specs.get('epochs', 'N/A'), '🔄'),
        ('Image Size', specs.get('image_size', 'N/A'), '📐'),
        ('Classes', specs.get('num_classes', 'N/A'), '🏷️'),
        ('Learning Rate', specs.get('learning_rate', 'N/A'), '📈'),
        ('Optimizer', specs.get('optimizer', 'N/A'), '⚙️'),
        ('Train Whole Model', specs.get('train_whole_model', 'N/A'), '🎯')
    ]

    for label, value, icon in spec_items:
        html += f"""
        <div style="background: #f8f9fa; padding: 15px; border-radius: 10px; 
                    border-left: 4px solid #667eea;">
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <span style="font-size: 1.5em; margin-right: 10px;">{icon}</span>
                <strong style="color: #495057;">{label}</strong>
            </div>
            <div style="font-size: 1.1em; color: #212529; font-family: monospace;">
                {value}
            </div>
        </div>
        """

    html += """
        </div>
    </div>
    """

    return html


def create_interactive_plot(metrics, full_size=False):
    """Create highly detailed interactive plot with Plotly."""

    epochs = metrics['epoch']

    # Adjust height based on whether it's full size
    height = 900 if full_size else 700

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    # Define plot configurations with better organization
    plot_configs = [
        # Primary losses
        ('loss', 'Training Loss', 'rgb(31, 119, 180)', 'solid', True, 2.5),
        ('val_loss', 'Validation Loss', 'rgb(255, 127, 14)', 'solid', True, 2.5),

        # Detection losses
        ('det_loss', 'Detection Loss (Train)', 'rgb(44, 160, 44)', 'dash', True, 2),
        ('val_det_loss', 'Detection Loss (Val)', 'rgb(152, 223, 138)', 'dash', True, 2),

        # Classification losses
        ('cls_loss', 'Classification Loss (Train)', 'rgb(214, 39, 40)', 'dot', True, 2),
        ('val_cls_loss', 'Classification Loss (Val)', 'rgb(255, 152, 150)', 'dot', True, 2),

        # Box losses
        ('box_loss', 'Box Loss (Train)', 'rgb(148, 103, 189)', 'dashdot', True, 2),
        ('val_box_loss', 'Box Loss (Val)', 'rgb(197, 176, 213)', 'dashdot', True, 2),

        # Regularization losses
        ('reg_l2_loss', 'L2 Regularization (Train)', 'rgb(140, 86, 75)', 'longdash', True, 1.5),
        ('val_reg_l2_loss', 'L2 Regularization (Val)', 'rgb(196, 156, 148)', 'longdash', True, 1.5),

        # Learning rate and gradient (secondary y-axis)
        ('learning_rate', 'Learning Rate', 'rgb(23, 190, 207)', 'solid', False, 2),
        ('gradient_norm', 'Gradient Norm', 'rgb(127, 127, 127)', 'longdashdot', False, 2),
    ]

    # Add traces
    for metric_key, label, color, dash, use_primary_y, width in plot_configs:
        if metric_key in metrics and len(metrics[metric_key]) > 0:
            # Make the main losses visible by default
            visible = True if metric_key in ['loss', 'val_loss'] else 'legendonly'

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
                    visible=visible
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
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.3)",
            borderwidth=1,
            font=dict(size=12 if not full_size else 14)
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        paper_bgcolor='white',
        font=dict(size=11 if not full_size else 13),
        margin=dict(r=200)  # Add right margin for legend
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
                background: #f8f9fa;
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
                    <h3>Validation Performance</h3>
                    <div class="metric-value">{eval_metrics['validation'].get('AP', 0):.3f}</div>
                    <div class="metric-label">Average Precision (AP)</div>
                    <div style="margin-top: 10px;">
                        <strong>AP50:</strong> {eval_metrics['validation'].get('AP50', 0):.3f} | 
                        <strong>AP75:</strong> {eval_metrics['validation'].get('AP75', 0):.3f}
                    </div>
                </div>
                <div class="eval-card">
                    <h3>Test Performance</h3>
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
            <div class="plot-title">📈 Interactive Training Metrics</div>
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


def visualize_training(log_filepath, spec_filepath=None):
    """Main function to visualize training results with simultaneous windows."""
    try:
        metrics, eval_metrics = parse_training_log(log_filepath)

        # Try to find spec file if not provided
        if spec_filepath is None:
            log_dir = Path(log_filepath).parent
            spec_filepath = log_dir / "training_specs.txt"

        model_specs = parse_model_specs(spec_filepath) if spec_filepath else {}

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
        print(f"  1. Dashboard (with enlargeable plots): {dashboard_file}")
        print(f"  2. Statistics Summary: {stats_file}")
        print("\n💡 Tip: Hover over any plot and click the ⤢ button to open it in a larger window!")
        print("     Individual subplots also have their own enlarge buttons.")

        # Print summary to console
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        for key, value in stats.items():
            display_name = key.replace('_', ' ').title()
            if isinstance(value, float):
                print(f"{display_name:.<30} {value:.4f}")
            else:
                print(f"{display_name:.<30} {value}")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n⚠️  Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Standalone mode - ask for file
    log_file = input("Enter the path to your log file: ").strip()

    # Remove quotes if present
    log_file = log_file.strip('"').strip("'")

    try:
        if visualize_training(log_file):
            print("\n" + "=" * 80)
            print("✅ Visualization complete!")
            print("=" * 80)
            input("\nPress Enter to exit...")
    except FileNotFoundError:
        print(f"Error: Could not find file '{log_file}'")
        input("\nPress Enter to exit...")

#
#
# def parse_training_log(log_file):
#     """Extract training metrics from log file - handles all progress formats."""
#
#     metrics = {
#         'epoch': [],
#         'det_loss': [],
#         'cls_loss': [],
#         'box_loss': [],
#         'reg_l2_loss': [],
#         'loss': [],
#         'learning_rate': [],
#         'gradient_norm': [],
#         'val_det_loss': [],
#         'val_cls_loss': [],
#         'val_box_loss': [],
#         'val_reg_l2_loss': [],
#         'val_loss': []
#     }
#
#     # Pattern to match the epoch number
#     epoch_pattern = r'Epoch (\d+)/(\d+)'
#     # Pattern to match all metrics
#     metrics_pattern = r'(\w+):\s+([\d.e+-]+)'
#
#     with open(log_file, 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#
#     current_epoch = None
#     epoch_data_buffer = {}
#
#     for i, line in enumerate(lines):
#         # Check for epoch start
#         epoch_match = re.search(epoch_pattern, line)
#         if epoch_match:
#             # Save previous epoch data if exists
#             if current_epoch and epoch_data_buffer:
#                 if 'loss' in epoch_data_buffer and 'val_loss' in epoch_data_buffer:
#                     for key in metrics.keys():
#                         if key in epoch_data_buffer:
#                             metrics[key].append(epoch_data_buffer[key])
#                         elif key == 'epoch':
#                             metrics[key].append(current_epoch)
#
#             # Start new epoch
#             current_epoch = int(epoch_match.group(1))
#             epoch_data_buffer = {'epoch': current_epoch}
#
#         # Look for completed progress bar with validation metrics
#         if (current_epoch and
#                 '[==============================]' in line and
#                 'val_loss:' in line and
#                 's/step' in line):
#
#             # Extract all metrics from this line
#             found_metrics = re.findall(metrics_pattern, line)
#
#             for metric_name, value in found_metrics:
#                 try:
#                     epoch_data_buffer[metric_name] = float(value)
#                 except ValueError:
#                     continue
#
#     # Don't forget the last epoch
#     if current_epoch and epoch_data_buffer:
#         if 'loss' in epoch_data_buffer and 'val_loss' in epoch_data_buffer:
#             for key in metrics.keys():
#                 if key in epoch_data_buffer:
#                     metrics[key].append(epoch_data_buffer[key])
#                 elif key == 'epoch':
#                     metrics[key].append(current_epoch)
#
#     return metrics
#
#
# def create_interactive_plot(metrics, full_size=False):
#     """Create highly detailed interactive plot with Plotly."""
#
#     epochs = metrics['epoch']
#
#     # Adjust height based on whether it's full size
#     height = 900 if full_size else 700
#
#     # Create figure with secondary y-axis - removed subplot_titles which was causing issues
#     fig = make_subplots(
#         rows=1, cols=1,
#         specs=[[{"secondary_y": True}]]
#     )
#
#     # Define plot configurations with better organization
#     plot_configs = [
#         # Primary losses
#         ('loss', 'Training Loss', 'rgb(31, 119, 180)', 'solid', True, 2.5),
#         ('val_loss', 'Validation Loss', 'rgb(255, 127, 14)', 'solid', True, 2.5),
#
#         # Detection losses
#         ('det_loss', 'Detection Loss (Train)', 'rgb(44, 160, 44)', 'dash', True, 2),
#         ('val_det_loss', 'Detection Loss (Val)', 'rgb(152, 223, 138)', 'dash', True, 2),
#
#         # Classification losses
#         ('cls_loss', 'Classification Loss (Train)', 'rgb(214, 39, 40)', 'dot', True, 2),
#         ('val_cls_loss', 'Classification Loss (Val)', 'rgb(255, 152, 150)', 'dot', True, 2),
#
#         # Box losses
#         ('box_loss', 'Box Loss (Train)', 'rgb(148, 103, 189)', 'dashdot', True, 2),
#         ('val_box_loss', 'Box Loss (Val)', 'rgb(197, 176, 213)', 'dashdot', True, 2),
#
#         # Regularization losses
#         ('reg_l2_loss', 'L2 Regularization (Train)', 'rgb(140, 86, 75)', 'longdash', True, 1.5),
#         ('val_reg_l2_loss', 'L2 Regularization (Val)', 'rgb(196, 156, 148)', 'longdash', True, 1.5),
#
#         # Learning rate and gradient (secondary y-axis)
#         ('learning_rate', 'Learning Rate', 'rgb(23, 190, 207)', 'solid', False, 2),
#         ('gradient_norm', 'Gradient Norm', 'rgb(127, 127, 127)', 'longdashdot', False, 2),
#     ]
#
#     # Add traces
#     for metric_key, label, color, dash, use_primary_y, width in plot_configs:
#         if metric_key in metrics and len(metrics[metric_key]) > 0:
#             # Make the main losses visible by default
#             visible = True if metric_key in ['loss', 'val_loss'] else 'legendonly'
#
#             fig.add_trace(
#                 go.Scatter(
#                     x=epochs,
#                     y=metrics[metric_key],
#                     name=label,
#                     mode='lines+markers',
#                     line=dict(color=color, dash=dash, width=width),
#                     marker=dict(size=4 if not full_size else 6),
#                     hovertemplate='<b>' + label + '</b><br>' +
#                                   'Epoch: %{x}<br>' +
#                                   'Value: %{y:.6f}<br>' +
#                                   '<extra></extra>',
#                     visible=visible
#                 ),
#                 secondary_y=not use_primary_y
#             )
#
#     # Calculate additional statistics for annotations
#     if len(metrics['val_loss']) > 0:
#         best_epoch = metrics['epoch'][metrics['val_loss'].index(min(metrics['val_loss']))]
#         best_val_loss = min(metrics['val_loss'])
#
#         # Add annotation for best validation loss
#         fig.add_annotation(
#             x=best_epoch,
#             y=best_val_loss,
#             text=f"Best Val Loss<br>{best_val_loss:.4f}",
#             showarrow=True,
#             arrowhead=2,
#             arrowsize=1,
#             arrowwidth=2,
#             arrowcolor="red",
#             ax=-50,
#             ay=-40,
#             bgcolor="rgba(255, 255, 255, 0.9)",
#             bordercolor="red",
#             borderwidth=2,
#             font=dict(size=12 if not full_size else 14)
#         )
#
#     # Update layout
#     fig.update_xaxes(
#         title_text="Epoch",
#         gridcolor='rgba(128, 128, 128, 0.2)',
#         showgrid=True,
#         zeroline=True,
#         title_font=dict(size=14 if not full_size else 16)
#     )
#
#     fig.update_yaxes(
#         title_text="Loss Value",
#         secondary_y=False,
#         gridcolor='rgba(128, 128, 128, 0.2)',
#         showgrid=True,
#         title_font=dict(size=14 if not full_size else 16)
#     )
#
#     fig.update_yaxes(
#         title_text="Learning Rate / Gradient Norm",
#         secondary_y=True,
#         gridcolor='rgba(128, 128, 128, 0.1)',
#         showgrid=True,
#         title_font=dict(size=14 if not full_size else 16)
#     )
#
#     fig.update_layout(
#         title={
#             'text': 'Detailed Training Metrics Visualization',
#             'font': {'size': 24 if full_size else 20, 'color': 'rgb(50, 50, 50)'},
#             'x': 0.5,
#             'xanchor': 'center'
#         },
#         hovermode='x unified',
#         height=height,
#         showlegend=True,
#         legend=dict(
#             orientation="v",
#             yanchor="top",
#             y=1,
#             xanchor="left",
#             x=1.02,
#             bgcolor="rgba(255, 255, 255, 0.9)",
#             bordercolor="rgba(0, 0, 0, 0.3)",
#             borderwidth=1,
#             font=dict(size=12 if not full_size else 14)
#         ),
#         plot_bgcolor='rgba(240, 240, 240, 0.5)',
#         paper_bgcolor='white',
#         font=dict(size=11 if not full_size else 13),
#         margin=dict(r=200)  # Add right margin for legend
#     )
#
#     # Add range slider
#     fig.update_xaxes(rangeslider_visible=True)
#
#     return fig
#
#
# def create_detailed_subplots(metrics, full_size=False):
#     """Create detailed subplot analysis with multiple perspectives."""
#
#     epochs = metrics['epoch']
#
#     # Adjust dimensions based on whether it's full size
#     height = 1400 if full_size else 1000
#     title_size = 18 if full_size else 14
#
#     # Create subplots
#     fig = make_subplots(
#         rows=3, cols=2,
#         subplot_titles=(
#             'Overall Loss Comparison',
#             'Loss Components Breakdown',
#             'Training vs Validation Gap',
#             'Learning Dynamics',
#             'Loss Ratios and Trends',
#             'Convergence Analysis'
#         ),
#         specs=[
#             [{"secondary_y": False}, {"secondary_y": False}],
#             [{"secondary_y": False}, {"secondary_y": True}],
#             [{"secondary_y": False}, {"secondary_y": False}]
#         ],
#         vertical_spacing=0.12,
#         horizontal_spacing=0.15
#     )
#
#     # Update subplot title font sizes
#     for annotation in fig['layout']['annotations']:
#         annotation['font'] = dict(size=title_size)
#
#     # 1. Overall Loss Comparison - Make sure this displays properly
#     if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
#         fig.add_trace(
#             go.Scatter(x=epochs, y=metrics['loss'], name='Training Loss',
#                        line=dict(color='blue', width=3 if full_size else 2),
#                        mode='lines+markers', marker=dict(size=4)),
#             row=1, col=1
#         )
#         fig.add_trace(
#             go.Scatter(x=epochs, y=metrics['val_loss'], name='Validation Loss',
#                        line=dict(color='red', width=3 if full_size else 2),
#                        mode='lines+markers', marker=dict(size=4)),
#             row=1, col=1
#         )
#
#         # Add smoothed trend lines
#         if len(epochs) > 5:
#             window = min(5, len(epochs) // 4)
#             train_smooth = np.convolve(metrics['loss'], np.ones(window) / window, mode='valid')
#             val_smooth = np.convolve(metrics['val_loss'], np.ones(window) / window, mode='valid')
#             epochs_smooth = epochs[window // 2:-(window // 2) if window > 1 else None]
#
#             fig.add_trace(
#                 go.Scatter(x=epochs_smooth, y=train_smooth, name='Training (Smoothed)',
#                            line=dict(color='lightblue', width=2 if full_size else 1, dash='dash'),
#                            showlegend=True),
#                 row=1, col=1
#             )
#             fig.add_trace(
#                 go.Scatter(x=epochs_smooth, y=val_smooth, name='Validation (Smoothed)',
#                            line=dict(color='pink', width=2 if full_size else 1, dash='dash'),
#                            showlegend=True),
#                 row=1, col=1
#             )
#
#     # 2. Loss Components Breakdown (Stacked Area)
#     if len(metrics['det_loss']) > 0:
#         fig.add_trace(
#             go.Scatter(x=epochs, y=metrics['det_loss'], name='Detection',
#                        stackgroup='one', fillcolor='rgba(44, 160, 44, 0.5)'),
#             row=1, col=2
#         )
#     if len(metrics['cls_loss']) > 0:
#         fig.add_trace(
#             go.Scatter(x=epochs, y=metrics['cls_loss'], name='Classification',
#                        stackgroup='one', fillcolor='rgba(214, 39, 40, 0.5)'),
#             row=1, col=2
#         )
#     if len(metrics['box_loss']) > 0:
#         fig.add_trace(
#             go.Scatter(x=epochs, y=metrics['box_loss'], name='Box',
#                        stackgroup='one', fillcolor='rgba(148, 103, 189, 0.5)'),
#             row=1, col=2
#         )
#
#     # 3. Training vs Validation Gap
#     if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
#         gaps = [t - v for t, v in zip(metrics['loss'], metrics['val_loss'])]
#         fig.add_trace(
#             go.Scatter(x=epochs, y=gaps, name='Train-Val Gap',
#                        fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.3)',
#                        line=dict(color='orange', width=3 if full_size else 2)),
#             row=2, col=1
#         )
#         fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
#
#     # 4. Learning Dynamics (dual axis)
#     if len(metrics['learning_rate']) > 0:
#         fig.add_trace(
#             go.Scatter(x=epochs, y=metrics['learning_rate'], name='Learning Rate',
#                        line=dict(color='cyan', width=3 if full_size else 2)),
#             row=2, col=2, secondary_y=False
#         )
#     if len(metrics['gradient_norm']) > 0:
#         fig.add_trace(
#             go.Scatter(x=epochs, y=metrics['gradient_norm'], name='Gradient Norm',
#                        line=dict(color='gray', width=3 if full_size else 2)),
#             row=2, col=2, secondary_y=True
#         )
#
#     # 5. Loss Ratios and Trends
#     if len(metrics['loss']) > 1:
#         loss_change_rate = [0] + [(metrics['loss'][i] - metrics['loss'][i - 1]) / metrics['loss'][i - 1] * 100
#                                   if metrics['loss'][i - 1] != 0 else 0
#                                   for i in range(1, len(metrics['loss']))]
#         val_loss_change_rate = [0] + [
#             (metrics['val_loss'][i] - metrics['val_loss'][i - 1]) / metrics['val_loss'][i - 1] * 100
#             if metrics['val_loss'][i - 1] != 0 else 0
#             for i in range(1, len(metrics['val_loss']))]
#
#         fig.add_trace(
#             go.Bar(x=epochs, y=loss_change_rate, name='Train Loss Change %',
#                    marker_color='blue', opacity=0.6),
#             row=3, col=1
#         )
#         fig.add_trace(
#             go.Bar(x=epochs, y=val_loss_change_rate, name='Val Loss Change %',
#                    marker_color='red', opacity=0.6),
#             row=3, col=1
#         )
#
#     # 6. Convergence Analysis (Log scale)
#     if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
#         # Filter out zero or negative values for log scale
#         positive_loss = [l if l > 0 else None for l in metrics['loss']]
#         positive_val_loss = [l if l > 0 else None for l in metrics['val_loss']]
#
#         fig.add_trace(
#             go.Scatter(x=epochs, y=positive_loss, name='Training (log)',
#                        line=dict(color='blue', width=3 if full_size else 2),
#                        connectgaps=False),
#             row=3, col=2
#         )
#         fig.add_trace(
#             go.Scatter(x=epochs, y=positive_val_loss, name='Validation (log)',
#                        line=dict(color='red', width=3 if full_size else 2),
#                        connectgaps=False),
#             row=3, col=2
#         )
#
#     # Update axes with larger fonts for full size
#     axis_font_size = 14 if full_size else 11
#
#     fig.update_xaxes(title_text="Epoch", row=3, col=1, title_font=dict(size=axis_font_size))
#     fig.update_xaxes(title_text="Epoch", row=3, col=2, title_font=dict(size=axis_font_size))
#     fig.update_yaxes(title_text="Loss", row=1, col=1, title_font=dict(size=axis_font_size))
#     fig.update_yaxes(title_text="Loss", row=1, col=2, title_font=dict(size=axis_font_size))
#     fig.update_yaxes(title_text="Gap", row=2, col=1, title_font=dict(size=axis_font_size))
#     fig.update_yaxes(title_text="Learning Rate", row=2, col=2, secondary_y=False, title_font=dict(size=axis_font_size))
#     fig.update_yaxes(title_text="Gradient Norm", row=2, col=2, secondary_y=True, title_font=dict(size=axis_font_size))
#     fig.update_yaxes(title_text="Change %", row=3, col=1, title_font=dict(size=axis_font_size))
#     fig.update_yaxes(title_text="Loss (log scale)", type="log", row=3, col=2, title_font=dict(size=axis_font_size))
#
#     # Update layout
#     fig.update_layout(
#         title_text="Comprehensive Training Analysis",
#         title_font=dict(size=24 if full_size else 20),
#         title_x=0.5,
#         title_xanchor='center',
#         height=height,
#         showlegend=True,
#         hovermode='x unified',
#         plot_bgcolor='rgba(240, 240, 240, 0.5)',
#         paper_bgcolor='white',
#         font=dict(size=12 if full_size else 11)
#     )
#
#     return fig
#
#
# def create_individual_plot_html(metrics, plot_type, plot_num, temp_dir):
#     """Create HTML for individual plot with full-size version."""
#
#     epochs = metrics['epoch']
#
#     if plot_type == "overall_loss":
#         fig = go.Figure()
#
#         if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
#             fig.add_trace(go.Scatter(x=epochs, y=metrics['loss'], name='Training Loss',
#                                      line=dict(color='blue', width=3),
#                                      mode='lines+markers', marker=dict(size=6)))
#             fig.add_trace(go.Scatter(x=epochs, y=metrics['val_loss'], name='Validation Loss',
#                                      line=dict(color='red', width=3),
#                                      mode='lines+markers', marker=dict(size=6)))
#
#             # Add smoothed lines if enough data
#             if len(epochs) > 5:
#                 window = min(5, len(epochs) // 4)
#                 train_smooth = np.convolve(metrics['loss'], np.ones(window) / window, mode='valid')
#                 val_smooth = np.convolve(metrics['val_loss'], np.ones(window) / window, mode='valid')
#                 epochs_smooth = epochs[window // 2:-(window // 2) if window > 1 else None]
#
#                 fig.add_trace(go.Scatter(x=epochs_smooth, y=train_smooth, name='Training (Smoothed)',
#                                          line=dict(color='lightblue', width=2, dash='dash')))
#                 fig.add_trace(go.Scatter(x=epochs_smooth, y=val_smooth, name='Validation (Smoothed)',
#                                          line=dict(color='pink', width=2, dash='dash')))
#
#         fig.update_layout(
#             title="Overall Loss Comparison",
#             xaxis_title="Epoch",
#             yaxis_title="Loss",
#             height=800,
#             hovermode='x unified',
#             showlegend=True
#         )
#
#     elif plot_type == "components":
#         fig = go.Figure()
#
#         if len(metrics['det_loss']) > 0:
#             fig.add_trace(go.Scatter(x=epochs, y=metrics['det_loss'], name='Detection',
#                                      stackgroup='one', fillcolor='rgba(44, 160, 44, 0.5)'))
#         if len(metrics['cls_loss']) > 0:
#             fig.add_trace(go.Scatter(x=epochs, y=metrics['cls_loss'], name='Classification',
#                                      stackgroup='one', fillcolor='rgba(214, 39, 40, 0.5)'))
#         if len(metrics['box_loss']) > 0:
#             fig.add_trace(go.Scatter(x=epochs, y=metrics['box_loss'], name='Box',
#                                      stackgroup='one', fillcolor='rgba(148, 103, 189, 0.5)'))
#
#         fig.update_layout(
#             title="Loss Components Breakdown",
#             xaxis_title="Epoch",
#             yaxis_title="Loss",
#             height=800,
#             hovermode='x unified'
#         )
#
#     elif plot_type == "gap":
#         fig = go.Figure()
#
#         if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
#             gaps = [t - v for t, v in zip(metrics['loss'], metrics['val_loss'])]
#
#             fig.add_trace(go.Scatter(x=epochs, y=gaps, name='Train-Val Gap',
#                                      fill='tozeroy', fillcolor='rgba(255, 127, 14, 0.3)',
#                                      line=dict(color='orange', width=3)))
#             fig.add_hline(y=0, line_dash="dash", line_color="gray")
#
#         fig.update_layout(
#             title="Training vs Validation Gap",
#             xaxis_title="Epoch",
#             yaxis_title="Gap",
#             height=800,
#             hovermode='x unified'
#         )
#
#     elif plot_type == "learning":
#         fig = make_subplots(specs=[[{"secondary_y": True}]])
#
#         if len(metrics['learning_rate']) > 0:
#             fig.add_trace(go.Scatter(x=epochs, y=metrics['learning_rate'], name='Learning Rate',
#                                      line=dict(color='cyan', width=3)), secondary_y=False)
#         if len(metrics['gradient_norm']) > 0:
#             fig.add_trace(go.Scatter(x=epochs, y=metrics['gradient_norm'], name='Gradient Norm',
#                                      line=dict(color='gray', width=3)), secondary_y=True)
#
#         fig.update_xaxes(title_text="Epoch")
#         fig.update_yaxes(title_text="Learning Rate", secondary_y=False)
#         fig.update_yaxes(title_text="Gradient Norm", secondary_y=True)
#         fig.update_layout(title="Learning Dynamics", height=800, hovermode='x unified')
#
#     elif plot_type == "ratios":
#         fig = go.Figure()
#         if len(metrics['loss']) > 1:
#             loss_change_rate = [0] + [(metrics['loss'][i] - metrics['loss'][i - 1]) / metrics['loss'][i - 1] * 100
#                                       if metrics['loss'][i - 1] != 0 else 0
#                                       for i in range(1, len(metrics['loss']))]
#             val_loss_change_rate = [0] + [
#                 (metrics['val_loss'][i] - metrics['val_loss'][i - 1]) / metrics['val_loss'][i - 1] * 100
#                 if metrics['val_loss'][i - 1] != 0 else 0
#                 for i in range(1, len(metrics['val_loss']))]
#
#             fig.add_trace(go.Bar(x=epochs, y=loss_change_rate, name='Train Loss Change %',
#                                  marker_color='blue', opacity=0.6))
#             fig.add_trace(go.Bar(x=epochs, y=val_loss_change_rate, name='Val Loss Change %',
#                                  marker_color='red', opacity=0.6))
#
#         fig.update_layout(
#             title="Loss Change Ratios",
#             xaxis_title="Epoch",
#             yaxis_title="Change %",
#             height=800,
#             hovermode='x unified'
#         )
#
#     elif plot_type == "convergence":
#         fig = go.Figure()
#
#         if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
#             # Filter out zero or negative values for log scale
#             positive_loss = [l if l > 0 else None for l in metrics['loss']]
#             positive_val_loss = [l if l > 0 else None for l in metrics['val_loss']]
#
#             fig.add_trace(go.Scatter(x=epochs, y=positive_loss, name='Training (log)',
#                                      line=dict(color='blue', width=3),
#                                      connectgaps=False))
#             fig.add_trace(go.Scatter(x=epochs, y=positive_val_loss, name='Validation (log)',
#                                      line=dict(color='red', width=3),
#                                      connectgaps=False))
#
#         fig.update_yaxes(type="log")
#         fig.update_layout(
#             title="Convergence Analysis (Log Scale)",
#             xaxis_title="Epoch",
#             yaxis_title="Loss (log scale)",
#             height=800,
#             hovermode='x unified'
#         )
#     else:
#         return None
#
#     # Save the individual plot
#     filename = f"plot_{plot_num}_{plot_type}.html"
#     filepath = os.path.join(temp_dir, filename)
#     fig.write_html(filepath, auto_open=False)
#
#     return filename
#
#
# def create_dashboard_with_enlargement(metrics, temp_dir):
#     """Create a dashboard HTML with enlargement functionality."""
#
#     # Create full-size versions
#     fig1_full = create_interactive_plot(metrics, full_size=True)
#     fig2_full = create_detailed_subplots(metrics, full_size=True)
#
#     # Create regular-size versions
#     fig1 = create_interactive_plot(metrics, full_size=False)
#     fig2 = create_detailed_subplots(metrics, full_size=False)
#
#     # Save full-size versions
#     file1_full = os.path.join(temp_dir, "training_interactive_full.html")
#     fig1_full.write_html(file1_full, auto_open=False)
#
#     file2_full = os.path.join(temp_dir, "training_detailed_full.html")
#     fig2_full.write_html(file2_full, auto_open=False)
#
#     # Create individual plot files for detailed view
#     individual_plots = []
#     for plot_type, num in [("overall_loss", 1), ("components", 2), ("gap", 3),
#                            ("learning", 4), ("ratios", 5), ("convergence", 6)]:
#         filename = create_individual_plot_html(metrics, plot_type, num, temp_dir)
#         if filename:
#             individual_plots.append(filename)
#         else:
#             individual_plots.append("")
#
#     # Convert figures to HTML strings with CDN
#     fig1_html = fig1.to_html(include_plotlyjs='cdn', div_id="plot1")
#     fig2_html = fig2.to_html(include_plotlyjs='cdn', div_id="plot2")
#
#     # Create the main dashboard HTML
#     dashboard_html = f"""
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <meta charset="UTF-8">
#         <title>Training Visualization Dashboard</title>
#         <style>
#             body {{
#                 font-family: Arial, sans-serif;
#                 margin: 0;
#                 padding: 20px;
#                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                 min-height: 100vh;
#             }}
#
#             .dashboard-header {{
#                 text-align: center;
#                 color: white;
#                 margin-bottom: 30px;
#             }}
#
#             .dashboard-header h1 {{
#                 font-size: 2.5em;
#                 margin: 0;
#                 text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#             }}
#
#             .dashboard-header p {{
#                 font-size: 1.2em;
#                 margin-top: 10px;
#                 opacity: 0.9;
#             }}
#
#             .plot-container {{
#                 background: white;
#                 border-radius: 15px;
#                 padding: 20px;
#                 margin-bottom: 30px;
#                 box-shadow: 0 10px 30px rgba(0,0,0,0.2);
#                 position: relative;
#                 transition: all 0.3s ease;
#             }}
#
#             .plot-container:hover {{
#                 box-shadow: 0 15px 40px rgba(0,0,0,0.3);
#                 transform: translateY(-2px);
#             }}
#
#             .enlarge-button {{
#                 position: absolute;
#                 top: 10px;
#                 right: 10px;
#                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                 color: white;
#                 border: none;
#                 border-radius: 50%;
#                 width: 40px;
#                 height: 40px;
#                 font-size: 20px;
#                 cursor: pointer;
#                 opacity: 0.7;
#                 transition: all 0.3s ease;
#                 z-index: 1000;
#                 display: flex;
#                 align-items: center;
#                 justify-content: center;
#             }}
#
#             .enlarge-button:hover {{
#                 opacity: 1;
#                 transform: scale(1.1);
#                 box-shadow: 0 5px 15px rgba(0,0,0,0.3);
#             }}
#
#             .plot-title {{
#                 font-size: 1.5em;
#                 font-weight: bold;
#                 color: #333;
#                 margin-bottom: 15px;
#                 padding-bottom: 10px;
#                 border-bottom: 2px solid #667eea;
#             }}
#
#             .subplot-grid {{
#                 display: grid;
#                 grid-template-columns: repeat(2, 1fr);
#                 gap: 20px;
#                 margin-top: 20px;
#             }}
#
#             .subplot-item {{
#                 background: #f8f9fa;
#                 border-radius: 10px;
#                 padding: 15px;
#                 position: relative;
#                 transition: all 0.3s ease;
#             }}
#
#             .subplot-item:hover {{
#                 background: #e9ecef;
#                 transform: scale(1.02);
#             }}
#
#             .subplot-enlarge {{
#                 position: absolute;
#                 top: 5px;
#                 right: 5px;
#                 background: #667eea;
#                 color: white;
#                 border: none;
#                 border-radius: 5px;
#                 padding: 5px 10px;
#                 font-size: 12px;
#                 cursor: pointer;
#                 opacity: 0.7;
#                 transition: opacity 0.3s ease;
#             }}
#
#             .subplot-enlarge:hover {{
#                 opacity: 1;
#             }}
#
#             .tooltip {{
#                 position: absolute;
#                 background: rgba(0, 0, 0, 0.8);
#                 color: white;
#                 padding: 5px 10px;
#                 border-radius: 5px;
#                 font-size: 12px;
#                 pointer-events: none;
#                 opacity: 0;
#                 transition: opacity 0.3s ease;
#                 z-index: 1001;
#             }}
#
#             .enlarge-button:hover + .tooltip {{
#                 opacity: 1;
#             }}
#         </style>
#     </head>
#     <body>
#         <div class="dashboard-header">
#             <h1>📊 Training Visualization Dashboard</h1>
#             <p>Interactive analysis of your model training metrics</p>
#         </div>
#
#         <div class="plot-container">
#             <button class="enlarge-button" onclick="openFullSize('training_interactive_full.html')" title="Open in new window">⤢</button>
#             <div class="tooltip">Click to enlarge in new window</div>
#             <div class="plot-title">📈 Interactive Training Metrics</div>
#             {fig1_html}
#         </div>
#
#         <div class="plot-container">
#             <button class="enlarge-button" onclick="openFullSize('training_detailed_full.html')" title="Open in new window">⤢</button>
#             <div class="tooltip">Click to enlarge in new window</div>
#             <div class="plot-title">📊 Detailed Analysis Grid</div>
#             {fig2_html}
#
#             <div class="subplot-grid">
#                 <div class="subplot-item">
#                     <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[0]}')">Enlarge</button>
#                     <strong>1. Overall Loss</strong>
#                 </div>
#                 <div class="subplot-item">
#                     <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[1]}')">Enlarge</button>
#                     <strong>2. Components</strong>
#                 </div>
#                 <div class="subplot-item">
#                     <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[2]}')">Enlarge</button>
#                     <strong>3. Train-Val Gap</strong>
#                 </div>
#                 <div class="subplot-item">
#                     <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[3]}')">Enlarge</button>
#                     <strong>4. Learning Dynamics</strong>
#                 </div>
#                 <div class="subplot-item">
#                     <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[4]}')">Enlarge</button>
#                     <strong>5. Loss Ratios</strong>
#                 </div>
#                 <div class="subplot-item">
#                     <button class="subplot-enlarge" onclick="openIndividualPlot('{individual_plots[5]}')">Enlarge</button>
#                     <strong>6. Convergence</strong>
#                 </div>
#             </div>
#         </div>
#
#         <script>
#             function openFullSize(filename) {{
#                 window.open(filename, '_blank', 'width=1400,height=900,scrollbars=yes,resizable=yes');
#             }}
#
#             function openIndividualPlot(filename) {{
#                 if (filename) {{
#                     window.open(filename, '_blank', 'width=1200,height=800,scrollbars=yes,resizable=yes');
#                 }}
#             }}
#
#             // Add hover effect for enlarge buttons
#             document.querySelectorAll('.enlarge-button').forEach(button => {{
#                 button.addEventListener('mouseenter', function() {{
#                     this.innerHTML = '🔍';
#                 }});
#                 button.addEventListener('mouseleave', function() {{
#                     this.innerHTML = '⤢';
#                 }});
#             }});
#         </script>
#     </body>
#     </html>
#     """
#
#     return dashboard_html
#
#
# # Keep the rest of the functions unchanged (calculate_statistics, create_statistics_table, visualize_training, and __main__)
# def calculate_statistics(metrics):
#     """Calculate detailed statistics for the training."""
#     stats = {}
#
#     if len(metrics['loss']) > 0:
#         # Basic statistics
#         stats['total_epochs'] = len(metrics['epoch'])
#         stats['final_train_loss'] = metrics['loss'][-1]
#         stats['final_val_loss'] = metrics['val_loss'][-1]
#         stats['best_val_loss'] = min(metrics['val_loss'])
#         stats['best_epoch'] = metrics['epoch'][metrics['val_loss'].index(min(metrics['val_loss']))]
#
#         # Convergence metrics
#         stats['train_loss_reduction'] = (metrics['loss'][0] - metrics['loss'][-1]) / metrics['loss'][0] * 100
#         stats['val_loss_reduction'] = (metrics['val_loss'][0] - metrics['val_loss'][-1]) / metrics['val_loss'][0] * 100
#
#         # Overfitting metrics
#         stats['final_overfit_gap'] = metrics['loss'][-1] - metrics['val_loss'][-1]
#         stats['avg_overfit_gap'] = np.mean([t - v for t, v in zip(metrics['loss'], metrics['val_loss'])])
#
#         # Stability metrics
#         if len(metrics['loss']) > 5:
#             last_5_train = metrics['loss'][-5:]
#             last_5_val = metrics['val_loss'][-5:]
#             stats['train_stability'] = np.std(last_5_train)
#             stats['val_stability'] = np.std(last_5_val)
#
#         # Learning rate stats
#         if 'learning_rate' in metrics and len(metrics['learning_rate']) > 0:
#             stats['initial_lr'] = metrics['learning_rate'][0]
#             stats['final_lr'] = metrics['learning_rate'][-1]
#             stats['lr_reduction'] = (stats['initial_lr'] - stats['final_lr']) / stats['initial_lr'] * 100
#
#     return stats
#
#
# def create_statistics_table(stats):
#     """Create an enhanced HTML table with training statistics."""
#
#     html = """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <meta charset="UTF-8">
#         <title>Training Statistics</title>
#         <style>
#             body {
#                 font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#                 padding: 40px;
#                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                 min-height: 100vh;
#                 margin: 0;
#             }
#
#             .container {
#                 max-width: 800px;
#                 margin: 0 auto;
#                 background: white;
#                 border-radius: 20px;
#                 box-shadow: 0 20px 60px rgba(0,0,0,0.3);
#                 overflow: hidden;
#             }
#
#             .header {
#                 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                 color: white;
#                 padding: 30px;
#                 text-align: center;
#             }
#
#             h1 {
#                 margin: 0;
#                 font-size: 2.5em;
#                 text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#             }
#
#             .subtitle {
#                 margin-top: 10px;
#                 opacity: 0.9;
#                 font-size: 1.1em;
#             }
#
#             table {
#                 width: 100%;
#                 border-collapse: collapse;
#             }
#
#             tr {
#                 transition: background-color 0.3s ease;
#             }
#
#             tr:hover {
#                 background-color: #f8f9fa;
#             }
#
#             td {
#                 padding: 20px 30px;
#                 border-bottom: 1px solid #e0e0e0;
#             }
#
#             .metric-name {
#                 font-weight: 600;
#                 color: #495057;
#                 font-size: 1.1em;
#             }
#
#             .metric-value {
#                 color: #212529;
#                 font-size: 1.1em;
#                 text-align: right;
#                 font-family: 'Courier New', monospace;
#             }
#
#             .good {
#                 color: #28a745;
#                 font-weight: bold;
#             }
#
#             .warning {
#                 color: #ffc107;
#                 font-weight: bold;
#             }
#
#             .bad {
#                 color: #dc3545;
#                 font-weight: bold;
#             }
#
#             .footer {
#                 padding: 20px;
#                 text-align: center;
#                 background: #f8f9fa;
#                 color: #6c757d;
#                 font-size: 0.9em;
#             }
#
#             .stat-icon {
#                 margin-right: 10px;
#                 font-size: 1.2em;
#             }
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <div class="header">
#                 <h1>📊 Training Statistics Summary</h1>
#                 <div class="subtitle">Comprehensive analysis of your model's training performance</div>
#             </div>
#             <table>
#     """
#
#     icons = {
#         'total_epochs': '📈',
#         'final_train_loss': '🎯',
#         'final_val_loss': '🎯',
#         'best_val_loss': '🏆',
#         'best_epoch': '⭐',
#         'train_loss_reduction': '📉',
#         'val_loss_reduction': '📉',
#         'final_overfit_gap': '⚖️',
#         'avg_overfit_gap': '⚖️',
#         'train_stability': '🔧',
#         'val_stability': '🔧',
#         'initial_lr': '🚀',
#         'final_lr': '🏁',
#         'lr_reduction': '🔻'
#     }
#
#     for key, value in stats.items():
#         display_name = key.replace('_', ' ').title()
#         icon = icons.get(key, '📌')
#
#         if isinstance(value, float):
#             formatted_value = f"{value:.4f}"
#
#             # Add color coding for certain metrics
#             if 'reduction' in key and value > 50:
#                 formatted_value = f'<span class="good">{formatted_value}%</span>'
#             elif 'overfit' in key and abs(value) > 0.1:
#                 formatted_value = f'<span class="warning">{formatted_value}</span>'
#             elif 'stability' in key and value > 0.1:
#                 formatted_value = f'<span class="bad">{formatted_value}</span>'
#         else:
#             formatted_value = str(value)
#
#         html += f"""
#             <tr>
#                 <td class="metric-name">
#                     <span class="stat-icon">{icon}</span>
#                     {display_name}
#                 </td>
#                 <td class="metric-value">{formatted_value}</td>
#             </tr>
#         """
#
#     html += """
#             </table>
#             <div class="footer">
#                 Generated with Training Visualizer | Use the dashboard for interactive exploration
#             </div>
#         </div>
#     </body>
#     </html>
#     """
#
#     return html
#
#
# def visualize_training(log_filepath):
#     """Main function to visualize training results with simultaneous windows."""
#     try:
#         metrics = parse_training_log(log_filepath)
#
#         if len(metrics['epoch']) == 0:
#             print("⚠️  No training data found in the log file!")
#             return False
#
#         print(f"✅ Found {len(metrics['epoch'])} epochs of training data")
#         print("\n🔄 Generating visualizations...")
#
#         # Create temp directory
#         temp_dir = tempfile.gettempdir()
#
#         # Create dashboard with enlargement functionality
#         dashboard_html = create_dashboard_with_enlargement(metrics, temp_dir)
#
#         # Calculate statistics
#         stats = calculate_statistics(metrics)
#         stats_html = create_statistics_table(stats)
#
#         # Save dashboard with UTF-8 encoding
#         dashboard_file = os.path.join(temp_dir, "training_dashboard.html")
#         with open(dashboard_file, 'w', encoding='utf-8') as f:
#             f.write(dashboard_html)
#
#         # Save statistics with UTF-8 encoding
#         stats_file = os.path.join(temp_dir, "training_statistics.html")
#         with open(stats_file, 'w', encoding='utf-8') as f:
#             f.write(stats_html)
#
#         # Open both in browser
#         print("\n📊 Opening visualizations in browser...")
#         webbrowser.open(f"file://{dashboard_file}", new=2)
#         webbrowser.open(f"file://{stats_file}", new=2)
#
#         print("\n✅ Visualizations opened in browser:")
#         print(f"  1. Dashboard (with enlargeable plots): {dashboard_file}")
#         print(f"  2. Statistics Summary: {stats_file}")
#         print("\n💡 Tip: Hover over any plot and click the ⤢ button to open it in a larger window!")
#         print("     Individual subplots also have their own enlarge buttons.")
#
#         # Print summary to console
#         print("\n" + "=" * 60)
#         print("TRAINING SUMMARY")
#         print("=" * 60)
#         for key, value in stats.items():
#             display_name = key.replace('_', ' ').title()
#             if isinstance(value, float):
#                 print(f"{display_name:.<30} {value:.4f}")
#             else:
#                 print(f"{display_name:.<30} {value}")
#         print("=" * 60)
#
#         return True
#
#     except Exception as e:
#         print(f"\n⚠️  Error during visualization: {e}")
#         import traceback
#         traceback.print_exc()
#         return False
#
#
# if __name__ == "__main__":
#     # Standalone mode - ask for file
#     log_file = input("Enter the path to your log file: ").strip()
#
#     # Remove quotes if present
#     log_file = log_file.strip('"').strip("'")
#
#     try:
#         if visualize_training(log_file):
#             print("\n" + "=" * 80)
#             print("✅ Visualization complete!")
#             print("=" * 80)
#             input("\nPress Enter to exit...")
#     except FileNotFoundError:
#         print(f"Error: Could not find file '{log_file}'")
#         input("\nPress Enter to exit...")
#
#
#
