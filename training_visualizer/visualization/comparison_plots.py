"""Comparison plots for multiple datasets."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_comparison_evaluation_plot(datasets, full_size=False):
    """Create vertically stacked evaluation metrics plots for multiple datasets."""

    num_datasets = len(datasets)
    if num_datasets == 1:
        from .plots import create_evaluation_metrics_plot
        return create_evaluation_metrics_plot(datasets[0]['eval_metrics'], full_size)

    # Create subplot for each dataset (stacked vertically)
    fig = make_subplots(
        rows=num_datasets, cols=1,
        subplot_titles=[f"<b>{d['name']}</b>" for d in datasets],
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"}] for _ in range(num_datasets)]
    )

    categories = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                  'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']

    colors_val = ['rgb(31, 119, 180)', 'rgb(44, 160, 44)', 'rgb(148, 103, 189)']
    colors_test = ['rgb(255, 127, 14)', 'rgb(214, 39, 40)', 'rgb(227, 119, 194)']

    for idx, dataset in enumerate(datasets, 1):
        eval_metrics = dataset['eval_metrics']

        val_values = [eval_metrics['validation'].get(cat, 0) for cat in categories]
        test_values = [eval_metrics['test_CLEAN'].get(cat, 0) for cat in categories]

        fig.add_trace(
            go.Scatter(
                name=f'Validation' if idx == 1 else f'Validation ({dataset["name"]})',
                x=categories,
                y=val_values,
                mode='lines+markers',
                line=dict(color=colors_val[idx - 1], width=4 if full_size else 3),
                marker=dict(size=12 if full_size else 10, symbol='circle'),
                text=[f'Val: {v:.3f}' for v in val_values],
                textposition='top center',
                textfont=dict(size=12 if full_size else 11),
                legendgroup=f'dataset{idx}',
                showlegend=True
            ),
            row=idx, col=1
        )

        fig.add_trace(
            go.Scatter(
                name=f'Test' if idx == 1 else f'Test ({dataset["name"]})',
                x=categories,
                y=test_values,
                mode='lines+markers',
                line=dict(color=colors_test[idx - 1], width=4 if full_size else 3, dash='dash'),
                marker=dict(size=12 if full_size else 10, symbol='diamond'),
                text=[f'Test: {v:.3f}' for v in test_values],
                textposition='bottom center',
                textfont=dict(size=12 if full_size else 11),
                legendgroup=f'dataset{idx}',
                showlegend=True
            ),
            row=idx, col=1
        )

    # Add comparison table below plots
    table_data = {
        'Metric': categories,
    }

    # Find best values for highlighting
    all_val_values = {}
    all_test_values = {}

    for dataset in datasets:
        val_key = f"{dataset['name']} (Val)"
        test_key = f"{dataset['name']} (Test)"
        table_data[val_key] = []
        table_data[test_key] = []

        for cat in categories:
            val_value = dataset['eval_metrics']['validation'].get(cat, 0)
            test_value = dataset['eval_metrics']['test_CLEAN'].get(cat, 0)
            table_data[val_key].append(f'{val_value:.3f}')
            table_data[test_key].append(f'{test_value:.3f}')

            if cat not in all_val_values:
                all_val_values[cat] = []
                all_test_values[cat] = []
            all_val_values[cat].append(val_value)
            all_test_values[cat].append(test_value)

    fig.update_layout(
        title={
            'text': 'Model Evaluation Metrics Comparison',
            'font': {'size': 28 if full_size else 24},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=300 * num_datasets + 200,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=14)
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.3)',
        paper_bgcolor='white',
        font=dict(size=13)
    )

    for i in range(1, num_datasets + 1):
        fig.update_xaxes(
            tickangle=-45,
            row=i, col=1,
            tickfont=dict(size=12)
        )
        fig.update_yaxes(
            range=[0, 1.1],
            row=i, col=1,
            tickfont=dict(size=12),
            gridcolor='rgba(128, 128, 128, 0.2)'
        )

    return fig, table_data, all_val_values, all_test_values


def create_comparison_detailed_subplots(datasets, full_size=False):
    """Create detailed subplot analysis for comparison."""

    num_datasets = len(datasets)
    height = 1600 if full_size else 1400
    title_size = 16 if full_size else 14
    line_width = 1.5 if full_size else 1  # Thinner lines

    row1_titles = [f"<b>{d['name']}</b><br>Loss Comparison" for d in datasets]
    row2_titles = [f"<b>{d['name']}</b><br>Components Breakdown" for d in datasets]
    row3_titles = [f"<b>{d['name']}</b><br>Train-Val Gap" for d in datasets]
    subplot_titles = row1_titles + row2_titles + row3_titles

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

        # Row 1: Loss Comparison
        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['loss'],
                    name=f'Training',
                    line=dict(color=colors['train'], width=line_width),
                    mode='lines+markers',
                    marker=dict(size=4),
                    legendgroup=f'dataset{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['val_loss'],
                    name=f'Validation',
                    line=dict(color=colors['val'], width=line_width),
                    mode='lines+markers',
                    marker=dict(size=4),
                    legendgroup=f'dataset{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

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

        # Row 2: Components Breakdown
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

        # Row 3: Train-Val Gap
        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            gaps = [t - v for t, v in zip(metrics['loss'], metrics['val_loss'])]
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

    # Update axes
    for i in range(1, num_datasets + 1):
        fig.update_xaxes(title_text="", row=1, col=i, tickfont=dict(size=11))
        fig.update_yaxes(title_text="Loss" if i == 1 else "", row=1, col=i, tickfont=dict(size=11))
        fig.update_xaxes(title_text="", row=2, col=i, tickfont=dict(size=11))
        fig.update_yaxes(title_text="Loss Components" if i == 1 else "", row=2, col=i, tickfont=dict(size=11))
        fig.update_xaxes(title_text="Epoch", row=3, col=i, tickfont=dict(size=11))
        fig.update_yaxes(title_text="Gap (Train - Val)" if i == 1 else "", row=3, col=i, tickfont=dict(size=11))

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

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=title_size)

    return fig


def create_comparison_convergence_plot(datasets, full_size=False):
    """Create convergence analysis plots for comparison."""

    num_datasets = len(datasets)
    line_width = 1.5 if full_size else 1  # Thinner lines

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

        # Row 1: Log scale convergence
        if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
            positive_loss = [l if l > 0 else None for l in metrics['loss']]
            positive_val_loss = [l if l > 0 else None for l in metrics['val_loss']]

            fig.add_trace(
                go.Scatter(
                    x=epochs, y=positive_loss,
                    name='Training (log)',
                    line=dict(color=colors['train'], width=line_width),
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
                    line=dict(color=colors['val'], width=line_width),
                    connectgaps=False,
                    legendgroup=f'log{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )

        # Row 2: Loss Change Rate with high contrast colors
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

            # High contrast colors
            fig.add_trace(
                go.Bar(
                    x=epochs, y=loss_change_rate,
                    name='Train Loss Change %',
                    marker_color='rgb(0, 0, 255)',  # Bright blue
                    opacity=0.7,
                    legendgroup=f'change{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )
            fig.add_trace(
                go.Bar(
                    x=epochs, y=val_loss_change_rate,
                    name='Val Loss Change %',
                    marker_color='rgb(255, 0, 0)',  # Bright red
                    opacity=0.7,
                    legendgroup=f'change{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )

            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=col)

    for i in range(1, num_datasets + 1):
        fig.update_yaxes(type="log", title_text="Loss (log scale)" if i == 1 else "", row=1, col=i)
        fig.update_xaxes(title_text="", row=1, col=i)
        fig.update_yaxes(title_text="Change %" if i == 1 else "", row=2, col=i)
        fig.update_xaxes(title_text="Epoch", row=2, col=i)

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

    return fig

def create_comparison_learning_dynamics_plot(datasets, full_size=False):
    """Create learning dynamics comparison."""

    num_datasets = len(datasets)
    line_width = 1.5 if full_size else 1  # Thinner lines

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

        # Row 1: Learning Rate (no dashed lines)
        if len(metrics['learning_rate']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['learning_rate'],
                    name='Learning Rate',
                    line=dict(color=color, width=line_width),
                    mode='lines+markers',
                    marker=dict(size=4),
                    legendgroup=f'lr{col}',
                    showlegend=(col == 1)
                ),
                row=1, col=col
            )
            # REMOVED: Vertical lines for LR drops

        # Row 2: Gradient Norm
        if len(metrics['gradient_norm']) > 0:
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=metrics['gradient_norm'],
                    name='Gradient Norm',
                    line=dict(color='gray', width=line_width),
                    mode='lines',
                    opacity=0.5,
                    legendgroup=f'grad{col}',
                    showlegend=(col == 1)
                ),
                row=2, col=col
            )

            # Add rolling average
            if len(metrics['gradient_norm']) > 5:
                window = min(5, len(metrics['gradient_norm']) // 4)
                grad_smooth = np.convolve(metrics['gradient_norm'],
                                          np.ones(window) / window, mode='valid')
                epochs_smooth = epochs[window // 2:-(window // 2) if window > 1 else None]

                fig.add_trace(
                    go.Scatter(
                        x=epochs_smooth, y=grad_smooth,
                        name='Gradient (Smoothed)',
                        line=dict(color='red', width=line_width),
                        legendgroup=f'grad_smooth{col}',
                        showlegend=(col == 1)
                    ),
                    row=2, col=col
                )

    for i in range(1, num_datasets + 1):
        fig.update_yaxes(title_text="Learning Rate" if i == 1 else "", row=1, col=i)
        fig.update_xaxes(title_text="", row=1, col=i)
        fig.update_yaxes(title_text="Gradient Norm" if i == 1 else "", row=2, col=i)
        fig.update_xaxes(title_text="Epoch", row=2, col=i)

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

    return fig