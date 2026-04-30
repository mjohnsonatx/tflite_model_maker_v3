"""Single dataset visualization plots."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_evaluation_metrics_plot(eval_metrics, full_size=False):
    """Create a vertically stacked plot for AP/AR evaluation metrics with comparison table."""

    categories = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                  'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']

    val_values = [eval_metrics['validation'].get(cat, 0) for cat in categories]
    test_values = [eval_metrics['test_CLEAN'].get(cat, 0) for cat in categories]

    # Create subplots: 2 rows (one for each dataset), 1 column
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Validation Metrics', 'Test Metrics'),
        vertical_spacing=0.15,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]]
    )

    # Validation plot
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=val_values,
            mode='lines+markers',
            name='Validation',
            line=dict(color='rgb(31, 119, 180)', width=3 if full_size else 2),
            marker=dict(size=10 if full_size else 8, symbol='circle'),
            text=[f'{v:.3f}' for v in val_values],
            textposition='top center',
            textfont=dict(size=10 if full_size else 9)
        ),
        row=1, col=1
    )

    # Test plot
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=test_values,
            mode='lines+markers',
            name='Test',
            line=dict(color='rgb(255, 127, 14)', width=3 if full_size else 2),
            marker=dict(size=10 if full_size else 8, symbol='diamond'),
            text=[f'{v:.3f}' for v in test_values],
            textposition='top center',
            textfont=dict(size=10 if full_size else 9)
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title={
            'text': 'Model Evaluation Metrics (Validation vs Test)',
            'font': {'size': 24 if full_size else 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        height=900 if full_size else 800,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='rgba(240, 240, 240, 0.3)',
        paper_bgcolor='white',
        font=dict(size=12 if full_size else 11)
    )

    # Update axes
    fig.update_xaxes(tickangle=-45, row=1, col=1)
    fig.update_xaxes(tickangle=-45, row=2, col=1)
    fig.update_yaxes(
        range=[0, 1.05],
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=1, col=1
    )
    fig.update_yaxes(
        range=[0, 1.05],
        gridcolor='rgba(128, 128, 128, 0.2)',
        row=2, col=1
    )

    return fig


def create_interactive_plot(metrics, full_size=False):
    """Create highly detailed interactive plot with Plotly."""

    epochs = metrics['epoch']
    height = 900 if full_size else 700

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )

    plot_groups = {
        'Main Losses': [
            ('loss', 'Training Loss', 'rgb(31, 119, 180)', 'solid', True, 3),
            ('val_loss', 'Validation Loss', 'rgb(255,  127, 14)', 'solid', True, 3),
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
            ('gradient_norm', 'Gradient Norm', 'rgb(127, 127, 127)', 'solid', False, 2),
        ]
    }

    for group_name, group_configs in plot_groups.items():
        for metric_key, label, color, dash, use_primary_y, width in group_configs:
            if metric_key in metrics and len(metrics[metric_key]) > 0:
                visible = True if metric_key in ['loss', 'val_loss'] else 'legendonly'
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
                        legendgrouptitle_text=group_name if metric_key == group_configs[0][0] else None
                    ),
                    secondary_y=not use_primary_y
                )

    if len(metrics['val_loss']) > 0:
        best_epoch = metrics['epoch'][metrics['val_loss'].index(min(metrics['val_loss']))]
        best_val_loss = min(metrics['val_loss'])

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
        margin=dict(r=250)
    )

    fig.update_xaxes(rangeslider_visible=True)

    return fig


def create_detailed_subplots(metrics, full_size=False):
    """Create detailed subplot analysis with multiple perspectives."""

    epochs = metrics['epoch']
    height = 1400 if full_size else 1000
    title_size = 18 if full_size else 14
    line_width = 1.5 if full_size else 1  # Thinner lines

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Overall Loss Comparison',
            'Loss Components Breakdown',
            'Training vs Validation Gap',
            'Learning Dynamics',
            'Loss Change Rate %',
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

    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=title_size)

    # 1. Overall Loss Comparison
    if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['loss'], name='Training Loss',
                       line=dict(color='blue', width=line_width),
                       mode='lines+markers', marker=dict(size=4)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['val_loss'], name='Validation Loss',
                       line=dict(color='red', width=line_width),
                       mode='lines+markers', marker=dict(size=4)),
            row=1, col=1
        )

        if len(epochs) > 5:
            window = min(5, len(epochs) // 4)
            train_smooth = np.convolve(metrics['loss'], np.ones(window) / window, mode='valid_FINAL_CLEAN_DONISH')
            val_smooth = np.convolve(metrics['val_loss'], np.ones(window) / window, mode='valid_FINAL_CLEAN_DONISH')
            epochs_smooth = epochs[window // 2:-(window // 2) if window > 1 else None]

            fig.add_trace(
                go.Scatter(x=epochs_smooth, y=train_smooth, name='Training (Smoothed)',
                           line=dict(color='lightblue', width=line_width * 0.8, dash='dash'),
                           showlegend=True),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_smooth, y=val_smooth, name='Validation (Smoothed)',
                           line=dict(color='pink', width=line_width * 0.8, dash='dash'),
                           showlegend=True),
                row=1, col=1
            )

    # 2. Loss Components Breakdown
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
                       line=dict(color='orange', width=line_width)),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # 4. Learning Dynamics
    if len(metrics['learning_rate']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['learning_rate'], name='Learning Rate',
                       line=dict(color='cyan', width=line_width),
                       mode='lines'),  # Removed dashed lines
            row=2, col=2, secondary_y=False
        )
    if len(metrics['gradient_norm']) > 0:
        fig.add_trace(
            go.Scatter(x=epochs, y=metrics['gradient_norm'], name='Gradient Norm',
                       line=dict(color='gray', width=line_width)),
            row=2, col=2, secondary_y=True
        )

    # 5. Loss Change Rate % - with better contrast colors
    if len(metrics['loss']) > 1:
        loss_change_rate = [0] + [(metrics['loss'][i] - metrics['loss'][i - 1]) / metrics['loss'][i - 1] * 100
                                  if metrics['loss'][i - 1] != 0 else 0
                                  for i in range(1, len(metrics['loss']))]
        val_loss_change_rate = [0] + [
            (metrics['val_loss'][i] - metrics['val_loss'][i - 1]) / metrics['val_loss'][i - 1] * 100
            if metrics['val_loss'][i - 1] != 0 else 0
            for i in range(1, len(metrics['val_loss']))]

        # Using high-contrast colors
        fig.add_trace(
            go.Bar(x=epochs, y=loss_change_rate, name='Train Loss Change %',
                   marker_color='rgb(0, 0, 255)', opacity=0.7),  # Bright blue
            row=3, col=1
        )
        fig.add_trace(
            go.Bar(x=epochs, y=val_loss_change_rate, name='Val Loss Change %',
                   marker_color='rgb(255, 0, 0)', opacity=0.7),  # Bright red
            row=3, col=1
        )

    # 6. Convergence Analysis
    if len(metrics['loss']) > 0 and len(metrics['val_loss']) > 0:
        positive_loss = [l if l > 0 else None for l in metrics['loss']]
        positive_val_loss = [l if l > 0 else None for l in metrics['val_loss']]

        fig.add_trace(
            go.Scatter(x=epochs, y=positive_loss, name='Training (log)',
                       line=dict(color='blue', width=line_width),
                       connectgaps=False),
            row=3, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs, y=positive_val_loss, name='Validation (log)',
                       line=dict(color='red', width=line_width),
                       connectgaps=False),
            row=3, col=2
        )

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