"""Core visualization logic."""
import os
import tempfile
import webbrowser
from pathlib import Path

from ..parsers.log_parser import parse_training_log
from ..parsers.spec_parser import parse_model_specs
from ..analysis.statistics import calculate_statistics
from ..visualization.plots import (
    create_evaluation_metrics_plot,
    create_interactive_plot,
    create_detailed_subplots
)
from ..visualization.comparison_plots import (
    create_comparison_evaluation_plot,
    create_comparison_detailed_subplots,
    create_comparison_convergence_plot,
    create_comparison_learning_dynamics_plot
)
from ..visualization.html_components import (
    create_model_specs_card,
    create_statistics_table,
    create_comparison_statistics_table,
    create_evaluation_results_table
)

# Export these for use in PDF script
__all__ = [
    'parse_training_log',
    'parse_model_specs',
    'calculate_statistics',
    'create_evaluation_metrics_plot',
    'create_interactive_plot',
    'create_detailed_subplots',
    'create_comparison_evaluation_plot',
    'create_comparison_detailed_subplots',
    'create_comparison_convergence_plot',
    'create_comparison_learning_dynamics_plot',
    'visualize_training',
    'visualize_training_comparison',
    'create_dashboard_with_enlargement',
    'create_comparison_dashboard',
    'create_individual_plot_html'
]


def create_individual_plot_html(metrics, plot_type, plot_num, temp_dir):
    """Create HTML for individual plot with full-size version."""

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np

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
                                 marker_color='rgb(0, 0, 255)', opacity=0.7))
            fig.add_trace(go.Bar(x=epochs, y=val_loss_change_rate, name='Val Loss Change %',
                                 marker_color='rgb(255, 0, 0)', opacity=0.7))

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

    # Create individual plot files
    individual_plots = []
    for plot_type, num in [("overall_loss", 1), ("components", 2), ("gap", 3),
                           ("learning", 4), ("ratios", 5), ("convergence", 6)]:
        filename = create_individual_plot_html(metrics, plot_type, num, temp_dir)
        if filename:
            individual_plots.append(filename)
        else:
            individual_plots.append("")

    # Convert figures to HTML strings
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


def create_comparison_dashboard(datasets, temp_dir):
    """Create a comparison dashboard for multiple datasets."""

    # Create all comparison plots with results table
    eval_result = create_comparison_evaluation_plot(datasets, full_size=False)
    eval_fig, table_data, all_val_values, all_test_values = eval_result

    detailed_fig = create_comparison_detailed_subplots(datasets, full_size=False)
    convergence_fig = create_comparison_convergence_plot(datasets, full_size=False)
    learning_fig = create_comparison_learning_dynamics_plot(datasets, full_size=False)

    # Convert to HTML
    eval_html = eval_fig.to_html(include_plotlyjs='cdn', div_id="eval_compare")
    detailed_html = detailed_fig.to_html(include_plotlyjs='cdn', div_id="detailed_compare")
    convergence_html = convergence_fig.to_html(include_plotlyjs='cdn', div_id="convergence_compare")
    learning_html = learning_fig.to_html(include_plotlyjs='cdn', div_id="learning_compare")

    # Create evaluation results table
    eval_table_html = create_evaluation_results_table(table_data, all_val_values, all_test_values)

    # Create comparison statistics table
    stats_html = create_comparison_statistics_table(datasets)

    # Create summary cards
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
                Compare validation and test performance across all metrics. Lines are stacked vertically for better visibility.
            </div>
            {eval_html}
            {eval_table_html}
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


def visualize_training_comparison(dataset_configs):
    """Visualize multiple training logs for comparison."""

    datasets = []

    for config in dataset_configs:
        try:
            # Parse metrics and specs
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


def visualize_training(log_filepath=None, spec_filepath=None, compare_mode=False):
    """Main function to visualize training results."""

    if compare_mode or log_filepath is None:
        # Open dataset selector GUI
        from ..ui.dataset_selector import DatasetSelector
        selector = DatasetSelector()
        selector.select_datasets()

        if not selector.datasets:
            print("No datasets selected")
            return False

        return visualize_training_comparison(selector.datasets)

    else:
        # Single dataset mode
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

            # Create dashboard
            dashboard_html = create_dashboard_with_enlargement(metrics, eval_metrics, model_specs, temp_dir)

            # Calculate statistics
            stats = calculate_statistics(metrics, eval_metrics)
            stats_html = create_statistics_table(stats)

            # Save dashboard
            dashboard_file = os.path.join(temp_dir, "training_dashboard.html")
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                f.write(dashboard_html)

            # Save statistics
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
