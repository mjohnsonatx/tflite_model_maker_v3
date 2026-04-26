"""Comparison table only functionality for comparing multiple models without plots."""
import os
import tempfile
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from ..parsers.log_parser import parse_training_log
from ..parsers.spec_parser import parse_model_specs
from ..analysis.statistics import calculate_statistics


def _check_model_completion(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Check if a single model has completed training (lightweight check).

    Args:
        config: Dictionary with 'path', 'name', and 'mtime' keys

    Returns:
        Dictionary with completion status or None if failed
    """
    try:
        # Quick check for evaluation metrics without full parsing
        with open(config['path'], 'r', encoding='utf-8') as f:
            content = f.read()

        # Look for validation or test_CLEAN metrics patterns
        has_validation = 'Validation metrics:' in content
        has_test = 'Test metrics:' in content

        return {
            'completed': has_validation or has_test,
            'name': config['name'],
            'path': config['path'],
            'mtime': config['mtime']
        }
    except Exception:
        return {
            'completed': False,
            'name': config['name'],
            'path': config['path'],
            'mtime': config['mtime']
        }


def _parse_single_model(config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse a single model configuration in a separate process.

    Args:
        config: Dictionary with 'path', 'name', and 'mtime' keys

    Returns:
        Dictionary with parsed model data or None if failed
    """
    try:
        # Parse metrics and specs
        metrics, eval_metrics = parse_training_log(config['path'])

        # Check if model finished training (has evaluation metrics)
        if not eval_metrics['validation'] and not eval_metrics['test_CLEAN']:
            return {
                'success': False,
                'name': config['name'],
                'error': 'No evaluation metrics (training incomplete)',
                'mtime': config['mtime']
            }

        # Try to find spec file
        log_dir = Path(config['path']).parent
        spec_filepath = log_dir / "training_specs.txt"
        model_specs = parse_model_specs(spec_filepath) if spec_filepath.exists() else {
            'main': {}, 'hyperparameters': {}
        }

        # Calculate statistics
        stats = calculate_statistics(metrics, eval_metrics)

        return {
            'success': True,
            'name': config['name'],
            'metrics': metrics,
            'eval_metrics': eval_metrics,
            'specs': model_specs,
            'stats': stats,
            'mtime': config['mtime']
        }

    except Exception as e:
        return {
            'success': False,
            'name': config['name'],
            'error': str(e),
            'mtime': config['mtime']
        }


class ComparisonTableOnly:
    """Class for generating table-only comparisons of multiple training models."""

    def __init__(self) -> None:
        """Initialize the comparison table generator."""
        self.datasets: List[Dict[str, Any]] = []
        self.failed_models: List[str] = []

    def load_models(self, model_configs: List[Dict[str, str]]) -> bool:
        """
        Load multiple models from their configurations using multiprocessing.

        Args:
            model_configs: List of dictionaries with 'path' and 'name' keys

        Returns:
            bool: True if at least one model was loaded successfully
        """
        self.datasets = []
        self.failed_models = []

        if not model_configs:
            print("❌ No model configurations provided")
            return False

        print(f"🔄 Loading {len(model_configs)} models using multiprocessing...")
        start_time = time.time()

        # Add mtime to configs if not present (for chronological ordering)
        configs_with_mtime = []
        for config in model_configs:
            if 'mtime' not in config:
                try:
                    log_path = Path(config['path'])
                    config['mtime'] = log_path.stat().st_mtime
                except:
                    config['mtime'] = 0
            configs_with_mtime.append(config)

        # Use multiprocessing to parse models in parallel
        max_workers = min(4, mp.cpu_count(), len(model_configs))
        print(f"📊 Using {max_workers} parallel processes...")

        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all parsing tasks
            future_to_config = {executor.submit(_parse_single_model, config): config
                                for config in configs_with_mtime}

            # Collect results as they complete
            for future in as_completed(future_to_config):
                result = future.result()
                if result:
                    results.append(result)

        # Sort results chronologically by modification time (newest first)
        results.sort(key=lambda x: x['mtime'], reverse=True)

        # Process results in chronological order
        for result in results:
            if result['success']:
                self.datasets.append({
                    'name': result['name'],
                    'metrics': result['metrics'],
                    'eval_metrics': result['eval_metrics'],
                    'specs': result['specs'],
                    'stats': result['stats']
                })
                print(f"✅ Loaded dataset: {result['name']}")
            else:
                self.failed_models.append(result['name'])
                print(f"⚠️ Skipping {result['name']}: {result['error']}")

        elapsed_time = time.time() - start_time
        print(f"⚡ Model loading completed in {elapsed_time:.2f} seconds")

        if not self.datasets:
            print("❌ No valid datasets loaded")
            return False

        return True

    def generate_comparison_html(self) -> str:
        """
        Generate HTML for comparison table only.

        Returns:
            str: HTML content for the comparison dashboard
        """
        # Create evaluation results table data
        table_data, all_val_values, all_test_values = self._create_table_data()

        # Create evaluation results table
        eval_table_html = self._create_evaluation_results_table(table_data, all_val_values, all_test_values)

        # Create comparison statistics table
        stats_html = self._create_comparison_statistics_table()

        # Create summary cards
        summary_cards_html = self._create_summary_cards()

        # Create failed models notification if any
        failed_models_html = ""
        if self.failed_models:
            failed_models_html = f"""
            <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px; 
                        padding: 15px; margin-bottom: 20px;">
                <h4 style="color: #856404; margin: 0 0 10px 0;">
                    ⚠️ Models Skipped Due to Incomplete Training:
                </h4>
                <p style="color: #856404; margin: 0; font-style: italic;">
                    {', '.join(self.failed_models)}
                </p>
            </div>
            """

        # Build dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Model Comparison Table</title>
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

                .container {{
                    max-width: 95%;
                    margin: 0 auto;
                }}

                .table-container {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    overflow: auto;
                }}

                .table-container:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 12px 35px rgba(0,0,0,0.25);
                }}

                .table-title {{
                    font-size: 1.8em;
                    font-weight: 600;
                    color: #333;
                    margin-bottom: 20px;
                    padding-bottom: 12px;
                    border-bottom: 3px solid #667eea;
                    display: flex;
                    align-items: center;
                }}

                .table-description {{
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

                /* Horizontal scrolling for tables */
                .table-scroll {{
                    overflow-x: auto;
                    margin: 20px 0;
                }}

                .wide-table {{
                    width: max-content;
                    min-width: 100%;
                }}

                /* Make tables more readable */
                table {{
                    border-collapse: collapse;
                    font-size: 1.05em;
                }}

                th, td {{
                    padding: 12px 15px;
                    text-align: center;
                    border-bottom: 1px solid #e0e0e0;
                    white-space: nowrap;
                }}

                th {{
                    background: #f8f9fa;
                    font-weight: 600;
                    border-bottom: 2px solid #667eea;
                }}

                th:first-child {{
                    text-align: left;
                    position: sticky;
                    left: 0;
                    background: #f8f9fa;
                    z-index: 10;
                }}

                td:first-child {{
                    text-align: left;
                    position: sticky;
                    left: 0;
                    background: white;
                    z-index: 9;
                    font-weight: 600;
                    color: #495057;
                }}

                .highlight-best {{
                    background-color: rgba(40, 167, 69, 0.25) !important;
                    font-weight: bold !important;
                    color: #28a745 !important;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="dashboard-header">
                    <h1>📊 Model Comparison Table</h1>
                    <p>Detailed specifications and evaluation results for {len(self.datasets)} models</p>
                </div>

                {failed_models_html}

                {summary_cards_html}

                {stats_html}

                <div class="section-divider"></div>

                <div class="table-container">
                    <div class="table-title">🎯 Evaluation Metrics Comparison</div>
                    <div class="table-description">
                        Compare AP/AR performance metrics across all models. Best scores are highlighted in green.
                    </div>
                    {eval_table_html}
                </div>

                <div style="text-align: center; color: white; padding: 30px; opacity: 0.9;">
                    <p style="font-size: 1.1em;">💡 Tip: Scroll horizontally to see all model comparisons</p>
                </div>
            </div>
        </body>
        </html>
        """

        return dashboard_html

    def _create_table_data(self) -> Tuple[Dict[str, List[str]], List[float], List[float]]:
        """Create table data for evaluation metrics comparison."""

        # Create mapping from DetectionBoxes names to short names
        metric_mapping = {
            'DetectionBoxes_Precision/mAP': 'AP',
            'DetectionBoxes_Precision/mAP@.50IOU': 'AP50',
            'DetectionBoxes_Precision/mAP@.75IOU': 'AP75',
            'DetectionBoxes_Precision/mAP (small)': 'APs',
            'DetectionBoxes_Precision/mAP (medium)': 'APm',
            'DetectionBoxes_Precision/mAP (large)': 'APl',
            'DetectionBoxes_Recall/AR@1': 'ARmax1',
            'DetectionBoxes_Recall/AR@10': 'ARmax10',
            'DetectionBoxes_Recall/AR@100': 'ARmax100',
            'DetectionBoxes_Recall/AR@100 (small)': 'ARs',
            'DetectionBoxes_Recall/AR@100 (medium)': 'ARm',
            'DetectionBoxes_Recall/AR@100 (large)': 'ARl'
        }

        # Use the same categories as the working comparison plots
        categories = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
                      'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']

        table_data = {'Metric': []}
        all_val_values = []
        all_test_values = []

        # Add column headers - SEPARATE Val and Test columns for each dataset (back to original format)
        for dataset in self.datasets:
            table_data[f"{dataset['name']} (Val)"] = []
            table_data[f"{dataset['name']} (Test)"] = []

        # Add each metric as a row
        for category in categories:
            table_data['Metric'].append(category)

            for dataset in self.datasets:
                val_data = dataset['eval_metrics']['validation']
                test_data = dataset['eval_metrics']['test_CLEAN']

                # Try to find the value using both short name and mapped long name
                val_value = val_data.get(category)
                test_value = test_data.get(category)

                # If not found, try the reverse mapping (long name)
                if val_value is None:
                    for long_name, short_name in metric_mapping.items():
                        if short_name == category and long_name in val_data:
                            val_value = val_data[long_name]
                            break

                if test_value is None:
                    for long_name, short_name in metric_mapping.items():
                        if short_name == category and long_name in test_data:
                            test_value = test_data[long_name]
                            break

                # Add validation value
                if val_value is not None:
                    table_data[f"{dataset['name']} (Val)"].append(f"{val_value:.3f}")
                    all_val_values.append(val_value)
                else:
                    table_data[f"{dataset['name']} (Val)"].append("N/A")

                # Add test_CLEAN value
                if test_value is not None:
                    table_data[f"{dataset['name']} (Test)"].append(f"{test_value:.3f}")
                    all_test_values.append(test_value)
                else:
                    table_data[f"{dataset['name']} (Test)"].append("N/A")

        return table_data, all_val_values, all_test_values

    def _create_evaluation_results_table(self, table_data: Dict[str, List[str]],
                                         all_val_values: List[float],
                                         all_test_values: List[float]) -> str:
        """Create HTML table for evaluation results with best score highlighting."""
        html = """
        <div class="table-scroll">
            <table class="wide-table">
                <thead>
                    <tr>
        """

        # Add headers
        for key in table_data.keys():
            html += f'<th>{key}</th>'

        html += """
                    </tr>
                </thead>
                <tbody>
        """

        # Add rows
        for i, metric in enumerate(table_data['Metric']):
            html += '<tr>'

            # Metric name
            html += f'<td>{metric}</td>'

            # Collect all values for this row to find the best
            row_values = []
            row_cells = []

            for key in table_data.keys():
                if key != 'Metric':
                    value_str = table_data[key][i]
                    if value_str and value_str != "N/A" and value_str.strip():
                        try:
                            row_values.append(float(value_str))
                            row_cells.append((key, value_str, float(value_str)))
                        except:
                            row_cells.append((key, value_str, None))
                    else:
                        row_cells.append((key, value_str, None))

            # Find best value (highest) - highlight all ties for the best value
            valid_values = [v for v in row_values if v is not None]
            best_value = max(valid_values) if valid_values else None

            # Add value cells
            for key, value_str, value_float in row_cells:
                css_class = ""
                # Highlight if this value equals the best value (including ties)
                if best_value is not None and value_float is not None and value_float == best_value:
                    css_class = "highlight-best"

                html += f'<td class="{css_class}">{value_str}</td>'

            html += '</tr>'

        html += """
                </tbody>
            </table>
        </div>
        <div style="margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 5px;">
            <p style="margin: 0; color: #6c757d; font-size: 0.95em;">
                <strong>Note:</strong> 
                <span style="background-color: rgba(40, 167, 69, 0.25); padding: 2px 8px; border-radius: 3px; color: #28a745; font-weight: bold;">Highlighted values</span>
                indicate the best performance for each metric
            </p>
        </div>
        """

        return html

    def _create_comparison_statistics_table(self) -> str:
        """Create comparison statistics table with hyperparameters."""
        html = """
        <div class="table-container">
            <div class="table-title">📊 Performance & Configuration Comparison</div>
            <div class="table-description">
                Training performance metrics and detailed hyperparameter comparison. Best values are highlighted in green.
                Different colors indicate different parameter values for easy A/B analysis.
            </div>

            <!-- Metrics Comparison Table - BACK TO VERTICAL LAYOUT -->
            <h3 style="color: #495057; margin-top: 20px; margin-bottom: 15px; font-size: 1.4em;">📈 Training & Evaluation Metrics</h3>

            <div class="table-scroll">
                <table class="wide-table">
                    <thead>
                        <tr>
                            <th>Metric</th>
        """

        # Add column headers
        for dataset in self.datasets:
            html += f'<th>{dataset["name"]}</th>'

        html += """
                        </tr>
                    </thead>
                    <tbody>
        """

        comparison_metrics = [
            ('Total Epochs', 'total_epochs', '{:.0f}'),
            ('Best Val Loss', 'best_val_loss', '{:.4f}'),
            ('Best Epoch', 'best_epoch', '{:.0f}'),
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
            html += f'<tr><td>{metric_name}</td>'

            values = []
            for dataset in self.datasets:
                if metric_key in dataset['stats']:
                    value = dataset['stats'][metric_key]
                    values.append(value)
                else:
                    values.append(None)

            # Determine best value
            if values and any(v is not None for v in values):
                valid_values = [v for v in values if v is not None]
                if 'loss' in metric_key.lower() and 'reduction' not in metric_key and 'best_epoch' not in metric_key:
                    best_val = min(valid_values) if valid_values else None
                else:
                    best_val = max(valid_values) if valid_values else None
            else:
                best_val = None

            # Add cells with highlighting
            for i, dataset in enumerate(self.datasets):
                css_class = ""
                if metric_key in dataset['stats'] and values[i] is not None:
                    value = dataset['stats'][metric_key]
                    formatted_value = format_str.format(value)

                    if best_val is not None and len(set(valid_values)) > 1:
                        if value == best_val:
                            css_class = "highlight-best"

                else:
                    formatted_value = 'N/A'

                html += f'<td class="{css_class}">{formatted_value}</td>'

            html += '</tr>'

        html += """
                    </tbody>
                </table>
            </div>

            <!-- Hyperparameters Comparison -->
            <h3 style="color: #495057; margin-top: 30px; margin-bottom: 15px; font-size: 1.4em;">⚙️ Hyperparameters Comparison</h3>
            <div class="table-scroll">
                <table class="wide-table">
                    <thead>
                        <tr>
                            <th>Parameter</th>
        """

        for dataset in self.datasets:
            html += f'<th>{dataset["name"]}</th>'

        html += """
                        </tr>
                    </thead>
                    <tbody>
        """

        # Collect all unique hyperparameters
        all_hyperparams = set()
        for dataset in self.datasets:
            all_hyperparams.update(dataset['specs']['hyperparameters'].keys())

        hyperparam_groups = [
            ('Model Architecture', ['model_name', 'architecture', 'backbone_name', 'num_classes', 'image_size']),
            ('Training Settings', ['batch_size', 'epochs', 'train_whole_model']),
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
            ('Other Training Settings', ['skip_crowd_during_training', 'max_instances_per_image',
                                         'mixed_precision', 'drop_remainder', 'map_freq'])
        ]

        # Create mapping for consistent colors
        param_color_map = {}
        color_index = 0
        colors = ['#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6610f2', '#e83e8c']

        for group_name, params in hyperparam_groups:
            # First check if any of the main specs should be included
            main_params = []
            if group_name == 'Model Architecture':
                for dataset in self.datasets:
                    main_params.extend(dataset['specs']['main'].keys())
            elif group_name == 'Training Settings':
                for dataset in self.datasets:
                    main_params.extend(dataset['specs']['main'].keys())

            group_params = [p for p in params if p in all_hyperparams or p in main_params]
            if not group_params:
                continue

            html += f"""
                <tr style="background: #e9ecef;">
                    <td colspan="{len(self.datasets) + 1}" style="padding: 10px; font-weight: bold; 
                        color: #667eea; font-size: 1.05em; border-top: 1px solid #dee2e6;">
                        {group_name}
                    </td>
                </tr>
            """

            for param in group_params:
                param_label = param.replace('_', ' ').title()
                html += f'<tr><td>{param_label}</td>'

                # Collect values for this parameter across datasets
                param_values = []
                for dataset in self.datasets:
                    # Check both main specs and hyperparameters
                    value = dataset['specs']['hyperparameters'].get(param)
                    if value is None:
                        value = dataset['specs']['main'].get(param, 'N/A')
                    param_values.append(str(value))

                # Check if values differ
                unique_values = set(v for v in param_values if v != 'N/A')
                values_differ = len(unique_values) > 1

                # Assign colors for different values
                if values_differ:
                    for val in unique_values:
                        if val not in param_color_map:
                            param_color_map[val] = colors[color_index % len(colors)]
                            color_index += 1

                # Add cells
                for value in param_values:
                    style = ""
                    if value != 'N/A':
                        if values_differ:
                            # Different values - assign consistent colors
                            bg_color = param_color_map[value] + "20"  # 20% opacity
                            text_color = param_color_map[value]
                            style = f'style="background-color: {bg_color}; color: {text_color}; font-weight: bold;"'
                        else:
                            # Same values - keep consistent color for booleans
                            if value in ['True', 'False']:
                                color = '#28a745' if value == 'True' else '#dc3545'
                                style = f'style="color: {color}; font-weight: bold;"'

                    html += f'<td {style}>{value}</td>'

                html += '</tr>'

        html += """
                    </tbody>
                </table>
            </div>

            <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
                    <strong>Legend:</strong> 
                    <span style="background-color: rgba(40, 167, 69, 0.2); padding: 2px 6px; border-radius: 3px;">Best value</span>
                    <span style="margin-left: 10px;">Different colors indicate different values for the same parameter</span>
                </p>
            </div>
        </div>
        """

        return html

    def _create_summary_cards(self) -> str:
        """Create summary cards for each model."""
        html = """
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
        """

        colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']

        for idx, dataset in enumerate(self.datasets):
            color = colors[idx % len(colors)]
            val_ap = dataset['stats'].get('val_ap', 0)
            test_ap = dataset['stats'].get('test_ap', 0)
            best_loss = dataset['stats'].get('best_val_loss', 0)
            best_epoch = dataset['stats'].get('best_epoch', 0)

            html += f"""
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

        html += "</div>"
        return html

    def visualize_comparison_table_only(self, model_configs: List[Dict[str, str]]) -> bool:
        """
        Generate and display comparison table only.

        Args:
            model_configs: List of dictionaries with 'path' and 'name' keys

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.load_models(model_configs):
            return False

        print(f"\n🔄 Generating comparison table for {len(self.datasets)} model(s)...")

        # Generate HTML
        dashboard_html = self.generate_comparison_html()

        # Save and open dashboard
        temp_dir = tempfile.gettempdir()
        dashboard_file = os.path.join(temp_dir, "model_comparison_table.html")

        with open(dashboard_file, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)

        print("\n📊 Opening comparison table in browser...")
        webbrowser.open(f"file://{dashboard_file}", new=2)

        print(f"\n✅ Comparison table opened: {dashboard_file}")
        return True

    @staticmethod
    def find_recent_models(base_path: str, limit: int = 10) -> List[Dict[str, str]]:
        """
        Find the most recent N completed models in a directory structure.
        Only returns models that have evaluation metrics (completed training).

        Args:
            base_path: Path to the base directory containing model folders
            limit: Maximum number of completed models to return

        Returns:
            List of model configurations sorted by modification time (newest first)

        Example directory structure:
            models/
            ├── efficientdet_lite0_2025-10-09_18-18-49/
            │   ├── training_log.txt
            │   └── training_specs.txt
            ├── efficientdet_lite0_2025-10-09_18-35-35/
            └── mobilenet_v2_2025-10-31_10-24-58/
        """
        base_dir = Path(base_path)
        if not base_dir.exists():
            print(f"❌ Base path does not exist: {base_path}")
            return []

        print(f"🔍 Scanning directory for models...")
        all_model_configs = []

        # Look for model directories with training_log.txt
        for model_dir in base_dir.iterdir():
            if model_dir.is_dir():
                log_file = model_dir / "training_log.txt"
                if log_file.exists():
                    # Use the parent folder name as the model name
                    model_name = model_dir.name
                    all_model_configs.append({
                        'path': str(log_file),
                        'name': model_name,
                        'mtime': log_file.stat().st_mtime
                    })

        # Sort by modification time (newest first)
        all_model_configs.sort(key=lambda x: x['mtime'], reverse=True)
        print(f"📁 Found {len(all_model_configs)} total models with training logs")

        if not all_model_configs:
            print("❌ No models with training logs found in directory")
            return []

        # Use multiprocessing to quickly check which models are completed
        print(f"🔄 Checking completion status using multiprocessing...")
        start_time = time.time()

        # Create lightweight configs for checking completion
        check_configs = [{'path': config['path'], 'name': config['name'], 'mtime': config['mtime']}
                         for config in all_model_configs]

        max_workers = min(4, mp.cpu_count(), len(check_configs))
        completed_models = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit completion check tasks
            future_to_config = {executor.submit(_check_model_completion, config): config
                                for config in check_configs}

            # Collect results as they complete
            for future in as_completed(future_to_config):
                result = future.result()
                if result and result['completed']:
                    completed_models.append(result)

                    # Stop when we have enough completed models
                    if len(completed_models) >= limit:
                        # Cancel remaining futures
                        for f in future_to_config:
                            f.cancel()
                        break

        # Sort completed models chronologically (newest first)
        completed_models.sort(key=lambda x: x['mtime'], reverse=True)
        completed_models = completed_models[:limit]

        # Remove mtime from the final result
        for config in completed_models:
            del config['mtime']

        elapsed_time = time.time() - start_time
        print(f"⚡ Completion check completed in {elapsed_time:.2f} seconds")
        print(f"📊 Found {len(completed_models)} completed models out of {len(all_model_configs)} total models")

        return completed_models

    @staticmethod
    def get_available_model_count(base_path: str) -> int:
        """
        Get the total number of available completed models in a directory.

        Args:
            base_path: Path to the base directory containing model folders

        Returns:
            int: Number of completed models available
        """
        base_dir = Path(base_path)
        if not base_dir.exists():
            return 0

        # Quick count of models with training logs
        model_count = 0
        for model_dir in base_dir.iterdir():
            if model_dir.is_dir():
                log_file = model_dir / "training_log.txt"
                if log_file.exists():
                    model_count += 1

        return model_count

# """Comparison table only functionality for comparing multiple models without plots."""
# import os
# import tempfile
# import webbrowser
# from pathlib import Path
# from typing import List, Dict, Any, Optional, Tuple
# import re
#
# from ..parsers.log_parser import parse_training_log
# from ..parsers.spec_parser import parse_model_specs
# from ..analysis.statistics import calculate_statistics
#
#
# class ComparisonTableOnly:
#     """Class for generating table-only comparisons of multiple training models."""
#
#     def __init__(self) -> None:
#         """Initialize the comparison table generator."""
#         self.datasets: List[Dict[str, Any]] = []
#         self.failed_models: List[str] = []
#
#     def load_models(self, model_configs: List[Dict[str, str]]) -> bool:
#         """
#         Load multiple models from their configurations.
#
#         Args:
#             model_configs: List of dictionaries with 'path' and 'name' keys
#
#         Returns:
#             bool: True if at least one model was loaded successfully
#         """
#         self.datasets = []
#         self.failed_models = []
#
#         for config in model_configs:
#             try:
#                 # Parse metrics and specs
#                 metrics, eval_metrics = parse_training_log(config['path'])
#
#                 # Check if model finished training (has evaluation metrics)
#                 if not eval_metrics['validation'] and not eval_metrics['test_CLEAN']:
#                     self.failed_models.append(config['name'])
#                     print(f"⚠️ Skipping {config['name']}: No evaluation metrics (training incomplete)")
#                     continue
#
#                 # Try to find spec file
#                 log_dir = Path(config['path']).parent
#                 spec_filepath = log_dir / "training_specs.txt"
#                 model_specs = parse_model_specs(spec_filepath) if spec_filepath.exists() else {
#                     'main': {}, 'hyperparameters': {}
#                 }
#
#                 # Calculate statistics
#                 stats = calculate_statistics(metrics, eval_metrics)
#
#                 self.datasets.append({
#                     'name': config['name'],
#                     'metrics': metrics,
#                     'eval_metrics': eval_metrics,
#                     'specs': model_specs,
#                     'stats': stats
#                 })
#
#                 print(f"✅ Loaded dataset: {config['name']}")
#
#             except Exception as e:
#                 self.failed_models.append(config['name'])
#                 print(f"⚠️ Error loading {config['name']}: {e}")
#
#         if not self.datasets:
#             print("❌ No valid datasets loaded")
#             return False
#
#         return True
#
#     def generate_comparison_html(self) -> str:
#         """
#         Generate HTML for comparison table only.
#
#         Returns:
#             str: HTML content for the comparison dashboard
#         """
#         # Create evaluation results table data
#         table_data, all_val_values, all_test_values = self._create_table_data()
#
#         # Create evaluation results table
#         eval_table_html = self._create_evaluation_results_table(table_data, all_val_values, all_test_values)
#
#         # Create comparison statistics table
#         stats_html = self._create_comparison_statistics_table()
#
#         # Create summary cards
#         summary_cards_html = self._create_summary_cards()
#
#         # Create failed models notification if any
#         failed_models_html = ""
#         if self.failed_models:
#             failed_models_html = f"""
#             <div style="background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px;
#                         padding: 15px; margin-bottom: 20px;">
#                 <h4 style="color: #856404; margin: 0 0 10px 0;">
#                     ⚠️ Models Skipped Due to Incomplete Training:
#                 </h4>
#                 <p style="color: #856404; margin: 0; font-style: italic;">
#                     {', '.join(self.failed_models)}
#                 </p>
#             </div>
#             """
#
#         # Build dashboard HTML
#         dashboard_html = f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <meta charset="UTF-8">
#             <title>Model Comparison Table</title>
#             <style>
#                 body {{
#                     font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
#                     margin: 0;
#                     padding: 20px;
#                     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#                     min-height: 100vh;
#                 }}
#
#                 .dashboard-header {{
#                     text-align: center;
#                     color: white;
#                     margin-bottom: 40px;
#                 }}
#
#                 .dashboard-header h1 {{
#                     font-size: 3em;
#                     margin: 0;
#                     text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
#                     font-weight: 600;
#                 }}
#
#                 .dashboard-header p {{
#                     font-size: 1.2em;
#                     margin-top: 10px;
#                     opacity: 0.95;
#                 }}
#
#                 .container {{
#                     max-width: 95%;
#                     margin: 0 auto;
#                 }}
#
#                 .table-container {{
#                     background: white;
#                     border-radius: 15px;
#                     padding: 25px;
#                     margin-bottom: 30px;
#                     box-shadow: 0 10px 30px rgba(0,0,0,0.2);
#                     overflow: auto;
#                 }}
#
#                 .table-container:hover {{
#                     transform: translateY(-2px);
#                     box-shadow: 0 12px 35px rgba(0,0,0,0.25);
#                 }}
#
#                 .table-title {{
#                     font-size: 1.8em;
#                     font-weight: 600;
#                     color: #333;
#                     margin-bottom: 20px;
#                     padding-bottom: 12px;
#                     border-bottom: 3px solid #667eea;
#                     display: flex;
#                     align-items: center;
#                 }}
#
#                 .table-description {{
#                     font-size: 0.95em;
#                     color: #6c757d;
#                     margin-bottom: 20px;
#                     line-height: 1.5;
#                 }}
#
#                 .section-divider {{
#                     height: 2px;
#                     background: linear-gradient(90deg, transparent, #667eea, transparent);
#                     margin: 40px 0;
#                 }}
#
#                 /* Horizontal scrolling for tables */
#                 .table-scroll {{
#                     overflow-x: auto;
#                     margin: 20px 0;
#                 }}
#
#                 .wide-table {{
#                     width: max-content;
#                     min-width: 100%;
#                 }}
#
#                 /* Make tables more readable */
#                 table {{
#                     border-collapse: collapse;
#                     font-size: 1.05em;
#                 }}
#
#                 th, td {{
#                     padding: 12px 15px;
#                     text-align: center;
#                     border-bottom: 1px solid #e0e0e0;
#                     white-space: nowrap;
#                 }}
#
#                 th {{
#                     background: #f8f9fa;
#                     font-weight: 600;
#                     border-bottom: 2px solid #667eea;
#                 }}
#
#                 th:first-child {{
#                     text-align: left;
#                     position: sticky;
#                     left: 0;
#                     background: #f8f9fa;
#                     z-index: 10;
#                 }}
#
#                 td:first-child {{
#                     text-align: left;
#                     position: sticky;
#                     left: 0;
#                     background: white;
#                     z-index: 9;
#                     font-weight: 600;
#                     color: #495057;
#                 }}
#
#                 .highlight-best {{
#                     background-color: rgba(40, 167, 69, 0.25) !important;
#                     font-weight: bold !important;
#                     color: #28a745 !important;
#                 }}
#             </style>
#         </head>
#         <body>
#             <div class="container">
#                 <div class="dashboard-header">
#                     <h1>📊 Model Comparison Table</h1>
#                     <p>Detailed specifications and evaluation results for {len(self.datasets)} models</p>
#                 </div>
#
#                 {failed_models_html}
#
#                 {summary_cards_html}
#
#                 {stats_html}
#
#                 <div class="section-divider"></div>
#
#                 <div class="table-container">
#                     <div class="table-title">🎯 Evaluation Metrics Comparison</div>
#                     <div class="table-description">
#                         Compare AP/AR performance metrics across all models. Best scores are highlighted in green.
#                     </div>
#                     {eval_table_html}
#                 </div>
#
#                 <div style="text-align: center; color: white; padding: 30px; opacity: 0.9;">
#                     <p style="font-size: 1.1em;">💡 Tip: Scroll horizontally to see all model comparisons</p>
#                 </div>
#             </div>
#         </body>
#         </html>
#         """
#
#         return dashboard_html
#
#     def _create_table_data(self) -> Tuple[Dict[str, List[str]], List[float], List[float]]:
#         """Create table data for evaluation metrics comparison."""
#
#         # Create mapping from DetectionBoxes names to short names
#         metric_mapping = {
#             'DetectionBoxes_Precision/mAP': 'AP',
#             'DetectionBoxes_Precision/mAP@.50IOU': 'AP50',
#             'DetectionBoxes_Precision/mAP@.75IOU': 'AP75',
#             'DetectionBoxes_Precision/mAP (small)': 'APs',
#             'DetectionBoxes_Precision/mAP (medium)': 'APm',
#             'DetectionBoxes_Precision/mAP (large)': 'APl',
#             'DetectionBoxes_Recall/AR@1': 'ARmax1',
#             'DetectionBoxes_Recall/AR@10': 'ARmax10',
#             'DetectionBoxes_Recall/AR@100': 'ARmax100',
#             'DetectionBoxes_Recall/AR@100 (small)': 'ARs',
#             'DetectionBoxes_Recall/AR@100 (medium)': 'ARm',
#             'DetectionBoxes_Recall/AR@100 (large)': 'ARl'
#         }
#
#         # Use the same categories as the working comparison plots
#         categories = ['AP', 'AP50', 'AP75', 'APs', 'APm', 'APl',
#                       'ARmax1', 'ARmax10', 'ARmax100', 'ARs', 'ARm', 'ARl']
#
#         table_data = {'Metric': []}
#         all_val_values = []
#         all_test_values = []
#
#         # Add column headers - SEPARATE Val and Test columns for each dataset (back to original format)
#         for dataset in self.datasets:
#             table_data[f"{dataset['name']} (Val)"] = []
#             table_data[f"{dataset['name']} (Test)"] = []
#
#         # Add each metric as a row
#         for category in categories:
#             table_data['Metric'].append(category)
#
#             for dataset in self.datasets:
#                 val_data = dataset['eval_metrics']['validation']
#                 test_data = dataset['eval_metrics']['test_CLEAN']
#
#                 # Try to find the value using both short name and mapped long name
#                 val_value = val_data.get(category)
#                 test_value = test_data.get(category)
#
#                 # If not found, try the reverse mapping (long name)
#                 if val_value is None:
#                     for long_name, short_name in metric_mapping.items():
#                         if short_name == category and long_name in val_data:
#                             val_value = val_data[long_name]
#                             break
#
#                 if test_value is None:
#                     for long_name, short_name in metric_mapping.items():
#                         if short_name == category and long_name in test_data:
#                             test_value = test_data[long_name]
#                             break
#
#                 # Add validation value
#                 if val_value is not None:
#                     table_data[f"{dataset['name']} (Val)"].append(f"{val_value:.3f}")
#                     all_val_values.append(val_value)
#                 else:
#                     table_data[f"{dataset['name']} (Val)"].append("N/A")
#
#                 # Add test_CLEAN value
#                 if test_value is not None:
#                     table_data[f"{dataset['name']} (Test)"].append(f"{test_value:.3f}")
#                     all_test_values.append(test_value)
#                 else:
#                     table_data[f"{dataset['name']} (Test)"].append("N/A")
#
#         return table_data, all_val_values, all_test_values
#
#     def _create_evaluation_results_table(self, table_data: Dict[str, List[str]],
#                                          all_val_values: List[float],
#                                          all_test_values: List[float]) -> str:
#         """Create HTML table for evaluation results with best score highlighting."""
#         html = """
#         <div class="table-scroll">
#             <table class="wide-table">
#                 <thead>
#                     <tr>
#         """
#
#         # Add headers
#         for key in table_data.keys():
#             html += f'<th>{key}</th>'
#
#         html += """
#                     </tr>
#                 </thead>
#                 <tbody>
#         """
#
#         # Add rows
#         for i, metric in enumerate(table_data['Metric']):
#             html += '<tr>'
#
#             # Metric name
#             html += f'<td>{metric}</td>'
#
#             # Collect all values for this row to find the best
#             row_values = []
#             row_cells = []
#
#             for key in table_data.keys():
#                 if key != 'Metric':
#                     value_str = table_data[key][i]
#                     if value_str and value_str != "N/A" and value_str.strip():
#                         try:
#                             row_values.append(float(value_str))
#                             row_cells.append((key, value_str, float(value_str)))
#                         except:
#                             row_cells.append((key, value_str, None))
#                     else:
#                         row_cells.append((key, value_str, None))
#
#             # Find best value (highest) - highlight all ties for the best value
#             valid_values = [v for v in row_values if v is not None]
#             best_value = max(valid_values) if valid_values else None
#
#             # Add value cells
#             for key, value_str, value_float in row_cells:
#                 css_class = ""
#                 # Highlight if this value equals the best value (including ties)
#                 if best_value is not None and value_float is not None and value_float == best_value:
#                     css_class = "highlight-best"
#
#                 html += f'<td class="{css_class}">{value_str}</td>'
#
#             html += '</tr>'
#
#         html += """
#                 </tbody>
#             </table>
#         </div>
#         <div style="margin-top: 10px; padding: 8px; background: #f8f9fa; border-radius: 5px;">
#             <p style="margin: 0; color: #6c757d; font-size: 0.95em;">
#                 <strong>Note:</strong>
#                 <span style="background-color: rgba(40, 167, 69, 0.25); padding: 2px 8px; border-radius: 3px; color: #28a745; font-weight: bold;">Highlighted values</span>
#                 indicate the best performance for each metric
#             </p>
#         </div>
#         """
#
#         return html
#
#     def _create_comparison_statistics_table(self) -> str:
#         """Create comparison statistics table with hyperparameters."""
#         html = """
#         <div class="table-container">
#             <div class="table-title">📊 Performance & Configuration Comparison</div>
#             <div class="table-description">
#                 Training performance metrics and detailed hyperparameter comparison. Best values are highlighted in green.
#                 Different colors indicate different parameter values for easy A/B analysis.
#             </div>
#
#             <!-- Metrics Comparison Table - BACK TO VERTICAL LAYOUT -->
#             <h3 style="color: #495057; margin-top: 20px; margin-bottom: 15px; font-size: 1.4em;">📈 Training & Evaluation Metrics</h3>
#
#             <div class="table-scroll">
#                 <table class="wide-table">
#                     <thead>
#                         <tr>
#                             <th>Metric</th>
#         """
#
#         # Add column headers
#         for dataset in self.datasets:
#             html += f'<th>{dataset["name"]}</th>'
#
#         html += """
#                         </tr>
#                     </thead>
#                     <tbody>
#         """
#
#         comparison_metrics = [
#             ('Total Epochs', 'total_epochs', '{:.0f}'),
#             ('Best Val Loss', 'best_val_loss', '{:.4f}'),
#             ('Best Epoch', 'best_epoch', '{:.0f}'),
#             ('Final Train Loss', 'final_train_loss', '{:.4f}'),
#             ('Final Val Loss', 'final_val_loss', '{:.4f}'),
#             ('Train Loss Reduction %', 'train_loss_reduction', '{:.2f}'),
#             ('Val Loss Reduction %', 'val_loss_reduction', '{:.2f}'),
#             ('Val AP', 'val_ap', '{:.3f}'),
#             ('Val AP50', 'val_ap50', '{:.3f}'),
#             ('Val AP75', 'val_ap75', '{:.3f}'),
#             ('Test AP', 'test_ap', '{:.3f}'),
#             ('Test AP50', 'test_ap50', '{:.3f}'),
#             ('Test AP75', 'test_ap75', '{:.3f}'),
#         ]
#
#         for metric_name, metric_key, format_str in comparison_metrics:
#             html += f'<tr><td>{metric_name}</td>'
#
#             values = []
#             for dataset in self.datasets:
#                 if metric_key in dataset['stats']:
#                     value = dataset['stats'][metric_key]
#                     values.append(value)
#                 else:
#                     values.append(None)
#
#             # Determine best value
#             if values and any(v is not None for v in values):
#                 valid_values = [v for v in values if v is not None]
#                 if 'loss' in metric_key.lower() and 'reduction' not in metric_key and 'best_epoch' not in metric_key:
#                     best_val = min(valid_values) if valid_values else None
#                 else:
#                     best_val = max(valid_values) if valid_values else None
#             else:
#                 best_val = None
#
#             # Add cells with highlighting
#             for i, dataset in enumerate(self.datasets):
#                 css_class = ""
#                 if metric_key in dataset['stats'] and values[i] is not None:
#                     value = dataset['stats'][metric_key]
#                     formatted_value = format_str.format(value)
#
#                     if best_val is not None and len(set(valid_values)) > 1:
#                         if value == best_val:
#                             css_class = "highlight-best"
#
#                 else:
#                     formatted_value = 'N/A'
#
#                 html += f'<td class="{css_class}">{formatted_value}</td>'
#
#             html += '</tr>'
#
#         html += """
#                     </tbody>
#                 </table>
#             </div>
#
#             <!-- Hyperparameters Comparison -->
#             <h3 style="color: #495057; margin-top: 30px; margin-bottom: 15px; font-size: 1.4em;">⚙️ Hyperparameters Comparison</h3>
#             <div class="table-scroll">
#                 <table class="wide-table">
#                     <thead>
#                         <tr>
#                             <th>Parameter</th>
#         """
#
#         for dataset in self.datasets:
#             html += f'<th>{dataset["name"]}</th>'
#
#         html += """
#                         </tr>
#                     </thead>
#                     <tbody>
#         """
#
#         # Collect all unique hyperparameters
#         all_hyperparams = set()
#         for dataset in self.datasets:
#             all_hyperparams.update(dataset['specs']['hyperparameters'].keys())
#
#         hyperparam_groups = [
#             ('Model Architecture', ['model_name', 'architecture', 'backbone_name', 'num_classes', 'image_size']),
#             ('Training Settings', ['batch_size', 'epochs', 'train_whole_model']),
#             ('NMS Configuration', ['nms_method', 'iou_thresh', 'score_thresh', 'sigma',
#                                    'pyfunc', 'max_nms_inputs', 'max_output_size']),
#             ('Loss Parameters', ['alpha', 'gamma', 'delta', 'box_loss_weight', 'label_smoothing']),
#             ('Learning Rate', ['learning_rate', 'lr_warmup_init', 'lr_warmup_epoch',
#                                'lr_decay_method', 'first_lr_drop_epoch', 'second_lr_drop_epoch',
#                                'poly_lr_power']),
#             ('Optimization', ['momentum', 'optimizer', 'weight_decay', 'clip_gradients_norm',
#                               'moving_average_decay']),
#             ('Augmentation', ['input_rand_hflip', 'jitter_min', 'jitter_max',
#                               'autoaugment_policy', 'grid_mask']),
#             ('Anchors/Detection', ['anchor_scale', 'aspect_ratios', 'num_scales']),
#             ('Other Training Settings', ['skip_crowd_during_training', 'max_instances_per_image',
#                                          'mixed_precision', 'drop_remainder', 'map_freq'])
#         ]
#
#         # Create mapping for consistent colors
#         param_color_map = {}
#         color_index = 0
#         colors = ['#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6610f2', '#e83e8c']
#
#         for group_name, params in hyperparam_groups:
#             # First check if any of the main specs should be included
#             main_params = []
#             if group_name == 'Model Architecture':
#                 for dataset in self.datasets:
#                     main_params.extend(dataset['specs']['main'].keys())
#             elif group_name == 'Training Settings':
#                 for dataset in self.datasets:
#                     main_params.extend(dataset['specs']['main'].keys())
#
#             group_params = [p for p in params if p in all_hyperparams or p in main_params]
#             if not group_params:
#                 continue
#
#             html += f"""
#                 <tr style="background: #e9ecef;">
#                     <td colspan="{len(self.datasets) + 1}" style="padding: 10px; font-weight: bold;
#                         color: #667eea; font-size: 1.05em; border-top: 1px solid #dee2e6;">
#                         {group_name}
#                     </td>
#                 </tr>
#             """
#
#             for param in group_params:
#                 param_label = param.replace('_', ' ').title()
#                 html += f'<tr><td>{param_label}</td>'
#
#                 # Collect values for this parameter across datasets
#                 param_values = []
#                 for dataset in self.datasets:
#                     # Check both main specs and hyperparameters
#                     value = dataset['specs']['hyperparameters'].get(param)
#                     if value is None:
#                         value = dataset['specs']['main'].get(param, 'N/A')
#                     param_values.append(str(value))
#
#                 # Check if values differ
#                 unique_values = set(v for v in param_values if v != 'N/A')
#                 values_differ = len(unique_values) > 1
#
#                 # Assign colors for different values
#                 if values_differ:
#                     for val in unique_values:
#                         if val not in param_color_map:
#                             param_color_map[val] = colors[color_index % len(colors)]
#                             color_index += 1
#
#                 # Add cells
#                 for value in param_values:
#                     style = ""
#                     if value != 'N/A':
#                         if values_differ:
#                             # Different values - assign consistent colors
#                             bg_color = param_color_map[value] + "20"  # 20% opacity
#                             text_color = param_color_map[value]
#                             style = f'style="background-color: {bg_color}; color: {text_color}; font-weight: bold;"'
#                         else:
#                             # Same values - keep consistent color for booleans
#                             if value in ['True', 'False']:
#                                 color = '#28a745' if value == 'True' else '#dc3545'
#                                 style = f'style="color: {color}; font-weight: bold;"'
#
#                     html += f'<td {style}>{value}</td>'
#
#                 html += '</tr>'
#
#         html += """
#                     </tbody>
#                 </table>
#             </div>
#
#             <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;">
#                 <p style="margin: 0; color: #6c757d; font-size: 0.9em;">
#                     <strong>Legend:</strong>
#                     <span style="background-color: rgba(40, 167, 69, 0.2); padding: 2px 6px; border-radius: 3px;">Best value</span>
#                     <span style="margin-left: 10px;">Different colors indicate different values for the same parameter</span>
#                 </p>
#             </div>
#         </div>
#         """
#
#         return html
#
#     def _create_summary_cards(self) -> str:
#         """Create summary cards for each model."""
#         html = """
#         <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
#         """
#
#         colors = ['#1f77b4', '#2ca02c', '#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
#                   '#17becf']
#
#         for idx, dataset in enumerate(self.datasets):
#             color = colors[idx % len(colors)]
#             val_ap = dataset['stats'].get('val_ap', 0)
#             test_ap = dataset['stats'].get('test_ap', 0)
#             best_loss = dataset['stats'].get('best_val_loss', 0)
#             best_epoch = dataset['stats'].get('best_epoch', 0)
#
#             html += f"""
#             <div style="background: white; border-radius: 10px; padding: 25px;
#                         box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-left: 5px solid {color};">
#                 <h3 style="margin: 0 0 20px 0; color: #333; font-size: 1.4em;">{dataset['name']}</h3>
#                 <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
#                     <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px;">
#                         <div style="color: #6c757d; font-size: 0.9em; margin-bottom: 5px;">Validation AP</div>
#                         <div style="font-size: 1.8em; font-weight: bold; color: {color};">{val_ap:.3f}</div>
#                     </div>
#                     <div style="text-align: center; padding: 15px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 8px;">
#                         <div style="color: #6c757d; font-size: 0.9em; margin-bottom: 5px;">Test AP</div>
#                         <div style="font-size: 1.8em; font-weight: bold; color: {color};">{test_ap:.3f}</div>
#                     </div>
#                 </div>
#                 <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
#                     <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 5px;">
#                         <div style="color: #6c757d; font-size: 0.85em;">Best Val Loss</div>
#                         <div style="font-size: 1.3em; font-weight: 600; color: #333;">{best_loss:.4f}</div>
#                     </div>
#                     <div style="text-align: center; padding: 12px; background: #f8f9fa; border-radius: 5px;">
#                         <div style="color: #6c757d; font-size: 0.85em;">Best Epoch</div>
#                         <div style="font-size: 1.3em; font-weight: 600; color: #333;">{best_epoch}</div>
#                     </div>
#                 </div>
#             </div>
#             """
#
#         html += "</div>"
#         return html
#
#     def visualize_comparison_table_only(self, model_configs: List[Dict[str, str]]) -> bool:
#         """
#         Generate and display comparison table only.
#
#         Args:
#             model_configs: List of dictionaries with 'path' and 'name' keys
#
#         Returns:
#             bool: True if successful, False otherwise
#         """
#         if not self.load_models(model_configs):
#             return False
#
#         print(f"\n🔄 Generating comparison table for {len(self.datasets)} model(s)...")
#
#         # Generate HTML
#         dashboard_html = self.generate_comparison_html()
#
#         # Save and open dashboard
#         temp_dir = tempfile.gettempdir()
#         dashboard_file = os.path.join(temp_dir, "model_comparison_table.html")
#
#         with open(dashboard_file, 'w', encoding='utf-8') as f:
#             f.write(dashboard_html)
#
#         print("\n📊 Opening comparison table in browser...")
#         webbrowser.open(f"file://{dashboard_file}", new=2)
#
#         print(f"\n✅ Comparison table opened: {dashboard_file}")
#         return True
#
#     @staticmethod
#     def find_recent_models(base_path: str, limit: int = 10) -> List[Dict[str, str]]:
#         """
#         Find the most recent N completed models in a directory structure.
#         Only returns models that have evaluation metrics (completed training).
#
#         Args:
#             base_path: Path to the base directory containing model folders
#             limit: Maximum number of completed models to return
#
#         Returns:
#             List of model configurations sorted by modification time (newest first)
#
#         Example directory structure:
#             models/
#             ├── efficientdet_lite0_2025-10-09_18-18-49/
#             │   ├── training_log.txt
#             │   └── training_specs.txt
#             ├── efficientdet_lite0_2025-10-09_18-35-35/
#             └── mobilenet_v2_2025-10-31_10-24-58/
#         """
#         base_dir = Path(base_path)
#         if not base_dir.exists():
#             print(f"❌ Base path does not exist: {base_path}")
#             return []
#
#         all_model_configs = []
#
#         # Look for model directories with training_log.txt
#         for model_dir in base_dir.iterdir():
#             if model_dir.is_dir():
#                 log_file = model_dir / "training_log.txt"
#                 if log_file.exists():
#                     # Use the parent folder name as the model name
#                     model_name = model_dir.name
#                     all_model_configs.append({
#                         'path': str(log_file),
#                         'name': model_name,
#                         'mtime': log_file.stat().st_mtime
#                     })
#
#         # Sort by modification time (newest first)
#         all_model_configs.sort(key=lambda x: x['mtime'], reverse=True)
#
#         # Filter to get only completed models (with evaluation metrics)
#         completed_models = []
#         for config in all_model_configs:
#             try:
#                 from ..parsers.log_parser import parse_training_log
#                 _, eval_metrics = parse_training_log(config['path'])
#
#                 # Check if model has evaluation metrics (completed training)
#                 if eval_metrics['validation'] or eval_metrics['test_CLEAN']:
#                     completed_models.append(config)
#
#                 # Stop when we have enough completed models
#                 if len(completed_models) >= limit:
#                     break
#
#             except Exception as e:
#                 print(f"⚠️ Error checking {config['name']}: {e}")
#                 continue
#
#         # Remove mtime from the final result
#         for config in completed_models:
#             del config['mtime']
#
#         print(f"📊 Found {len(completed_models)} completed models out of {len(all_model_configs)} total models")
#         return completed_models
