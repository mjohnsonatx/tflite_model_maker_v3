"""HTML component generation utilities."""


def create_model_specs_card(specs):
    """Create an HTML card displaying model specifications."""

    html = """
    <div style="background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h2 style="color: #333; margin-bottom: 20px; border-bottom: 2px solid #667eea; 
                   padding-bottom: 10px;">🔧 Model Configuration</h2>

        <!-- Main Parameters -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                    gap: 15px; margin-bottom: 20px;">
    """

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

        <!-- Hyperparameters Section (Not scrollable) -->
        <div style="margin-top: 20px; border: 1px solid #dee2e6; border-radius: 10px; 
                    overflow: hidden; background: white;">
            <div style="font-weight: bold; color: #667eea; 
                        font-size: 1.1em; padding: 15px; background: #f8f9fa;
                        border-bottom: 1px solid #dee2e6;">
                ⚙️ Advanced Hyperparameters
            </div>

            <div style="padding: 20px; background: white;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                           gap: 30px;">
    """

    # Group hyperparameters by category with same/different value highlighting
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

    # Create a simple hash for parameter values to assign consistent colors
    def get_param_color(param_name, param_value):
        """Get color for parameter value - consistent for same values."""
        if param_value in ['True', 'False']:
            return '#28a745' if param_value == 'True' else '#dc3545'
        # For numeric/string values, create a simple hash-based color
        hash_val = hash(f"{param_name}:{param_value}") % 7
        colors = ['#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6610f2', '#e83e8c']
        return colors[hash_val]

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

                # Get consistent color for value
                value_color = get_param_color(param, param_value)
                value_style = f"color: {value_color}; font-weight: 600;"

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
        </div>
    </div>
    """

    return html


def create_statistics_table(stats):
    """Create an HTML table with training statistics."""

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

    html += """
        <tr>
            <td colspan="2" class="section-header metric-name">Training Metrics</td>
        </tr>
    """

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


def create_comparison_statistics_table(datasets):
    """Create a comparison table for multiple datasets."""

    html = """
    <div style="background: white; border-radius: 15px; padding: 20px; margin-bottom: 30px; 
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h2 style="color: #333; margin-bottom: 20px; border-bottom: 2px solid #667eea; 
                   padding-bottom: 10px; font-size: 1.8em;">📊 Performance Comparison</h2>

        <!-- Metrics Comparison Table with Results Box -->
        <h3 style="color: #495057; margin-top: 20px; margin-bottom: 15px; font-size: 1.4em;">📈 Training & Evaluation Metrics</h3>

        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 1.1em; margin-bottom: 20px;">
                <thead>
                    <tr style="background: #f8f9fa;">
                        <th style="padding: 15px; text-align: left; border-bottom: 2px solid #667eea; font-size: 1.2em;">Metric</th>
    """

    # Add column headers
    for dataset in datasets:
        html += f'<th style="padding: 15px; text-align: center; border-bottom: 2px solid #667eea; font-size: 1.2em;">{dataset["name"]}</th>'

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
        html += f"""
            <tr style="border-bottom: 1px solid #e0e0e0;">
                <td style="padding: 14px; font-weight: 600; color: #495057; font-size: 1.05em;">{metric_name}</td>
        """

        values = []
        for dataset in datasets:
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
        for i, dataset in enumerate(datasets):
            if metric_key in dataset['stats'] and values[i] is not None:
                value = dataset['stats'][metric_key]
                formatted_value = format_str.format(value)

                # Highlight best value
                style = "padding: 14px; text-align: center; font-family: monospace; font-size: 1.1em;"

                # Only highlight if there are multiple different values
                if best_val is not None and len(set(valid_values)) > 1:
                    if value == best_val:
                        style += " background-color: rgba(40, 167, 69, 0.2); font-weight: bold; color: #28a745;"

            else:
                formatted_value = 'N/A'
                style = "padding: 14px; text-align: center; font-family: monospace; font-size: 1.1em; color: #6c757d;"

            html += f'<td style="{style}">{formatted_value}</td>'

        html += '</tr>'

    html += """
        </tbody>
    </table>
    </div>

    <!-- Hyperparameters Comparison -->
    <h3 style="color: #495057; margin-top: 30px; margin-bottom: 15px; font-size: 1.4em;">⚙️ Hyperparameters Comparison</h3>
    <div style="border: 1px solid #dee2e6; border-radius: 8px;">
    <table style="width: 100%; border-collapse: collapse; font-size: 1.05em;">
        <thead style="background: white; z-index: 10;">
            <tr style="background: #f8f9fa;">
                <th style="padding: 12px; text-align: left; border-bottom: 2px solid #667eea; font-size: 1.1em;">Parameter</th>
    """

    for dataset in datasets:
        html += f'<th style="padding: 12px; text-align: center; border-bottom: 2px solid #667eea; font-size: 1.1em;">{dataset["name"]}</th>'

    html += """
            </tr>
        </thead>
        <tbody>
    """

    # Collect all unique hyperparameters
    all_hyperparams = set()
    for dataset in datasets:
        all_hyperparams.update(dataset['specs']['hyperparameters'].keys())

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

    # Create mapping for consistent colors
    param_color_map = {}
    color_index = 0
    colors = ['#007bff', '#28a745', '#dc3545', '#ffc107', '#17a2b8', '#6610f2', '#e83e8c']

    for group_name, params in hyperparam_groups:
        group_params = [p for p in params if p in all_hyperparams]
        if not group_params:
            continue

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

            # Collect values for this parameter across datasets
            param_values = []
            for dataset in datasets:
                value = dataset['specs']['hyperparameters'].get(param, 'N/A')
                param_values.append(value)

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
            for i, value in enumerate(param_values):
                style = "padding: 10px 12px; text-align: center; font-family: monospace; font-size: 1.05em;"

                if value != 'N/A':
                    if values_differ:
                        # Different values - assign consistent colors
                        style += f" background-color: {param_color_map[value]}20; color: {param_color_map[value]}; font-weight: bold;"
                    else:
                        # Same values - keep consistent color
                        if value in ['True', 'False']:
                            color = '#28a745' if value == 'True' else '#dc3545'
                            style += f" color: {color}; font-weight: bold;"

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
            <span style="margin-left: 10px;">Different colors indicate different values for the same parameter</span>
        </p>
    </div>
    </div>
    """

    return html


def create_evaluation_results_table(table_data, all_val_values, all_test_values):
    """Create a results table box for evaluation metrics with highlighting."""

    html = """
    <div style="margin-top: 20px; background: white; border-radius: 10px; padding: 20px; 
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
        <h3 style="color: #333; margin-bottom: 15px; font-size: 1.3em;">📊 Evaluation Results Comparison</h3>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 1.1em;">
                <thead>
                    <tr style="background: #f8f9fa;">
    """

    # Add headers with larger font
    for key in table_data.keys():
        html += f'<th style="padding: 12px; text-align: {"left" if key == "Metric" else "center"}; border-bottom: 2px solid #667eea; font-size: 1.15em; font-weight: 600;">{key}</th>'

    html += """
                    </tr>
                </thead>
                <tbody>
    """

    # Add rows
    for i, metric in enumerate(table_data['Metric']):
        html += '<tr style="border-bottom: 1px solid #e0e0e0;">'

        # Metric name (larger font)
        html += f'<td style="padding: 12px; font-weight: 600; font-size: 1.05em;">{metric}</td>'

        # Collect all values for this row to find the best
        row_values = []
        for key in table_data.keys():
            if key != 'Metric':
                try:
                    row_values.append(float(table_data[key][i]))
                except:
                    row_values.append(None)

        # Find best value (highest) - only if values differ
        valid_values = [v for v in row_values if v is not None]
        best_value = max(valid_values) if valid_values and len(set(valid_values)) > 1 else None

        # Add value cells
        value_idx = 0
        for key in table_data.keys():
            if key == 'Metric':
                continue

            value_str = table_data[key][i]
            try:
                value_float = float(value_str)
            except:
                value_float = None

            style = "padding: 12px; text-align: center; font-family: monospace; font-size: 1.1em;"

            # Highlight best value with green background
            if best_value is not None and value_float == best_value:
                style += " background-color: rgba(40, 167, 69, 0.25); font-weight: bold; color: #28a745;"

            html += f'<td style="{style}">{value_str}</td>'
            value_idx += 1

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
    </div>
    """

    return html
