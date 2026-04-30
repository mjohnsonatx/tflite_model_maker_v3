"""Training visualizer tab for GUI integration."""
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Any, Optional
from pathlib import Path
import tempfile

# Import tkinterweb for HTML display
try:
    from tkinterweb import HtmlFrame
except ImportError:
    messagebox.showerror("Missing Dependency",
                         "tkinterweb is required for this feature. Please install it with: pip install tkinterweb")

# Import training visualizer modules
from training_visualizer.parsers.log_parser import parse_training_log
from training_visualizer.parsers.spec_parser import parse_model_specs
from training_visualizer.analysis.statistics import calculate_statistics
from training_visualizer.visualization.html_components import (
    create_comparison_statistics_table,
    create_evaluation_results_table
)


class VisualizerTab:
    """Tab for visualizing training logs using interactive plots."""

    def __init__(self, parent: tk.Widget, log_queue: Any):
        """Initialize the visualizer tab.

        Args:
            parent: Parent tkinter widget
            log_queue: Queue for logging messages
        """
        self.parent = parent
        self.log_queue = log_queue
        self.selected_logs: List[Dict[str, str]] = []
        self.html_frame: Optional[HtmlFrame] = None
        self.processing_thread: Optional[threading.Thread] = None
        self.temp_dir: str = tempfile.gettempdir()

        self.setup_ui()

    def setup_ui(self) -> None:
        """Set up the user interface for the visualizer tab."""
        # Main frame with padding
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Title and description
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill="x", pady=(0, 20))

        title_label = ttk.Label(title_frame, text="Training Log Visualizer",
                                font=('TkDefaultFont', 14, 'bold'))
        title_label.pack(anchor=tk.W)

        desc_label = ttk.Label(title_frame,
                               text="Select 1-3 training logs to analyze evaluation metrics and model performance",
                               font=('TkDefaultFont', 9))
        desc_label.pack(anchor=tk.W, pady=(5, 0))

        # Log selection section
        selection_frame = ttk.LabelFrame(main_frame, text="Training Log Selection", padding="10")
        selection_frame.pack(fill="x", pady=(0, 20))

        # Buttons for adding/removing logs
        button_frame = ttk.Frame(selection_frame)
        button_frame.pack(fill="x", pady=(0, 10))

        self.add_log_btn = ttk.Button(button_frame, text="Add Training Log",
                                      command=self.add_training_log)
        self.add_log_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.remove_log_btn = ttk.Button(button_frame, text="Remove Selected",
                                         command=self.remove_selected_log)
        self.remove_log_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.clear_logs_btn = ttk.Button(button_frame, text="Clear All",
                                         command=self.clear_all_logs)
        self.clear_logs_btn.pack(side=tk.LEFT)

        # Selected logs listbox
        listbox_frame = ttk.Frame(selection_frame)
        listbox_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(listbox_frame, text="Selected Training Logs:").pack(anchor=tk.W)

        # Create listbox with scrollbar
        listbox_scroll_frame = ttk.Frame(listbox_frame)
        listbox_scroll_frame.pack(fill="x", pady=(5, 0))

        self.logs_listbox = tk.Listbox(listbox_scroll_frame, height=6, font=('TkDefaultFont', 9))
        scrollbar = ttk.Scrollbar(listbox_scroll_frame, orient=tk.VERTICAL,
                                  command=self.logs_listbox.yview)
        self.logs_listbox.configure(yscrollcommand=scrollbar.set)

        self.logs_listbox.pack(side=tk.LEFT, fill="both", expand=True)
        scrollbar.pack(side=tk.RIGHT, fill="y")

        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=(0, 20))

        self.visualize_btn = ttk.Button(control_frame, text="Generate Analysis",
                                        command=self.start_analysis, state="disabled",
                                        style="Accent.TButton")
        self.visualize_btn.pack(side=tk.LEFT, padx=(0, 10))

        self.progress_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(control_frame, textvariable=self.progress_var,
                                      font=('TkDefaultFont', 9))
        self.status_label.pack(side=tk.LEFT)

        # Analysis display area
        analysis_frame = ttk.LabelFrame(main_frame, text="Training Analysis", padding="10")
        analysis_frame.pack(fill="both", expand=True)

        # Add information label
        info_frame = ttk.Frame(analysis_frame)
        info_frame.pack(fill="x", pady=(0, 10))

        info_label = ttk.Label(
            info_frame,
            text="💡 Tip: Use the ▼ button in the log panel header to collapse it for more viewing space",
            font=('TkDefaultFont', 9),
            foreground="blue"
        )
        info_label.pack(side=tk.LEFT)

        # HTML frame for displaying analysis
        try:
            self.html_frame = HtmlFrame(analysis_frame, horizontal_scrollbar="auto")
            self.html_frame.pack(fill="both", expand=True)

            # Load initial welcome message
            self._load_welcome_html()

        except Exception as e:
            # Fallback if HtmlFrame fails
            fallback_label = ttk.Label(analysis_frame,
                                       text=f"HTML display not available: {str(e)}\n"
                                            "Analysis will open in your web browser instead.",
                                       justify=tk.CENTER, foreground="gray")
            fallback_label.pack(expand=True)
            self.html_frame = None

        # Update button states
        self._update_button_states()

    def _load_welcome_html(self) -> None:
        """Load welcome HTML content."""
        welcome_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Training Analyzer</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    padding: 20px;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    margin: 0;
                    height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .welcome-container {
                    background: white;
                    border-radius: 10px;
                    padding: 30px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    text-align: center;
                    max-width: 500px;
                }
                .welcome-title {
                    color: #333;
                    font-size: 24px;
                    margin-bottom: 15px;
                }
                .welcome-text {
                    color: #666;
                    font-size: 16px;
                    line-height: 1.5;
                    margin-bottom: 20px;
                }
                .steps {
                    text-align: left;
                    background: #f8f9fa;
                    border-radius: 8px;
                    padding: 20px;
                    margin-top: 20px;
                }
                .step {
                    display: flex;
                    align-items: center;
                    margin-bottom: 10px;
                    font-size: 14px;
                }
                .step-number {
                    background: #007bff;
                    color: white;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 12px;
                    font-weight: bold;
                    font-size: 12px;
                }
            </style>
        </head>
        <body>
            <div class="welcome-container">
                <h1 class="welcome-title">🚀 Training Analyzer</h1>
                <p class="welcome-text">
                    Analyze and compare your model training evaluation metrics and performance statistics.
                </p>

                <div class="steps">
                    <div class="step">
                        <div class="step-number">1</div>
                        <span>Click "Add Training Log" to select your training_log.txt files</span>
                    </div>
                    <div class="step">
                        <div class="step-number">2</div>
                        <span>Select up to 3 logs for comparison (or just 1 for single analysis)</span>
                    </div>
                    <div class="step">
                        <div class="step-number">3</div>
                        <span>Click "Generate Analysis" to view evaluation metrics and statistics</span>
                    </div>
                    <div class="step">
                        <div class="step-number">💡</div>
                        <span><strong>Pro Tip:</strong> Use the ▼ button in the log panel to collapse it for more space!</span>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        if self.html_frame:
            self.html_frame.load_html(welcome_html)

    def add_training_log(self) -> None:
        """Add a training log file to the selection."""
        if len(self.selected_logs) >= 3:
            messagebox.showwarning("Maximum Logs", "You can select a maximum of 3 training logs.")
            return

        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Training Log",
            filetypes=[
                ("Training logs", "training_log.txt"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ],
            initialdir=os.getcwd()
        )

        if not file_path:
            return

        # Check if already selected
        for log in self.selected_logs:
            if log['path'] == file_path:
                messagebox.showwarning("Duplicate Log", "This training log is already selected.")
                return

        # Validate the log file
        if not self._validate_training_log(file_path):
            messagebox.showerror("Invalid Log",
                                 "The selected file doesn't appear to be a valid_FINAL_CLEAN_DONISH training log.")
            return

        # Create a display name from the file path
        log_dir = Path(file_path).parent
        display_name = f"{log_dir.name} ({Path(file_path).name})"

        # Add to selection
        log_info = {
            'name': display_name,
            'path': file_path,
            'display': f"{len(self.selected_logs) + 1}. {display_name}"
        }

        self.selected_logs.append(log_info)
        self.logs_listbox.insert(tk.END, log_info['display'])

        self._update_button_states()
        self._log_message(f"Added training log: {display_name}")

    def remove_selected_log(self) -> None:
        """Remove the selected log from the listbox."""
        selection = self.logs_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a log to remove.")
            return

        index = selection[0]
        removed_log = self.selected_logs.pop(index)
        self.logs_listbox.delete(index)

        # Re-number the remaining items
        self.logs_listbox.delete(0, tk.END)
        for i, log in enumerate(self.selected_logs):
            log['display'] = f"{i + 1}. {log['name']}"
            self.logs_listbox.insert(tk.END, log['display'])

        self._update_button_states()
        self._log_message(f"Removed training log: {removed_log['name']}")

    def clear_all_logs(self) -> None:
        """Clear all selected logs."""
        if not self.selected_logs:
            return

        count = len(self.selected_logs)
        self.selected_logs.clear()
        self.logs_listbox.delete(0, tk.END)

        self._update_button_states()
        self._log_message(f"Cleared {count} training log(s)")

    def start_analysis(self) -> None:
        """Start the analysis process in a separate thread."""
        if not self.selected_logs:
            return

        # Disable button and update status
        self.visualize_btn.configure(state="disabled")
        self.progress_var.set("Processing...")

        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self._process_analysis, daemon=True)
        self.processing_thread.start()

    def _process_analysis(self) -> None:
        """Process the analysis in a separate thread."""
        try:
            self._log_message("Starting training analysis...")

            # Parse all selected logs
            datasets = []

            for i, log_info in enumerate(self.selected_logs):
                self.parent.after(0, lambda i=i: self.progress_var.set(
                    f"Processing log {i + 1}/{len(self.selected_logs)}..."))

                log_path = log_info['path']

                # Parse log file
                metrics, eval_metrics = parse_training_log(log_path)

                # Find specs file
                spec_path = Path(log_path).parent / "training_specs.txt"
                model_specs = {}
                if spec_path.exists():
                    model_specs = parse_model_specs(spec_path)
                else:
                    model_specs = {'main': {}, 'hyperparameters': {}}

                # Calculate statistics
                stats = calculate_statistics(metrics, eval_metrics)

                datasets.append({
                    'name': log_info['name'],
                    'metrics': metrics,
                    'eval_metrics': eval_metrics,
                    'specs': model_specs,
                    'stats': stats
                })

                self._log_message(f"Processed {log_info['name']}: {len(metrics['epoch'])} epochs")

            # Create analysis dashboard
            self.parent.after(0, lambda: self.progress_var.set("Generating analysis..."))
            dashboard_html = self._create_analysis_dashboard(datasets)

            # Display the analysis
            self.parent.after(0, lambda: self._display_analysis(dashboard_html))

        except Exception as e:
            self._log_message(f"Error during analysis: {str(e)}")
            self.parent.after(0, lambda: messagebox.showerror("Analysis Error", str(e)))
        finally:
            # Re-enable button and reset status
            self.parent.after(0, self._reset_ui_state)

    def _create_analysis_dashboard(self, datasets: List[Dict[str, Any]]) -> str:
        """Create HTML analysis dashboard without plots."""

        if len(datasets) == 1:
            return self._create_single_analysis_dashboard(datasets[0])
        else:
            return self._create_comparison_analysis_dashboard(datasets)

    def _create_single_analysis_dashboard(self, dataset: Dict[str, Any]) -> str:
        """Create analysis dashboard for single dataset."""

        # Create evaluation results table
        eval_result = self._create_evaluation_table(dataset['eval_metrics'])

        # Create statistics table
        stats_html = create_comparison_statistics_table([dataset])

        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Training Analysis: {dataset['name']}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
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

                .content-container {{
                    background: white;
                    border-radius: 15px;
                    padding: 30px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                }}

                .section-title {{
                    font-size: 1.5em;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #667eea;
                }}
            </style>
        </head>
        <body>
            <div class="dashboard-header">
                <h1>📊 Training Analysis</h1>
                <p>Detailed evaluation metrics and statistics for {dataset['name']}</p>
            </div>

            <div class="content-container">
                <div class="section-title">📈 Training Statistics</div>
                {stats_html}
            </div>

            <div class="content-container">
                <div class="section-title">🎯 Evaluation Metrics</div>
                {eval_result}
            </div>
        </body>
        </html>
        """

        return dashboard_html

    def _create_comparison_analysis_dashboard(self, datasets: List[Dict[str, Any]]) -> str:
        """Create comparison analysis dashboard for multiple datasets."""

        # Create evaluation comparison with table data
        eval_result = self._create_comparison_evaluation_table(datasets)
        table_html, all_val_values, all_test_values = eval_result

        # Create evaluation results table
        eval_table_html = create_evaluation_results_table(table_html, all_val_values, all_test_values)

        # Create comparison statistics table
        stats_html = create_comparison_statistics_table(datasets)

        # Create summary cards
        summary_cards_html = self._create_summary_cards(datasets)

        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Training Comparison Analysis</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
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

                .content-container {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    margin-bottom: 30px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    transition: transform 0.3s ease;
                }}

                .content-container:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 12px 35px rgba(0,0,0,0.25);
                }}

                .section-title {{
                    font-size: 1.8em;
                    font-weight: 600;
                    color: #333;
                    margin-bottom: 20px;
                    padding-bottom: 12px;
                    border-bottom: 3px solid #667eea;
                    display: flex;
                    align-items: center;
                }}

                .section-description {{
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
                <h1>📊 Model Training Comparison</h1>
                <p>Comprehensive analysis of {len(datasets)} training runs</p>
            </div>

            {summary_cards_html}

            {stats_html}

            <div class="section-divider"></div>

            <div class="content-container">
                <div class="section-title">🎯 Evaluation Metrics Comparison</div>
                <div class="section-description">
                    Compare validation and test_CLEAN performance across all metrics. Detailed breakdown of AP/AR scores.
                </div>
                {eval_table_html}
            </div>

            <div style="text-align: center; color: white; padding: 30px; opacity: 0.9;">
                <p style="font-size: 1.1em;">💡 Tip: This analysis focuses on evaluation metrics and training statistics</p>
            </div>
        </body>
        </html>
        """

        return dashboard_html

    def _create_evaluation_table(self, eval_metrics: Dict[str, Any]) -> str:
        """Create evaluation table for single dataset."""
        table_data = []
        all_val_values = {}
        all_test_values = {}

        if 'validation' in eval_metrics and eval_metrics['validation']:
            all_val_values = eval_metrics['validation']

        if 'test_CLEAN' in eval_metrics and eval_metrics['test_CLEAN']:
            all_test_values = eval_metrics['test_CLEAN']

        # Create table data structure expected by create_evaluation_results_table
        table_data = [{'name': 'Current Model', 'val_ap': all_val_values.get('AP', 0),
                       'test_ap': all_test_values.get('AP', 0)}]

        return create_evaluation_results_table(table_data, all_val_values, all_test_values)

    def _create_comparison_evaluation_table(self, datasets: List[Dict[str, Any]]):
        """Create comparison evaluation table data."""
        table_data = []
        all_val_values = {}
        all_test_values = {}

        for dataset in datasets:
            eval_metrics = dataset['eval_metrics']
            val_ap = eval_metrics.get('validation', {}).get('AP', 0)
            test_ap = eval_metrics.get('test_CLEAN', {}).get('AP', 0)

            table_data.append({
                'name': dataset['name'],
                'val_ap': val_ap,
                'test_ap': test_ap
            })

            # Collect all validation metrics for detailed comparison
            if 'validation' in eval_metrics:
                for key, value in eval_metrics['validation'].items():
                    if key not in all_val_values:
                        all_val_values[key] = []
                    all_val_values[key].append(value if value is not None else 0)

            # Collect all test_CLEAN metrics for detailed comparison
            if 'test_CLEAN' in eval_metrics:
                for key, value in eval_metrics['test_CLEAN'].items():
                    if key not in all_test_values:
                        all_test_values[key] = []
                    all_test_values[key].append(value if value is not None else 0)

        return table_data, all_val_values, all_test_values

    def _create_summary_cards(self, datasets: List[Dict[str, Any]]) -> str:
        """Create summary cards for comparison view."""
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
        return summary_cards_html

    def _display_analysis(self, html_content: str) -> None:
        """Display the analysis HTML content."""
        try:
            if self.html_frame:
                # Display in embedded HTML frame
                self.html_frame.load_html(html_content)
                self._log_message("Analysis displayed in embedded view")
            else:
                # Fallback: save to file and open in browser
                import webbrowser
                html_file = os.path.join(self.temp_dir, "training_analysis.html")
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                webbrowser.open(f"file://{html_file}", new=2)
                self._log_message(f"Analysis opened in browser: {html_file}")

            self.progress_var.set("Analysis complete!")

        except Exception as e:
            self._log_message(f"Error displaying analysis: {str(e)}")
            raise

    def _reset_ui_state(self) -> None:
        """Reset UI state after processing."""
        self.visualize_btn.configure(state="normal" if self.selected_logs else "disabled")
        self.progress_var.set("Ready")

    def _update_button_states(self) -> None:
        """Update button states based on current selection."""
        has_logs = len(self.selected_logs) > 0
        has_max_logs = len(self.selected_logs) >= 3

        self.visualize_btn.configure(state="normal" if has_logs else "disabled")
        self.add_log_btn.configure(state="disabled" if has_max_logs else "normal")
        self.remove_log_btn.configure(state="normal" if has_logs else "disabled")
        self.clear_logs_btn.configure(state="normal" if has_logs else "disabled")

    def _validate_training_log(self, file_path: str) -> bool:
        """Validate that the file is a training log."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Read first few lines to check format
                lines = [f.readline().strip() for _ in range(10)]

            # Look for training log indicators
            content = ' '.join(lines).lower()
            indicators = ['training session started', 'epoch', 'loss', 'det_loss', 'cls_loss']

            return any(indicator in content for indicator in indicators)

        except Exception:
            return False

    def _log_message(self, message: str) -> None:
        """Log a message to the shared log queue."""
        if self.log_queue:
            self.log_queue.put(f"[Visualizer] {message}")

# """Training visualizer tab for GUI integration."""
# import os
# import threading
# import tkinter as tk
# from tkinter import ttk, filedialog, messagebox
# from typing import List, Dict, Any, Optional
# from pathlib import Path
#
# # Import training visualizer modules for parsing only
# from training_visualizer.parsers.log_parser import parse_training_log
# from training_visualizer.parsers.spec_parser import parse_model_specs
# from training_visualizer.analysis.statistics import calculate_statistics
#
#
# class VisualizerTab:
#     """Tab for visualizing training logs using interactive plots."""
#
#     def __init__(self, parent: tk.Widget, log_queue: Any):
#         """Initialize the visualizer tab.
#
#         Args:
#             parent: Parent tkinter widget
#             log_queue: Queue for logging messages
#         """
#         self.parent = parent
#         self.log_queue = log_queue
#         self.selected_logs: List[Dict[str, str]] = []
#         self.processing_thread: Optional[threading.Thread] = None
#         self.results_frame: Optional[ttk.Frame] = None
#
#         self.setup_ui()
#
#     def setup_ui(self) -> None:
#         """Set up the user interface for the visualizer tab."""
#         # Main frame with padding
#         main_frame = ttk.Frame(self.parent, padding="10")
#         main_frame.pack(fill="both", expand=True)
#
#         # Title and description
#         title_frame = ttk.Frame(main_frame)
#         title_frame.pack(fill="x", pady=(0, 20))
#
#         title_label = ttk.Label(title_frame, text="Training Log Visualizer",
#                                 font=('TkDefaultFont', 14, 'bold'))
#         title_label.pack(anchor=tk.W)
#
#         desc_label = ttk.Label(title_frame,
#                                text="Select 1-3 training logs to analyze training metrics and performance",
#                                font=('TkDefaultFont', 9))
#         desc_label.pack(anchor=tk.W, pady=(5, 0))
#
#         # Log selection section
#         selection_frame = ttk.LabelFrame(main_frame, text="Training Log Selection", padding="10")
#         selection_frame.pack(fill="x", pady=(0, 20))
#
#         # Buttons for adding/removing logs
#         button_frame = ttk.Frame(selection_frame)
#         button_frame.pack(fill="x", pady=(0, 10))
#
#         self.add_log_btn = ttk.Button(button_frame, text="Add Training Log",
#                                       command=self.add_training_log)
#         self.add_log_btn.pack(side=tk.LEFT, padx=(0, 10))
#
#         self.remove_log_btn = ttk.Button(button_frame, text="Remove Selected",
#                                          command=self.remove_selected_log)
#         self.remove_log_btn.pack(side=tk.LEFT, padx=(0, 10))
#
#         self.clear_logs_btn = ttk.Button(button_frame, text="Clear All",
#                                          command=self.clear_all_logs)
#         self.clear_logs_btn.pack(side=tk.LEFT)
#
#         # Selected logs listbox
#         listbox_frame = ttk.Frame(selection_frame)
#         listbox_frame.pack(fill="x", pady=(0, 10))
#
#         ttk.Label(listbox_frame, text="Selected Training Logs:").pack(anchor=tk.W)
#
#         # Create listbox with scrollbar
#         listbox_scroll_frame = ttk.Frame(listbox_frame)
#         listbox_scroll_frame.pack(fill="x", pady=(5, 0))
#
#         self.logs_listbox = tk.Listbox(listbox_scroll_frame, height=6, font=('TkDefaultFont', 9))
#         scrollbar = ttk.Scrollbar(listbox_scroll_frame, orient=tk.VERTICAL,
#                                   command=self.logs_listbox.yview)
#         self.logs_listbox.configure(yscrollcommand=scrollbar.set)
#
#         self.logs_listbox.pack(side=tk.LEFT, fill="both", expand=True)
#         scrollbar.pack(side=tk.RIGHT, fill="y")
#
#         # Control buttons
#         control_frame = ttk.Frame(main_frame)
#         control_frame.pack(fill="x", pady=(0, 20))
#
#         self.visualize_btn = ttk.Button(control_frame, text="Parse Training Data",
#                                         command=self.start_parsing, state="disabled",
#                                         style="Accent.TButton")
#         self.visualize_btn.pack(side=tk.LEFT, padx=(0, 10))
#
#         self.progress_var = tk.StringVar(value="Ready")
#         self.status_label = ttk.Label(control_frame, textvariable=self.progress_var,
#                                       font=('TkDefaultFont', 9))
#         self.status_label.pack(side=tk.LEFT)
#
#         # Results display area with scrolling
#         results_label_frame = ttk.LabelFrame(main_frame, text="Training Data Analysis", padding="10")
#         results_label_frame.pack(fill="both", expand=True)
#
#         # Add information label
#         info_frame = ttk.Frame(results_label_frame)
#         info_frame.pack(fill="x", pady=(0, 10))
#
#         info_label = ttk.Label(
#             info_frame,
#             text="💡 Tip: Use the ▼ button in the log panel header to collapse it for more data viewing space",
#             font=('TkDefaultFont', 9),
#             foreground="blue"
#         )
#         info_label.pack(side=tk.LEFT)
#
#         # Create scrollable frame for results
#         self.canvas = tk.Canvas(results_label_frame, highlightthickness=0)
#         self.scrollbar = ttk.Scrollbar(results_label_frame, orient="vertical", command=self.canvas.yview)
#         self.scrollable_frame = ttk.Frame(self.canvas)
#
#         self.scrollable_frame.bind(
#             "<Configure>",
#             lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
#         )
#
#         self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
#         self.canvas.configure(yscrollcommand=self.scrollbar.set)
#
#         # Pack scrollable components
#         self.canvas.pack(side="left", fill="both", expand=True)
#         self.scrollbar.pack(side="right", fill="y")
#
#         # Bind mouse wheel to canvas
#         self.canvas.bind("<MouseWheel>", self._on_mousewheel)
#         self.canvas.bind("<Button-4>", self._on_mousewheel)  # Linux
#         self.canvas.bind("<Button-5>", self._on_mousewheel)  # Linux
#
#         # Load initial welcome message
#         self._load_welcome_message()
#
#         # Update button states
#         self._update_button_states()
#
#     def _load_welcome_message(self) -> None:
#         """Load welcome message in the scrollable frame."""
#         welcome_frame = ttk.Frame(self.scrollable_frame, padding="20")
#         welcome_frame.pack(fill="both", expand=True)
#
#         # Title
#         title_label = ttk.Label(welcome_frame, text="🚀 Training Data Analyzer",
#                                 font=('TkDefaultFont', 16, 'bold'))
#         title_label.pack(pady=(0, 15))
#
#         # Description
#         desc_label = ttk.Label(welcome_frame,
#                                text="Analyze and compare your model training metrics and evaluation results.",
#                                font=('TkDefaultFont', 11), justify=tk.CENTER)
#         desc_label.pack(pady=(0, 20))
#
#         # Steps frame
#         steps_frame = ttk.LabelFrame(welcome_frame, text="How to Use", padding="15")
#         steps_frame.pack(fill="x", pady=(0, 10))
#
#         steps = [
#             "1. Click 'Add Training Log' to select your training_log.txt files",
#             "2. Select up to 3 logs for comparison (or just 1 for single analysis)",
#             "3. Click 'Parse Training Data' to analyze metrics and evaluation results",
#             "💡 Pro Tip: Use the ▼ button in the log panel for more viewing space!"
#         ]
#
#         for step in steps:
#             step_label = ttk.Label(steps_frame, text=step, font=('TkDefaultFont', 10))
#             step_label.pack(anchor=tk.W, pady=2)
#
#     def _on_mousewheel(self, event) -> None:
#         """Handle mouse wheel scrolling."""
#         if event.delta:
#             # Windows and MacOS
#             self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
#         else:
#             # Linux
#             if event.num == 4:
#                 self.canvas.yview_scroll(-1, "units")
#             elif event.num == 5:
#                 self.canvas.yview_scroll(1, "units")
#
#     def add_training_log(self) -> None:
#         """Add a training log file to the selection."""
#         if len(self.selected_logs) >= 3:
#             messagebox.showwarning("Maximum Logs", "You can select a maximum of 3 training logs.")
#             return
#
#         # Open file dialog
#         file_path = filedialog.askopenfilename(
#             title="Select Training Log",
#             filetypes=[
#                 ("Training logs", "training_log.txt"),
#                 ("Text files", "*.txt"),
#                 ("All files", "*.*")
#             ],
#             initialdir=os.getcwd()
#         )
#
#         if not file_path:
#             return
#
#         # Check if already selected
#         for log in self.selected_logs:
#             if log['path'] == file_path:
#                 messagebox.showwarning("Duplicate Log", "This training log is already selected.")
#                 return
#
#         # Validate the log file
#         if not self._validate_training_log(file_path):
#             messagebox.showerror("Invalid Log",
#                                  "The selected file doesn't appear to be a valid_FINAL_CLEAN_DONISH training log.")
#             return
#
#         # Create a display name from the file path
#         log_dir = Path(file_path).parent
#         display_name = f"{log_dir.name} ({Path(file_path).name})"
#
#         # Add to selection
#         log_info = {
#             'name': display_name,
#             'path': file_path,
#             'display': f"{len(self.selected_logs) + 1}. {display_name}"
#         }
#
#         self.selected_logs.append(log_info)
#         self.logs_listbox.insert(tk.END, log_info['display'])
#
#         self._update_button_states()
#         self._log_message(f"Added training log: {display_name}")
#
#     def remove_selected_log(self) -> None:
#         """Remove the selected log from the listbox."""
#         selection = self.logs_listbox.curselection()
#         if not selection:
#             messagebox.showwarning("No Selection", "Please select a log to remove.")
#             return
#
#         index = selection[0]
#         removed_log = self.selected_logs.pop(index)
#         self.logs_listbox.delete(index)
#
#         # Re-number the remaining items
#         self.logs_listbox.delete(0, tk.END)
#         for i, log in enumerate(self.selected_logs):
#             log['display'] = f"{i + 1}. {log['name']}"
#             self.logs_listbox.insert(tk.END, log['display'])
#
#         self._update_button_states()
#         self._log_message(f"Removed training log: {removed_log['name']}")
#
#     def clear_all_logs(self) -> None:
#         """Clear all selected logs."""
#         if not self.selected_logs:
#             return
#
#         count = len(self.selected_logs)
#         self.selected_logs.clear()
#         self.logs_listbox.delete(0, tk.END)
#
#         self._update_button_states()
#         self._log_message(f"Cleared {count} training log(s)")
#
#     def start_parsing(self) -> None:
#         """Start the data parsing process in a separate thread."""
#         if not self.selected_logs:
#             return
#
#         # Disable button and update status
#         self.visualize_btn.configure(state="disabled")
#         self.progress_var.set("Parsing...")
#
#         # Start processing in separate thread
#         self.processing_thread = threading.Thread(target=self._process_parsing, daemon=True)
#         self.processing_thread.start()
#
#     def _process_parsing(self) -> None:
#         """Process the data parsing in a separate thread."""
#         try:
#             self._log_message("Starting training data parsing...")
#
#             # Parse all selected logs
#             parsed_data = []
#
#             for i, log_info in enumerate(self.selected_logs):
#                 self.parent.after(0, lambda i=i: self.progress_var.set(
#                     f"Parsing log {i + 1}/{len(self.selected_logs)}..."))
#
#                 log_path = log_info['path']
#
#                 # Parse log file
#                 metrics, eval_metrics = parse_training_log(log_path)
#
#                 # Find specs file
#                 spec_path = Path(log_path).parent / "training_specs.txt"
#                 model_specs = {}
#                 if spec_path.exists():
#                     model_specs = parse_model_specs(spec_path)
#                 else:
#                     model_specs = {'main': {}, 'hyperparameters': {}}
#
#                 # Calculate statistics
#                 stats = calculate_statistics(metrics, eval_metrics)
#
#                 parsed_data.append({
#                     'name': log_info['name'],
#                     'metrics': metrics,
#                     'eval_metrics': eval_metrics,
#                     'specs': model_specs,
#                     'stats': stats
#                 })
#
#                 self._log_message(f"Parsed {log_info['name']}: {len(metrics['epoch'])} epochs")
#
#             # Display the parsed data
#             self.parent.after(0, lambda: self._display_parsed_data(parsed_data))
#
#         except Exception as e:
#             self._log_message(f"Error during parsing: {str(e)}")
#             self.parent.after(0, lambda: messagebox.showerror("Parsing Error", str(e)))
#         finally:
#             # Re-enable button and reset status
#             self.parent.after(0, self._reset_ui_state)
#
#     def _display_parsed_data(self, parsed_data: List[Dict[str, Any]]) -> None:
#         """Display the parsed training data in the scrollable frame."""
#         try:
#             # Clear existing content
#             for widget in self.scrollable_frame.winfo_children():
#                 widget.destroy()
#
#             # Main container
#             main_container = ttk.Frame(self.scrollable_frame, padding="10")
#             main_container.pack(fill="both", expand=True)
#
#             if len(parsed_data) == 1:
#                 self._display_single_analysis(main_container, parsed_data[0])
#             else:
#                 self._display_comparison_analysis(main_container, parsed_data)
#
#             # Update canvas scroll region
#             self.scrollable_frame.update_idletasks()
#             self.canvas.configure(scrollregion=self.canvas.bbox("all"))
#
#             self.progress_var.set("Analysis complete!")
#             self._log_message(f"Successfully displayed analysis for {len(parsed_data)} training run(s)")
#
#         except Exception as e:
#             self._log_message(f"Error displaying data: {str(e)}")
#             raise
#
#     def _display_single_analysis(self, parent: ttk.Frame, data: Dict[str, Any]) -> None:
#         """Display analysis for a single training run."""
#         # Title
#         title_frame = ttk.Frame(parent)
#         title_frame.pack(fill="x", pady=(0, 20))
#
#         title_label = ttk.Label(title_frame, text=f"📊 Training Analysis: {data['name']}",
#                                 font=('TkDefaultFont', 14, 'bold'))
#         title_label.pack(anchor=tk.W)
#
#         # Model specifications
#         self._create_specs_section(parent, data['specs'], data['name'])
#
#         # Training statistics
#         self._create_stats_section(parent, data['stats'], data['name'])
#
#         # Evaluation metrics
#         self._create_eval_section(parent, data['eval_metrics'], data['name'])
#
#     def _display_comparison_analysis(self, parent: ttk.Frame, data_list: List[Dict[str, Any]]) -> None:
#         """Display comparison analysis for multiple training runs."""
#         # Title
#         title_frame = ttk.Frame(parent)
#         title_frame.pack(fill="x", pady=(0, 20))
#
#         title_label = ttk.Label(title_frame, text=f"📊 Training Comparison: {len(data_list)} Models",
#                                 font=('TkDefaultFont', 14, 'bold'))
#         title_label.pack(anchor=tk.W)
#
#         # Summary comparison table
#         self._create_comparison_table(parent, data_list)
#
#         # Individual model details
#         for i, data in enumerate(data_list):
#             separator = ttk.Separator(parent, orient='horizontal')
#             separator.pack(fill="x", pady=(20, 10))
#
#             model_frame = ttk.LabelFrame(parent, text=f"Model {i + 1}: {data['name']}", padding="10")
#             model_frame.pack(fill="x", pady=(0, 10))
#
#             # Create a two-column layout for model details
#             details_frame = ttk.Frame(model_frame)
#             details_frame.pack(fill="x")
#
#             left_frame = ttk.Frame(details_frame)
#             left_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 10))
#
#             right_frame = ttk.Frame(details_frame)
#             right_frame.pack(side=tk.LEFT, fill="both", expand=True)
#
#             # Specs in left column, stats in right column
#             self._create_specs_section(left_frame, data['specs'], data['name'])
#             self._create_stats_section(right_frame, data['stats'], data['name'])
#
#             # Eval metrics full width
#             self._create_eval_section(model_frame, data['eval_metrics'], data['name'])
#
#     def _create_specs_section(self, parent: ttk.Frame, specs: Dict[str, Any], model_name: str) -> None:
#         """Create model specifications section."""
#         specs_frame = ttk.LabelFrame(parent, text="Model Specifications", padding="10")
#         specs_frame.pack(fill="x", pady=(0, 10))
#
#         # Main specs
#         if 'main' in specs and specs['main']:
#             for key, value in specs['main'].items():
#                 if value:  # Only show non-empty values
#                     row_frame = ttk.Frame(specs_frame)
#                     row_frame.pack(fill="x", pady=1)
#
#                     key_label = ttk.Label(row_frame, text=f"{key}:", font=('TkDefaultFont', 9, 'bold'))
#                     key_label.pack(side=tk.LEFT)
#
#                     value_label = ttk.Label(row_frame, text=str(value), font=('TkDefaultFont', 9))
#                     value_label.pack(side=tk.LEFT, padx=(5, 0))
#
#         # Hyperparameters
#         if 'hyperparameters' in specs and specs['hyperparameters']:
#             ttk.Separator(specs_frame, orient='horizontal').pack(fill="x", pady=(5, 5))
#
#             hyper_title = ttk.Label(specs_frame, text="Hyperparameters:", font=('TkDefaultFont', 9, 'bold'))
#             hyper_title.pack(anchor=tk.W, pady=(0, 5))
#
#             for key, value in specs['hyperparameters'].items():
#                 if value:  # Only show non-empty values
#                     row_frame = ttk.Frame(specs_frame)
#                     row_frame.pack(fill="x", pady=1)
#
#                     key_label = ttk.Label(row_frame, text=f"  {key}:", font=('TkDefaultFont', 9))
#                     key_label.pack(side=tk.LEFT)
#
#                     value_label = ttk.Label(row_frame, text=str(value), font=('TkDefaultFont', 9))
#                     value_label.pack(side=tk.LEFT, padx=(5, 0))
#
#     def _create_stats_section(self, parent: ttk.Frame, stats: Dict[str, Any], model_name: str) -> None:
#         """Create training statistics section."""
#         stats_frame = ttk.LabelFrame(parent, text="Training Statistics", padding="10")
#         stats_frame.pack(fill="x", pady=(0, 10))
#
#         # Key statistics
#         key_stats = [
#             ('Total Epochs', 'total_epochs'),
#             ('Best Epoch', 'best_epoch'),
#             ('Best Val Loss', 'best_val_loss'),
#             ('Final Train Loss', 'final_train_loss'),
#             ('Final Val Loss', 'final_val_loss'),
#         ]
#
#         for label, key in key_stats:
#             if key in stats and stats[key] is not None:
#                 row_frame = ttk.Frame(stats_frame)
#                 row_frame.pack(fill="x", pady=1)
#
#                 key_label = ttk.Label(row_frame, text=f"{label}:", font=('TkDefaultFont', 9, 'bold'))
#                 key_label.pack(side=tk.LEFT)
#
#                 value = stats[key]
#                 if isinstance(value, float):
#                     value_text = f"{value:.4f}"
#                 else:
#                     value_text = str(value)
#
#                 value_label = ttk.Label(row_frame, text=value_text, font=('TkDefaultFont', 9))
#                 value_label.pack(side=tk.LEFT, padx=(5, 0))
#
#     def _create_eval_section(self, parent: ttk.Frame, eval_metrics: Dict[str, Any], model_name: str) -> None:
#         """Create evaluation metrics section."""
#         eval_frame = ttk.LabelFrame(parent, text="Evaluation Metrics", padding="10")
#         eval_frame.pack(fill="x", pady=(0, 10))
#
#         # Validation metrics
#         if 'validation' in eval_metrics and eval_metrics['validation']:
#             val_title = ttk.Label(eval_frame, text="Validation Results:", font=('TkDefaultFont', 10, 'bold'))
#             val_title.pack(anchor=tk.W, pady=(0, 5))
#
#             self._create_metrics_table(eval_frame, eval_metrics['validation'], "Validation")
#
#         # Test metrics
#         if 'test_CLEAN' in eval_metrics and eval_metrics['test_CLEAN']:
#             if 'validation' in eval_metrics and eval_metrics['validation']:
#                 ttk.Separator(eval_frame, orient='horizontal').pack(fill="x", pady=(10, 10))
#
#             test_title = ttk.Label(eval_frame, text="Test Results:", font=('TkDefaultFont', 10, 'bold'))
#             test_title.pack(anchor=tk.W, pady=(0, 5))
#
#             self._create_metrics_table(eval_frame, eval_metrics['test_CLEAN'], "Test")
#
#     def _create_metrics_table(self, parent: ttk.Frame, metrics: Dict[str, Any], prefix: str) -> None:
#         """Create a table for AP/AR metrics."""
#         # Create frame for table
#         table_frame = ttk.Frame(parent)
#         table_frame.pack(fill="x", pady=(0, 5))
#
#         # Headers
#         headers = ["Metric", "Value"]
#
#         # Create treeview for table display
#         columns = tuple(headers)
#         tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=6)
#
#         # Configure column headings
#         for col in columns:
#             tree.heading(col, text=col)
#             tree.column(col, width=120, anchor='center')
#
#         # Add data rows
#         metric_items = [
#             ('AP (IoU=0.50:0.95)', 'AP'),
#             ('AP (IoU=0.50)', 'AP50'),
#             ('AP (IoU=0.75)', 'AP75'),
#             ('AP (small)', 'APs'),
#             ('AP (medium)', 'APm'),
#             ('AP (large)', 'APl'),
#             ('AR (max=1)', 'AR1'),
#             ('AR (max=10)', 'AR10'),
#             ('AR (max=100)', 'AR100'),
#             ('AR (small)', 'ARs'),
#             ('AR (medium)', 'ARm'),
#             ('AR (large)', 'ARl'),
#         ]
#
#         for display_name, key in metric_items:
#             if key in metrics and metrics[key] is not None:
#                 value = f"{metrics[key]:.3f}" if isinstance(metrics[key], (int, float)) else str(metrics[key])
#                 tree.insert('', 'end', values=(display_name, value))
#
#         tree.pack(fill="x")
#
#         # Add scrollbar for table
#         table_scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
#         tree.configure(yscrollcommand=table_scrollbar.set)
#         table_scrollbar.pack(side="right", fill="y")
#
#     def _create_comparison_table(self, parent: ttk.Frame, data_list: List[Dict[str, Any]]) -> None:
#         """Create comparison summary table."""
#         comp_frame = ttk.LabelFrame(parent, text="Performance Comparison", padding="10")
#         comp_frame.pack(fill="x", pady=(0, 20))
#
#         # Create comparison table
#         headers = ["Model"] + ["Val AP", "Test AP", "Best Epoch", "Best Val Loss"]
#         columns = tuple(headers)
#
#         tree = ttk.Treeview(comp_frame, columns=columns, show='headings', height=min(len(data_list) + 1, 8))
#
#         # Configure columns
#         tree.column("Model", width=200, anchor='w')
#         for col in headers[1:]:
#             tree.heading(col, text=col)
#             tree.column(col, width=100, anchor='center')
#
#         tree.heading("Model", text="Model")
#
#         # Add data rows
#         for data in data_list:
#             stats = data['stats']
#             eval_metrics = data['eval_metrics']
#
#             # Get validation and test_CLEAN AP values
#             val_ap = eval_metrics.get('validation', {}).get('AP', 0.0)
#             test_ap = eval_metrics.get('test_CLEAN', {}).get('AP', 0.0)
#             best_epoch = stats.get('best_epoch', 'N/A')
#             best_val_loss = stats.get('best_val_loss', 0.0)
#
#             val_ap_str = f"{val_ap:.3f}" if isinstance(val_ap, (int, float)) else "N/A"
#             test_ap_str = f"{test_ap:.3f}" if isinstance(test_ap, (int, float)) else "N/A"
#             best_loss_str = f"{best_val_loss:.4f}" if isinstance(best_val_loss, (int, float)) else "N/A"
#
#             tree.insert('', 'end', values=(
#                 data['name'],
#                 val_ap_str,
#                 test_ap_str,
#                 str(best_epoch),
#                 best_loss_str
#             ))
#
#         tree.pack(fill="x")
#
#         # Add scrollbar
#         comp_scrollbar = ttk.Scrollbar(comp_frame, orient="vertical", command=tree.yview)
#         tree.configure(yscrollcommand=comp_scrollbar.set)
#         comp_scrollbar.pack(side="right", fill="y")
#
#     def _reset_ui_state(self) -> None:
#         """Reset UI state after processing."""
#         self.visualize_btn.configure(state="normal" if self.selected_logs else "disabled")
#         self.progress_var.set("Ready")
#
#     def _validate_training_log(self, file_path: str) -> bool:
#         """Validate that the file is a training log."""
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 # Read first few lines to check format
#                 lines = [f.readline().strip() for _ in range(10)]
#
#             # Look for training log indicators
#             content = ' '.join(lines).lower()
#             indicators = ['training session started', 'epoch', 'loss', 'det_loss', 'cls_loss']
#
#             return any(indicator in content for indicator in indicators)
#
#         except Exception:
#             return False
#
#     def _update_button_states(self) -> None:
#         """Update button states based on current selection."""
#         has_logs = len(self.selected_logs) > 0
#         has_max_logs = len(self.selected_logs) >= 3
#
#         self.visualize_btn.configure(state="normal" if has_logs else "disabled")
#         self.add_log_btn.configure(state="disabled" if has_max_logs else "normal")
#         self.remove_log_btn.configure(state="normal" if has_logs else "disabled")
#         self.clear_logs_btn.configure(state="normal" if has_logs else "disabled")
#
#     def _log_message(self, message: str) -> None:
#         """Log a message to the shared log queue."""
#         if self.log_queue:
#             self.log_queue.put(f"[Visualizer] {message}")
#
