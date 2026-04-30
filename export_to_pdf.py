"""Export training visualizations to PDF."""
import sys
import os
from pathlib import Path
from datetime import datetime

# Ensure the project root is in the path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training_visualizer.parsers.log_parser import parse_training_log
from training_visualizer.parsers.spec_parser import parse_model_specs
from training_visualizer.analysis.statistics import calculate_statistics
from training_visualizer.visualization.plots import (
    create_evaluation_metrics_plot,
    create_interactive_plot,
    create_detailed_subplots
)
from training_visualizer.visualization.comparison_plots import (
    create_comparison_evaluation_plot,
    create_comparison_detailed_subplots,
    create_comparison_convergence_plot,
    create_comparison_learning_dynamics_plot
)

try:
    import kaleido  # For static image export

    KALEIDO_AVAILABLE = True
except ImportError:
    KALEIDO_AVAILABLE = False
    print("⚠️  Warning: kaleido not installed. Install with: pip install kaleido")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib import colors

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("⚠️  Warning: reportlab not installed. Install with: pip install reportlab")


def export_single_training_to_pdf(log_filepath, output_dir=None, spec_filepath=None):
    """Export single training visualization to PDF."""

    if not KALEIDO_AVAILABLE or not REPORTLAB_AVAILABLE:
        print("❌ Error: Missing required packages for PDF export")
        print("   Install with: pip install kaleido reportlab")
        return False

    try:
        # Parse data
        print(f"📖 Reading training log: {log_filepath}")
        metrics, eval_metrics = parse_training_log(log_filepath)

        # Try to find spec file
        if spec_filepath is None:
            log_dir = Path(log_filepath).parent
            spec_filepath = log_dir / "training_specs.txt"

        model_specs = parse_model_specs(spec_filepath) if spec_filepath and Path(spec_filepath).exists() else {
            'main': {}, 'hyperparameters': {}
        }

        if len(metrics['epoch']) == 0:
            print("⚠️  No training data found in the log file!")
            return False

        print(f"✅ Found {len(metrics['epoch'])} epochs of training data")

        # Calculate statistics
        stats = calculate_statistics(metrics, eval_metrics)

        # Setup output directory
        if output_dir is None:
            output_dir = Path.cwd() / "training_reports"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_specs['main'].get('model_name', 'model')
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).strip()
        pdf_filename = f"{safe_model_name}_{timestamp}.pdf"
        pdf_path = output_dir / pdf_filename

        print(f"\n🔄 Generating PDF report...")

        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        # Container for elements
        elements = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12
        )

        # Title
        elements.append(Paragraph("📊 Training Analysis Report", title_style))
        elements.append(Spacer(1, 0.2 * inch))

        # Model Information
        elements.append(Paragraph("Model Configuration", heading_style))

        config_data = [
            ['Parameter', 'Value'],
            ['Model Name', model_specs['main'].get('model_name', 'N/A')],
            ['Architecture', model_specs['main'].get('architecture', 'N/A')],
            ['Batch Size', model_specs['main'].get('batch_size', 'N/A')],
            ['Epochs', model_specs['main'].get('epochs', 'N/A')],
        ]

        config_table = Table(config_data, colWidths=[2.5 * inch, 4 * inch])
        config_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))

        elements.append(config_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Key Statistics
        elements.append(Paragraph("Key Training Statistics", heading_style))

        stats_data = [
            ['Metric', 'Value'],
            ['Total Epochs', f"{stats.get('total_epochs', 'N/A')}"],
            ['Best Validation Loss', f"{stats.get('best_val_loss', 0):.4f}"],
            ['Best Epoch', f"{stats.get('best_epoch', 'N/A')}"],
            ['Final Train Loss', f"{stats.get('final_train_loss', 0):.4f}"],
            ['Final Val Loss', f"{stats.get('final_val_loss', 0):.4f}"],
            ['Validation AP', f"{stats.get('val_ap', 0):.3f}"],
            ['Test AP', f"{stats.get('test_ap', 0):.3f}"],
        ]

        stats_table = Table(stats_data, colWidths=[2.5 * inch, 4 * inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ]))

        elements.append(stats_table)
        elements.append(PageBreak())

        # Generate and add plots
        temp_dir = output_dir / "temp_images"
        temp_dir.mkdir(exist_ok=True)

        print("  📈 Generating evaluation metrics plot...")
        elements.append(Paragraph("Evaluation Metrics", heading_style))
        eval_fig = create_evaluation_metrics_plot(eval_metrics, full_size=True)
        eval_img_path = temp_dir / "eval_metrics.png"
        eval_fig.write_image(str(eval_img_path), width=1200, height=800)
        elements.append(Image(str(eval_img_path), width=7 * inch, height=4.67 * inch))
        elements.append(PageBreak())

        print("  📉 Generating training metrics plot...")
        elements.append(Paragraph("Training Metrics Overview", heading_style))
        training_fig = create_interactive_plot(metrics, full_size=True)
        training_img_path = temp_dir / "training_metrics.png"
        training_fig.write_image(str(training_img_path), width=1200, height=800)
        elements.append(Image(str(training_img_path), width=7 * inch, height=4.67 * inch))
        elements.append(PageBreak())

        print("  📊 Generating detailed analysis plot...")
        elements.append(Paragraph("Detailed Training Analysis", heading_style))
        detailed_fig = create_detailed_subplots(metrics, full_size=True)
        detailed_img_path = temp_dir / "detailed_analysis.png"
        detailed_fig.write_image(str(detailed_img_path), width=1200, height=1400)
        elements.append(Image(str(detailed_img_path), width=7 * inch, height=8.17 * inch))

        # Build PDF
        print("  📝 Building PDF document...")
        doc.build(elements)

        # Cleanup temp images
        import shutil
        shutil.rmtree(temp_dir)

        print(f"\n✅ PDF report generated successfully!")
        print(f"   📄 Location: {pdf_path}")
        print(f"   📏 Size: {pdf_path.stat().st_size / 1024:.1f} KB")

        return True

    except Exception as e:
        print(f"\n❌ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_comparison_to_pdf(dataset_configs, output_dir=None):
    """Export comparison visualization to PDF."""

    if not KALEIDO_AVAILABLE or not REPORTLAB_AVAILABLE:
        print("❌ Error: Missing required packages for PDF export")
        print("   Install with: pip install kaleido reportlab")
        return False

    try:
        # Load all datasets
        datasets = []

        for config in dataset_configs:
            try:
                print(f"📖 Reading training log: {config['name']}")
                metrics, eval_metrics = parse_training_log(config['path'])

                log_dir = Path(config['path']).parent
                spec_filepath = log_dir / "training_specs.txt"
                model_specs = parse_model_specs(spec_filepath) if spec_filepath.exists() else {
                    'main': {}, 'hyperparameters': {}
                }

                stats = calculate_statistics(metrics, eval_metrics)

                datasets.append({
                    'name': config['name'],
                    'metrics': metrics,
                    'eval_metrics': eval_metrics,
                    'specs': model_specs,
                    'stats': stats
                })

                print(f"   ✅ Loaded: {config['name']}")

            except Exception as e:
                print(f"   ⚠️ Error loading {config['name']}: {e}")

        if not datasets:
            print("❌ No valid_FINAL_CLEAN_DONISH datasets loaded")
            return False

        # Setup output directory
        if output_dir is None:
            output_dir = Path.cwd() / "training_reports"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"comparison_{timestamp}.pdf"
        pdf_path = output_dir / pdf_filename

        print(f"\n🔄 Generating comparison PDF report...")

        # Create PDF
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=0.5 * inch,
            leftMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch
        )

        elements = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=12,
            spaceBefore=12
        )

        # Title
        elements.append(Paragraph(f"📊 Training Comparison Report ({len(datasets)} Models)", title_style))
        elements.append(Spacer(1, 0.2 * inch))

        # Summary table
        elements.append(Paragraph("Performance Summary", heading_style))

        summary_data = [['Model', 'Best Val Loss', 'Val AP', 'Test AP']]
        for dataset in datasets:
            summary_data.append([
                dataset['name'],
                f"{dataset['stats'].get('best_val_loss', 0):.4f}",
                f"{dataset['stats'].get('val_ap', 0):.3f}",
                f"{dataset['stats'].get('test_ap', 0):.3f}"
            ])

        summary_table = Table(summary_data, colWidths=[2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))

        elements.append(summary_table)
        elements.append(PageBreak())

        # Generate plots
        temp_dir = output_dir / "temp_images"
        temp_dir.mkdir(exist_ok=True)

        print("  📈 Generating evaluation comparison...")
        elements.append(Paragraph("Evaluation Metrics Comparison", heading_style))
        eval_fig, _, _, _ = create_comparison_evaluation_plot(datasets, full_size=True)
        eval_img_path = temp_dir / "eval_comparison.png"
        eval_fig.write_image(str(eval_img_path), width=1200, height=300 * len(datasets))
        elements.append(Image(str(eval_img_path), width=7 * inch, height=min(9 * inch, 1.75 * inch * len(datasets))))
        elements.append(PageBreak())

        print("  📉 Generating training analysis comparison...")
        elements.append(Paragraph("Detailed Training Analysis", heading_style))
        detailed_fig = create_comparison_detailed_subplots(datasets, full_size=True)
        detailed_img_path = temp_dir / "detailed_comparison.png"
        detailed_fig.write_image(str(detailed_img_path), width=1400, height=1600)
        elements.append(Image(str(detailed_img_path), width=7 * inch, height=8 * inch))
        elements.append(PageBreak())

        print("  📊 Generating convergence analysis...")
        elements.append(Paragraph("Convergence Analysis", heading_style))
        convergence_fig = create_comparison_convergence_plot(datasets, full_size=True)
        convergence_img_path = temp_dir / "convergence_comparison.png"
        convergence_fig.write_image(str(convergence_img_path), width=1400, height=1000)
        elements.append(Image(str(convergence_img_path), width=7 * inch, height=5 * inch))
        elements.append(PageBreak())

        print("  ⚡ Generating learning dynamics...")
        elements.append(Paragraph("Learning Dynamics", heading_style))
        learning_fig = create_comparison_learning_dynamics_plot(datasets, full_size=True)
        learning_img_path = temp_dir / "learning_comparison.png"
        learning_fig.write_image(str(learning_img_path), width=1400, height=1000)
        elements.append(Image(str(learning_img_path), width=7 * inch, height=5 * inch))

        # Build PDF
        print("  📝 Building PDF document...")
        doc.build(elements)

        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)

        print(f"\n✅ Comparison PDF report generated successfully!")
        print(f"   📄 Location: {pdf_path}")
        print(f"   📏 Size: {pdf_path.stat().st_size / 1024:.1f} KB")

        return True

    except Exception as e:
        print(f"\n❌ Error generating comparison PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for PDF export."""

    print("Training Log PDF Export")
    print("-" * 40)
    print("1. Export single training log")
    print("2. Export comparison of multiple logs")
    print("-" * 40)

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "2":
        # Multiple logs
        from training_visualizer.ui.dataset_selector import DatasetSelector
        selector = DatasetSelector()
        selector.select_datasets()

        if not selector.datasets:
            print("No datasets selected")
            return

        # Ask for output directory
        output_dir = input("\nEnter output directory (press Enter for default './training_reports'): ").strip()
        if not output_dir:
            output_dir = None

        export_comparison_to_pdf(selector.datasets, output_dir)

    else:
        # Single log
        log_file = input("Enter the path to your log file: ").strip()
        log_file = log_file.strip('"').strip("'")

        if not Path(log_file).exists():
            print(f"❌ Error: File not found: {log_file}")
            return

        # Ask for output directory
        output_dir = input("\nEnter output directory (press Enter for default './training_reports'): ").strip()
        if not output_dir:
            output_dir = None

        export_single_training_to_pdf(log_file, output_dir)

    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()