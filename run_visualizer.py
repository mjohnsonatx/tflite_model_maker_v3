"""Entry point script for training visualizer."""
import sys
from pathlib import Path

# Ensure the project root is in the path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training_visualizer.core.comparison_table_only import ComparisonTableOnly
from training_visualizer.core.csv_exporter import export_training_data_csv


def main():
    """Main function."""
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Comparison mode
            _visualize_training(compare_mode=True)
        elif sys.argv[1] == "--compare-table-only":
            # Comparison table only mode (for subprocess calls)
            run_comparison_table_only_subprocess()
        elif sys.argv[1] == "--export-csv":
            # CSV export mode
            run_csv_export_subprocess()
        else:
            # Single file mode
            log_file = sys.argv[1].strip('"').strip("'")
            _visualize_training(log_file)
    else:
        # Interactive mode
        print("Training Log Visualizer")
        print("-" * 40)
        print("1. Visualize single training log")
        print("2. Compare multiple training logs")
        print("3. Compare training logs (table only)")
        print("4. Export visualizer data to CSV")
        print("-" * 40)

        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

        if choice == "2":
            _visualize_training(compare_mode=True)
        elif choice == "3":
            run_comparison_table_only()
        elif choice == "4":
            run_csv_export()
        else:
            log_file = input("Enter the path to your log file: ").strip()
            log_file = log_file.strip('"').strip("'")
            _visualize_training(log_file)

        input("\nPress Enter to exit...")


def run_comparison_table_only():
    """Run the comparison table only mode (interactive)."""
    print("\nComparison Table Mode")
    print("-" * 40)
    print("1. Select individual training logs")
    print("2. Compare recent models (you choose how many)")
    print("-" * 40)

    choice = input("Enter your choice (1 or 2): ").strip()

    comparison = ComparisonTableOnly()

    if choice == "2":
        # Run Recent Models with user-specified count
        base_path = input("Enter the path to models directory: ").strip()
        base_path = base_path.strip('"').strip("'")

        # Get available model count
        available_count = ComparisonTableOnly.get_available_model_count(base_path)
        if available_count == 0:
            print("❌ No models found in the specified directory")
            return

        print(f"📊 Found {available_count} models in directory")

        # Get user input for number of models
        while True:
            try:
                user_input = input(
                    f"How many recent models to compare? (1-{available_count}, or press Enter for 10): ").strip()
                if not user_input:
                    limit = min(10, available_count)
                    break
                else:
                    limit = int(user_input)
                    if 1 <= limit <= available_count:
                        break
                    else:
                        print(f"❌ Please enter a number between 1 and {available_count}")
            except ValueError:
                print("❌ Please enter a valid_FINAL_CLEAN_DONISH number")

        print(f"🔄 Loading {limit} most recent models...")
        model_configs = ComparisonTableOnly.find_recent_models(base_path, limit=limit)

        if not model_configs:
            print("❌ No completed models found in the specified directory")
            return

        print(f"\n✅ Found {len(model_configs)} completed models:")
        for i, config in enumerate(model_configs, 1):
            print(f"  {i}. {config['name']}")

        if comparison.visualize_comparison_table_only(model_configs):
            print("✅ Comparison table generated successfully")
        else:
            print("❌ Failed to generate comparison table")

    else:
        # Select individual models
        from training_visualizer.ui.table_selector import TableSelector
        selector = TableSelector()
        selector.select_datasets()

        if not selector.datasets:
            print("No datasets selected")
            return

        if comparison.visualize_comparison_table_only(selector.datasets):
            print("✅ Comparison table generated successfully")
        else:
            print("❌ Failed to generate comparison table")


def run_comparison_table_only_subprocess():
    """Run comparison table only mode for subprocess calls."""
    # This could be extended to accept additional arguments for specific model paths
    # For now, it defaults to interactive mode
    comparison = ComparisonTableOnly()

    # Check if there are additional arguments for model paths
    if len(sys.argv) > 2:
        # Additional arguments provided - treat as model paths
        model_configs = []
        for i, path in enumerate(sys.argv[2:], 1):
            model_path = path.strip('"').strip("'")
            model_name = Path(model_path).parent.name
            model_configs.append({'path': model_path, 'name': model_name})

        comparison.visualize_comparison_table_only(model_configs)
    else:
        # No additional arguments - use interactive mode
        run_comparison_table_only()


def run_csv_export():
    """Run CSV export mode (interactive)."""
    print("\nCSV Export Mode")
    print("-" * 40)
    print("1. Export a single training log")
    print("2. Select multiple training logs")
    print("3. Export recent completed models")
    print("-" * 40)

    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == "2":
        from training_visualizer.ui.table_selector import TableSelector
        selector = TableSelector()
        selector.select_datasets()

        if not selector.datasets:
            print("No datasets selected")
            return

        model_configs = selector.datasets

    elif choice == "3":
        base_path = input("Enter the path to models directory: ").strip()
        base_path = base_path.strip('"').strip("'")

        available_count = ComparisonTableOnly.get_available_model_count(base_path)
        if available_count == 0:
            print("No models found in the specified directory")
            return

        while True:
            try:
                user_input = input(
                    f"How many recent models to export? (1-{available_count}, or press Enter for 10): "
                ).strip()
                limit = min(10, available_count) if not user_input else int(user_input)
                if 1 <= limit <= available_count:
                    break
                print(f"Please enter a number between 1 and {available_count}")
            except ValueError:
                print("Please enter a valid number")

        model_configs = ComparisonTableOnly.find_recent_models(base_path, limit=limit)
        if not model_configs:
            print("No completed models found in the specified directory")
            return

    else:
        log_file = input("Enter the path to your log file: ").strip()
        log_file = log_file.strip('"').strip("'")
        model_configs = [{'path': log_file, 'name': Path(log_file).parent.name or Path(log_file).stem}]

    output_path = input("Enter output CSV path (or press Enter for default): ").strip()
    output_path = output_path.strip('"').strip("'") or None
    _export_csv(model_configs, output_path)


def run_csv_export_subprocess():
    """Run CSV export mode from command line arguments."""
    args = [arg.strip('"').strip("'") for arg in sys.argv[2:]]

    if not args:
        run_csv_export()
        return

    output_path = None
    if "--out" in args:
        out_index = args.index("--out")
        if out_index + 1 >= len(args):
            print("Usage: python run_visualizer.py --export-csv <log_file> [output.csv]")
            print("       python run_visualizer.py --export-csv --compare <log1> <log2> ... --out output.csv")
            print("       python run_visualizer.py --export-csv --recent <models_dir> [limit] --out output.csv")
            return
        output_path = args[out_index + 1]
        del args[out_index:out_index + 2]

    if args and args[0] == "--recent":
        if len(args) < 2:
            print("Usage: python run_visualizer.py --export-csv --recent <models_dir> [limit] --out output.csv")
            return

        base_path = args[1]
        try:
            limit = int(args[2]) if len(args) > 2 else 10
        except ValueError:
            print("Recent model limit must be a number")
            return

        model_configs = ComparisonTableOnly.find_recent_models(base_path, limit=limit)

    elif args and args[0] == "--compare":
        log_files = args[1:]
        if not log_files:
            print("Usage: python run_visualizer.py --export-csv --compare <log1> <log2> ... --out output.csv")
            return

        model_configs = [
            {'path': log_file, 'name': Path(log_file).parent.name or Path(log_file).stem}
            for log_file in log_files
        ]

    else:
        log_file = args[0]
        if len(args) > 1 and output_path is None:
            output_path = args[1]
        model_configs = [{'path': log_file, 'name': Path(log_file).parent.name or Path(log_file).stem}]

    if not model_configs:
        print("No training logs found for CSV export")
        return

    _export_csv(model_configs, output_path)


def _export_csv(model_configs, output_path=None):
    """Export the provided training log configs and print the result."""
    try:
        csv_path, row_count = export_training_data_csv(model_configs, output_path)
        print(f"\nCSV export complete: {csv_path}")
        print(f"Rows written: {row_count}")
    except Exception as e:
        print(f"\nError exporting CSV: {e}")


def _visualize_training(*args, **kwargs):
    """Import plotting code only when visualization is requested."""
    from training_visualizer.core.visualizer import visualize_training
    return visualize_training(*args, **kwargs)


if __name__ == "__main__":
    main()

