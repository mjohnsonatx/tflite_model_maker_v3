"""Entry point script for training visualizer."""
import sys
from pathlib import Path

# Ensure the project root is in the path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training_visualizer.core.visualizer import visualize_training
from training_visualizer.core.comparison_table_only import ComparisonTableOnly


def main():
    """Main function."""
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Comparison mode
            visualize_training(compare_mode=True)
        elif sys.argv[1] == "--compare-table-only":
            # Comparison table only mode (for subprocess calls)
            run_comparison_table_only_subprocess()
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
        print("3. Compare training logs (table only)")
        print("-" * 40)

        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == "2":
            visualize_training(compare_mode=True)
        elif choice == "3":
            run_comparison_table_only()
        else:
            log_file = input("Enter the path to your log file: ").strip()
            log_file = log_file.strip('"').strip("'")
            visualize_training(log_file)

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
                print("❌ Please enter a valid number")

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


if __name__ == "__main__":
    main()

