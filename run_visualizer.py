"""Entry point script for training visualizer."""
import sys
from pathlib import Path

# Ensure the project root is in the path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training_visualizer.core.visualizer import visualize_training


def main():
    """Main function."""
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--compare":
            # Comparison mode
            visualize_training(compare_mode=True)
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
        print("-" * 40)

        choice = input("Enter your choice (1 or 2): ").strip()

        if choice == "2":
            visualize_training(compare_mode=True)
        else:
            log_file = input("Enter the path to your log file: ").strip()
            log_file = log_file.strip('"').strip("'")
            visualize_training(log_file)

        input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()