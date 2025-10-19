"""Main entry point for training visualizer."""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from training_visualizer
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

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


