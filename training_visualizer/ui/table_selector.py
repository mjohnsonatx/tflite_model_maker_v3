"""Table dataset selection GUI."""
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
from typing import List, Dict, Any


class TableSelector:
    """GUI for selecting multiple datasets to compare in table-only mode."""

    def __init__(self) -> None:
        """Initialize the table selector."""
        self.datasets: List[Dict[str, Any]] = []
        self.root: tk.Tk = None

    def select_datasets(self) -> None:
        """Open GUI for dataset selection."""
        self.root = tk.Tk()
        self.root.title("Select Training Logs for Table Comparison")
        self.root.geometry("700x450")

        # Header
        header = tk.Label(
            self.root,
            text="Select Training Logs for Table Comparison",
            font=("Arial", 16, "bold")
        )
        header.pack(pady=10)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="Add up to 10 training log files to compare specifications and evaluation metrics",
            font=("Arial", 10)
        )
        instructions.pack()

        # Dataset list frame
        list_frame = tk.Frame(self.root)
        list_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Listbox for datasets
        self.listbox = tk.Listbox(list_frame, height=12)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        # Button frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)

        # Add dataset button
        add_btn = tk.Button(
            button_frame,
            text="Add Dataset",
            command=self.add_dataset,
            bg="#4CAF50",
            fg="white",
            padx=20,
            pady=5
        )
        add_btn.pack(side=tk.LEFT, padx=5)

        # Remove dataset button
        remove_btn = tk.Button(
            button_frame,
            text="Remove Selected",
            command=self.remove_dataset,
            bg="#f44336",
            fg="white",
            padx=20,
            pady=5
        )
        remove_btn.pack(side=tk.LEFT, padx=5)

        # Clear all button
        clear_btn = tk.Button(
            button_frame,
            text="Clear All",
            command=self.clear_datasets,
            bg="#FF9800",
            fg="white",
            padx=20,
            pady=5
        )
        clear_btn.pack(side=tk.LEFT, padx=5)

        # Generate Table button
        self.generate_btn = tk.Button(
            button_frame,
            text="Generate Table",
            command=self.generate_table,
            bg="#2196F3",
            fg="white",
            padx=30,
            pady=5,
            state=tk.DISABLED
        )
        self.generate_btn.pack(side=tk.LEFT, padx=20)

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="No datasets selected",
            font=("Arial", 10),
            fg="gray"
        )
        self.status_label.pack(pady=5)

        self.root.mainloop()

    def add_dataset(self) -> None:
        """Add a dataset via file dialog."""
        if len(self.datasets) >= 10:
            messagebox.showwarning("Limit Reached", "Maximum of 10 datasets can be compared")
            return

        filepath = filedialog.askopenfilename(
            title="Select Training Log File",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filepath:
            # Use the parent folder name as the model name (as requested)
            parent_folder_name = Path(filepath).parent.name
            
            # Check if this model name already exists
            existing_names = [d['name'] for d in self.datasets]
            if parent_folder_name in existing_names:
                # Add a suffix to make it unique
                counter = 1
                unique_name = f"{parent_folder_name}_{counter}"
                while unique_name in existing_names:
                    counter += 1
                    unique_name = f"{parent_folder_name}_{counter}"
                model_name = unique_name
            else:
                model_name = parent_folder_name

            self.datasets.append({'path': filepath, 'name': model_name})
            self.listbox.insert(tk.END, f"{model_name} - {Path(filepath).name}")
            self.update_status()

    def remove_dataset(self) -> None:
        """Remove selected dataset."""
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            self.listbox.delete(idx)
            del self.datasets[idx]
            self.update_status()

    def clear_datasets(self) -> None:
        """Clear all datasets."""
        self.listbox.delete(0, tk.END)
        self.datasets = []
        self.update_status()

    def update_status(self) -> None:
        """Update status label and button states."""
        count = len(self.datasets)
        if count == 0:
            self.status_label.config(text="No datasets selected", fg="gray")
            self.generate_btn.config(state=tk.DISABLED)
        else:
            self.status_label.config(text=f"{count} dataset(s) selected", fg="green")
            self.generate_btn.config(state=tk.NORMAL)

    def generate_table(self) -> None:
        """Start table generation."""
        self.root.quit()
        self.root.destroy()
