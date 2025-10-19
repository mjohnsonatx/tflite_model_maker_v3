"""Dataset selection GUI."""
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path


class DatasetSelector:
    """GUI for selecting multiple datasets to compare."""

    def __init__(self):
        self.datasets = []
        self.root = None

    def select_datasets(self):
        """Open GUI for dataset selection."""
        self.root = tk.Tk()
        self.root.title("Select Training Logs to Compare")
        self.root.geometry("600x400")

        # Header
        header = tk.Label(
            self.root,
            text="Select Training Logs for Comparison",
            font=("Arial", 16, "bold")
        )
        header.pack(pady=10)

        # Instructions
        instructions = tk.Label(
            self.root,
            text="Add up to 3 training log files to compare side-by-side",
            font=("Arial", 10)
        )
        instructions.pack()

        # Dataset list frame
        list_frame = tk.Frame(self.root)
        list_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        # Listbox for datasets
        self.listbox = tk.Listbox(list_frame, height=10)
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

        # Visualize button
        self.visualize_btn = tk.Button(
            button_frame,
            text="Visualize",
            command=self.visualize,
            bg="#2196F3",
            fg="white",
            padx=30,
            pady=5,
            state=tk.DISABLED
        )
        self.visualize_btn.pack(side=tk.LEFT, padx=20)

        # Status label
        self.status_label = tk.Label(
            self.root,
            text="No datasets selected",
            font=("Arial", 10),
            fg="gray"
        )
        self.status_label.pack(pady=5)

        self.root.mainloop()

    def add_dataset(self):
        """Add a dataset via file dialog."""
        if len(self.datasets) >= 3:
            messagebox.showwarning("Limit Reached", "Maximum of 3 datasets can be compared")
            return

        filepath = filedialog.askopenfilename(
            title="Select Training Log File",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
        )

        if filepath:
            # Get a custom name for the dataset
            name = simpledialog.askstring(
                "Dataset Name",
                f"Enter a name for this dataset:",
                initialvalue=Path(filepath).stem
            )

            if name:
                self.datasets.append({'path': filepath, 'name': name})
                self.listbox.insert(tk.END, f"{name} - {Path(filepath).name}")
                self.update_status()

    def remove_dataset(self):
        """Remove selected dataset."""
        selection = self.listbox.curselection()
        if selection:
            idx = selection[0]
            self.listbox.delete(idx)
            del self.datasets[idx]
            self.update_status()

    def clear_datasets(self):
        """Clear all datasets."""
        self.listbox.delete(0, tk.END)
        self.datasets = []
        self.update_status()

    def update_status(self):
        """Update status label and button states."""
        count = len(self.datasets)
        if count == 0:
            self.status_label.config(text="No datasets selected", fg="gray")
            self.visualize_btn.config(state=tk.DISABLED)
        else:
            self.status_label.config(text=f"{count} dataset(s) selected", fg="green")
            self.visualize_btn.config(state=tk.NORMAL)

    def visualize(self):
        """Start visualization."""
        self.root.quit()
        self.root.destroy()