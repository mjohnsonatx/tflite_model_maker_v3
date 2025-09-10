import os
import random
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np

class FileOperationsTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.operation_thread = None
        self.stop_flag = threading.Event()
        self.setup_ui()

    def setup_ui(self):
        # Main content frame - directly in parent, no canvas/scrollbar
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Add progress bar at the top
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_frame.pack(fill="x", pady=(0, 20))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill="x", pady=(0, 5))

        self.progress_label = ttk.Label(self.progress_frame, text="Ready")
        self.progress_label.pack()

        # Create two-column layout
        columns_frame = ttk.Frame(main_frame)
        columns_frame.pack(fill="both", expand=True)

        # Left column
        left_column = ttk.Frame(columns_frame)
        left_column.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 10))

        # Right column
        right_column = ttk.Frame(columns_frame)
        right_column.pack(side=tk.LEFT, fill="both", expand=True, padx=(10, 0))

        # Section 1: Directory Comparison (Left column)
        ttk.Label(left_column, text="Directory Comparison",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        compare_frame = ttk.LabelFrame(left_column, text="Compare Two Directories", padding="10")
        compare_frame.pack(fill="x", pady=(0, 20))

        # Directory 1
        dir1_frame = ttk.Frame(compare_frame)
        dir1_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(dir1_frame, text="Directory 1:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.dir1_var = tk.StringVar()
        ttk.Entry(dir1_frame, textvariable=self.dir1_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                         padx=(0, 5))
        ttk.Button(dir1_frame, text="Browse",
                   command=lambda: self.browse_directory(self.dir1_var)).grid(row=0, column=2)
        dir1_frame.columnconfigure(1, weight=1)

        # Directory 2
        dir2_frame = ttk.Frame(compare_frame)
        dir2_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(dir2_frame, text="Directory 2:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.dir2_var = tk.StringVar()
        ttk.Entry(dir2_frame, textvariable=self.dir2_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                         padx=(0, 5))
        ttk.Button(dir2_frame, text="Browse",
                   command=lambda: self.browse_directory(self.dir2_var)).grid(row=0, column=2)
        dir2_frame.columnconfigure(1, weight=1)

        self.compare_btn = ttk.Button(compare_frame, text="Compare Directories",
                                      command=self.compare_directories)
        self.compare_btn.pack(pady=(10, 0))

        # Section 3: Split Dataset (Left column)
        ttk.Label(left_column, text="Split Dataset",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(20, 10))

        split_frame = ttk.LabelFrame(left_column, text="Split into Train/Test/Validation", padding="10")
        split_frame.pack(fill="x", pady=(0, 20))

        # Source directory
        source_frame = ttk.Frame(split_frame)
        source_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(source_frame, text="Source:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.split_source_var = tk.StringVar()
        ttk.Entry(source_frame, textvariable=self.split_source_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                                   padx=(0, 5))
        ttk.Button(source_frame, text="Browse",
                   command=lambda: self.browse_directory(self.split_source_var)).grid(row=0, column=2)
        source_frame.columnconfigure(1, weight=1)

        # Output base directory
        output_frame = ttk.Frame(split_frame)
        output_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(output_frame, text="Output:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.split_output_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.split_output_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                                   padx=(0, 5))
        ttk.Button(output_frame, text="Browse",
                   command=lambda: self.browse_directory(self.split_output_var)).grid(row=0, column=2)
        output_frame.columnconfigure(1, weight=1)

        # Split ratios
        ratio_frame = ttk.Frame(split_frame)
        ratio_frame.pack(fill="x", pady=(0, 10))

        ttk.Label(ratio_frame, text="Split Ratios:").pack(anchor=tk.W)

        ratio_input_frame = ttk.Frame(ratio_frame)
        ratio_input_frame.pack(fill="x", pady=(5, 0))

        ttk.Label(ratio_input_frame, text="Train:").grid(row=0, column=0, padx=(20, 5))
        self.train_ratio_var = tk.StringVar(value="70")
        ttk.Entry(ratio_input_frame, textvariable=self.train_ratio_var, width=5).grid(row=0, column=1)
        ttk.Label(ratio_input_frame, text="%").grid(row=0, column=2, padx=(0, 15))

        ttk.Label(ratio_input_frame, text="Test:").grid(row=0, column=3, padx=(0, 5))
        self.test_ratio_var = tk.StringVar(value="20")
        ttk.Entry(ratio_input_frame, textvariable=self.test_ratio_var, width=5).grid(row=0, column=4)
        ttk.Label(ratio_input_frame, text="%").grid(row=0, column=5, padx=(0, 15))

        ttk.Label(ratio_input_frame, text="Valid:").grid(row=0, column=6, padx=(0, 5))
        self.valid_ratio_var = tk.StringVar(value="10")
        ttk.Entry(ratio_input_frame, textvariable=self.valid_ratio_var, width=5).grid(row=0, column=7)
        ttk.Label(ratio_input_frame, text="%").grid(row=0, column=8)

        self.shuffle_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(split_frame, text="Shuffle files before splitting",
                        variable=self.shuffle_var).pack(pady=(5, 0))

        self.split_btn = ttk.Button(split_frame, text="Split Dataset",
                                    command=self.split_dataset)
        self.split_btn.pack(pady=(10, 0))

        # NEW SECTION: Add Prefix to Files (Left column)
        ttk.Label(left_column, text="Add Prefix to Files",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(20, 10))

        prefix_frame = ttk.LabelFrame(left_column, text="Add Custom Prefix to All Files", padding="10")
        prefix_frame.pack(fill="x", pady=(0, 20))

        # Directory selection
        prefix_dir_frame = ttk.Frame(prefix_frame)
        prefix_dir_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(prefix_dir_frame, text="Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.prefix_dir_var = tk.StringVar()
        ttk.Entry(prefix_dir_frame, textvariable=self.prefix_dir_var, width=30).grid(row=0, column=1,
                                                                                     sticky=(tk.W, tk.E),
                                                                                     padx=(0, 5))
        ttk.Button(prefix_dir_frame, text="Browse",
                   command=lambda: self.browse_directory(self.prefix_dir_var)).grid(row=0, column=2)
        prefix_dir_frame.columnconfigure(1, weight=1)

        # Prefix input
        prefix_input_frame = ttk.Frame(prefix_frame)
        prefix_input_frame.pack(fill="x", pady=(5, 10))
        ttk.Label(prefix_input_frame, text="Prefix:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.prefix_var = tk.StringVar(value="highres_")
        ttk.Entry(prefix_input_frame, textvariable=self.prefix_var, width=20).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                                   padx=(0, 5))
        ttk.Label(prefix_input_frame, text="(e.g., 'highres_' or 'v2_')").grid(row=0, column=2, sticky=tk.W)
        prefix_input_frame.columnconfigure(1, weight=1)

        # Options
        self.prefix_copy_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(prefix_frame, text="Copy files instead of renaming (preserves originals)",
                        variable=self.prefix_copy_var).pack(anchor=tk.W)

        self.prefix_include_xml_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prefix_frame, text="Also rename/copy associated XML files",
                        variable=self.prefix_include_xml_var).pack(anchor=tk.W, pady=(5, 10))

        # Preview and execute buttons
        button_frame = ttk.Frame(prefix_frame)
        button_frame.pack(fill="x")

        ttk.Button(button_frame, text="Preview Changes",
                   command=self.preview_prefix_changes).pack(side=tk.LEFT, padx=(0, 10))

        self.prefix_btn = ttk.Button(button_frame, text="Add Prefix",
                                     command=self.add_prefix_to_files)
        self.prefix_btn.pack(side=tk.LEFT)

        # Section 2: Combine Folders/Directories (Right column)
        ttk.Label(right_column, text="Combine Folders/Directories",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        # Notebook for combine options
        combine_notebook = ttk.Notebook(right_column)
        combine_notebook.pack(fill="x", pady=(0, 20))

        # Tab 1: Combine Two Folders
        two_folders_frame = ttk.Frame(combine_notebook)
        combine_notebook.add(two_folders_frame, text="Two Folders")

        combine_frame = ttk.Frame(two_folders_frame, padding="10")
        combine_frame.pack(fill="x")

        # First directory
        orig_frame = ttk.Frame(combine_frame)
        orig_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(orig_frame, text="Folder 1:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.orig_dir_var = tk.StringVar()
        ttk.Entry(orig_frame, textvariable=self.orig_dir_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                             padx=(0, 5))
        ttk.Button(orig_frame, text="Browse",
                   command=lambda: self.browse_directory(self.orig_dir_var)).grid(row=0, column=2)
        orig_frame.columnconfigure(1, weight=1)

        # Second directory
        aug_frame = ttk.Frame(combine_frame)
        aug_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(aug_frame, text="Folder 2:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.aug_dir_var = tk.StringVar()
        ttk.Entry(aug_frame, textvariable=self.aug_dir_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                           padx=(0, 5))
        ttk.Button(aug_frame, text="Browse",
                   command=lambda: self.browse_directory(self.aug_dir_var)).grid(row=0, column=2)
        aug_frame.columnconfigure(1, weight=1)

        # Output directory
        comb_frame = ttk.Frame(combine_frame)
        comb_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(comb_frame, text="Output:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.comb_dir_var = tk.StringVar()
        ttk.Entry(comb_frame, textvariable=self.comb_dir_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                             padx=(0, 5))
        ttk.Button(comb_frame, text="Browse",
                   command=lambda: self.browse_directory(self.comb_dir_var)).grid(row=0, column=2)
        comb_frame.columnconfigure(1, weight=1)

        self.combine_btn = ttk.Button(combine_frame, text="Combine Two Folders",
                                      command=self.combine_datasets)
        self.combine_btn.pack(pady=(10, 0))

        # Tab 2: Combine All Folders in Directory
        all_folders_frame = ttk.Frame(combine_notebook)
        combine_notebook.add(all_folders_frame, text="All Folders")

        batch_frame = ttk.Frame(all_folders_frame, padding="10")
        batch_frame.pack(fill="x")

        # Parent directory containing folders
        parent_frame = ttk.Frame(batch_frame)
        parent_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(parent_frame, text="Parent Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.parent_dir_var = tk.StringVar()
        ttk.Entry(parent_frame, textvariable=self.parent_dir_var, width=25).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                                 padx=(0, 5))
        ttk.Button(parent_frame, text="Browse",
                   command=lambda: self.browse_directory(self.parent_dir_var)).grid(row=0, column=2)
        parent_frame.columnconfigure(1, weight=1)

        # Output directory for batch
        batch_output_frame = ttk.Frame(batch_frame)
        batch_output_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(batch_output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.batch_output_var = tk.StringVar()
        ttk.Entry(batch_output_frame, textvariable=self.batch_output_var, width=25).grid(row=0, column=1,
                                                                                         sticky=(tk.W, tk.E),
                                                                                         padx=(0, 5))
        ttk.Button(batch_output_frame, text="Browse",
                   command=lambda: self.browse_directory(self.batch_output_var)).grid(row=0, column=2)
        batch_output_frame.columnconfigure(1, weight=1)

        # Options
        options_frame = ttk.Frame(batch_frame)
        options_frame.pack(fill="x", pady=(0, 10))

        self.include_parent_files_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Include files from parent directory",
                        variable=self.include_parent_files_var).pack(anchor=tk.W)

        self.recursive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Include subdirectories recursively",
                        variable=self.recursive_var).pack(anchor=tk.W)

        # Preview button
        ttk.Button(batch_frame, text="Preview Folders",
                   command=self.preview_batch_folders).pack(pady=(5, 0))

        self.batch_combine_btn = ttk.Button(batch_frame, text="Combine All Folders",
                                            command=self.batch_combine_folders)
        self.batch_combine_btn.pack(pady=(10, 0))

        # Section 4: Move Percentage of Files (Right column)
        ttk.Label(right_column, text="Move/Copy Files",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(20, 10))

        move_frame = ttk.LabelFrame(right_column, text="Move or Copy Percentage of Files", padding="10")
        move_frame.pack(fill="x", pady=(0, 20))

        # Source directory
        move_source_frame = ttk.Frame(move_frame)
        move_source_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(move_source_frame, text="Source:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.move_source_var = tk.StringVar()
        ttk.Entry(move_source_frame, textvariable=self.move_source_var, width=30).grid(row=0, column=1,
                                                                                       sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(move_source_frame, text="Browse",
                   command=lambda: self.browse_directory(self.move_source_var)).grid(row=0, column=2)
        move_source_frame.columnconfigure(1, weight=1)

        # Destination directory
        move_dest_frame = ttk.Frame(move_frame)
        move_dest_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(move_dest_frame, text="Destination:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.move_dest_var = tk.StringVar()
        ttk.Entry(move_dest_frame, textvariable=self.move_dest_var, width=30).grid(row=0, column=1, sticky=(tk.W, tk.E),
                                                                                   padx=(0, 5))
        ttk.Button(move_dest_frame, text="Browse",
                   command=lambda: self.browse_directory(self.move_dest_var)).grid(row=0, column=2)
        move_dest_frame.columnconfigure(1, weight=1)

        # Percentage
        percent_frame = ttk.Frame(move_frame)
        percent_frame.pack(fill="x", pady=(10, 10))
        ttk.Label(percent_frame, text="Percentage to process:").pack(side=tk.LEFT, padx=(0, 5))
        self.move_percent_var = tk.StringVar(value="50")
        ttk.Entry(percent_frame, textvariable=self.move_percent_var, width=5).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(percent_frame, text="%").pack(side=tk.LEFT)

        # Operation type
        operation_frame = ttk.Frame(move_frame)
        operation_frame.pack(fill="x")
        self.move_copy_var = tk.StringVar(value="move")
        ttk.Radiobutton(operation_frame, text="Move files",
                        variable=self.move_copy_var, value="move").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(operation_frame, text="Copy files",
                        variable=self.move_copy_var, value="copy").pack(side=tk.LEFT)

        self.move_btn = ttk.Button(move_frame, text="Execute",
                                   command=self.move_files)
        self.move_btn.pack(pady=(10, 0))

        # NEW SECTION: Shuffle Files (Right column)
        ttk.Label(right_column, text="Shuffle Files",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor=tk.W, pady=(20, 10))

        shuffle_frame = ttk.LabelFrame(right_column, text="Randomly Shuffle Files in Directory", padding="10")
        shuffle_frame.pack(fill="x", pady=(0, 20))

        # Directory selection
        shuffle_dir_frame = ttk.Frame(shuffle_frame)
        shuffle_dir_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(shuffle_dir_frame, text="Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.shuffle_dir_var = tk.StringVar()
        ttk.Entry(shuffle_dir_frame, textvariable=self.shuffle_dir_var, width=30).grid(row=0, column=1,
                                                                                       sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(shuffle_dir_frame, text="Browse",
                   command=lambda: self.browse_directory(self.shuffle_dir_var)).grid(row=0, column=2)
        shuffle_dir_frame.columnconfigure(1, weight=1)

        # Shuffle options
        shuffle_options_frame = ttk.Frame(shuffle_frame)
        shuffle_options_frame.pack(fill="x", pady=(0, 10))

        self.shuffle_copy_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(shuffle_options_frame, text="Copy to new directory (preserves original order)",
                        variable=self.shuffle_copy_var, command=self.toggle_shuffle_output).pack(anchor=tk.W)

        # Output directory (only shown when copy is selected)
        self.shuffle_output_frame = ttk.Frame(shuffle_frame)
        shuffle_output_label = ttk.Label(self.shuffle_output_frame, text="Output:")
        shuffle_output_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.shuffle_output_var = tk.StringVar()
        shuffle_output_entry = ttk.Entry(self.shuffle_output_frame, textvariable=self.shuffle_output_var, width=30)
        shuffle_output_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        shuffle_output_btn = ttk.Button(self.shuffle_output_frame, text="Browse",
                                        command=lambda: self.browse_directory(self.shuffle_output_var))
        shuffle_output_btn.grid(row=0, column=2)
        self.shuffle_output_frame.columnconfigure(1, weight=1)
        self.shuffle_output_frame.pack(fill="x", pady=(5, 10))

        # Seed option
        seed_frame = ttk.Frame(shuffle_frame)
        seed_frame.pack(fill="x", pady=(0, 10))
        self.use_seed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(seed_frame, text="Use seed for reproducible shuffle:",
                        variable=self.use_seed_var, command=self.toggle_seed_entry).pack(side=tk.LEFT)
        self.seed_var = tk.StringVar(value="42")
        self.seed_entry = ttk.Entry(seed_frame, textvariable=self.seed_var, width=10, state='disabled')
        self.seed_entry.pack(side=tk.LEFT, padx=(10, 0))

        # Info label
        info_label = ttk.Label(shuffle_frame,
                               text="Note: Shuffling will randomize file order while maintaining image-XML pairs",
                               foreground="gray")
        info_label.pack(pady=(5, 10))

        # Buttons
        button_frame = ttk.Frame(shuffle_frame)
        button_frame.pack(fill="x")

        ttk.Button(button_frame, text="Preview Shuffle",
                   command=self.preview_shuffle).pack(side=tk.LEFT, padx=(0, 10))

        self.shuffle_files_btn = ttk.Button(button_frame, text="Shuffle Files",
                                            command=self.shuffle_files)
        self.shuffle_files_btn.pack(side=tk.LEFT)

        # Stop button (initially hidden)
        self.stop_btn = ttk.Button(main_frame, text="Stop Operation",
                                   command=self.stop_operation)

    # Rest of the methods remain the same...
    def toggle_shuffle_output(self):
        if self.shuffle_copy_var.get():
            self.shuffle_output_frame.pack(fill="x", pady=(5, 10))
        else:
            self.shuffle_output_frame.pack_forget()

    def toggle_seed_entry(self):
        if self.use_seed_var.get():
            self.seed_entry.config(state='normal')
        else:
            self.seed_entry.config(state='disabled')

    def preview_shuffle(self):
        directory = self.shuffle_dir_var.get()
        if not directory:
            messagebox.showwarning("Warning", "Please select a directory first.")
            return

        try:
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = [f for f in os.listdir(directory)
                           if Path(f).suffix.lower() in image_extensions]

            if not image_files:
                messagebox.showinfo("Info", "No image files found in the selected directory.")
                return

            # Create a copy for preview
            preview_files = image_files.copy()

            # Apply seed if specified
            if self.use_seed_var.get():
                try:
                    seed = int(self.seed_var.get())
                    random.seed(seed)
                    np.random.seed(seed)
                except ValueError:
                    messagebox.showerror("Error", "Invalid seed value. Please enter an integer.")
                    return

            # Shuffle the preview
            random.shuffle(preview_files)

            # Show preview window
            preview_window = tk.Toplevel(self.parent)
            preview_window.title("Shuffle Preview")
            preview_window.geometry("600x500")

            # Create scrolled text widget
            text_widget = scrolledtext.ScrolledText(preview_window, wrap=tk.WORD, width=70, height=25)
            text_widget.pack(padx=10, pady=10, fill='both', expand=True)

            # Add preview information
            text_widget.insert(tk.END, f"Directory: {directory}\n")
            text_widget.insert(tk.END, f"Total files: {len(image_files)}\n")
            if self.use_seed_var.get():
                text_widget.insert(tk.END, f"Seed: {self.seed_var.get()}\n")
            text_widget.insert(tk.END, "\n" + "=" * 60 + "\n\n")

            # Show before and after (first 10 files)
            text_widget.insert(tk.END, "ORIGINAL ORDER → SHUFFLED ORDER\n")
            text_widget.insert(tk.END, "-" * 60 + "\n")

            for i in range(min(10, len(image_files))):
                text_widget.insert(tk.END, f"{i + 1:3d}. {image_files[i][:30]:30s} → {preview_files[i][:30]}\n")

            if len(image_files) > 10:
                text_widget.insert(tk.END, f"\n... and {len(image_files) - 10} more files\n")

            text_widget.config(state='disabled')

            ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Error previewing shuffle: {str(e)}")

    def shuffle_files(self):
        directory = self.shuffle_dir_var.get()

        if not directory:
            messagebox.showwarning("Warning", "Please select a directory.")
            return

        if self.shuffle_copy_var.get():
            output_dir = self.shuffle_output_var.get()
            if not output_dir:
                messagebox.showwarning("Warning", "Please select an output directory.")
                return
        else:
            # Confirm in-place shuffle
            if not messagebox.askyesno("Confirm",
                                       "Are you sure you want to shuffle files in-place?\n\n"
                                       "This will permanently change the file order in the directory.\n"
                                       "Consider using 'Copy to new directory' option to preserve original order."):
                return

        self.stop_flag.clear()
        self.operation_thread = threading.Thread(
            target=self._shuffle_thread,
            args=(directory,),
            daemon=True
        )
        self.operation_thread.start()

    def _shuffle_thread(self, directory):
        try:
            self.set_buttons_state(False)
            self.update_progress(0, "Preparing to shuffle files...")

            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = [f for f in os.listdir(directory)
                           if Path(f).suffix.lower() in image_extensions]

            if not image_files:
                self.log_queue.put("No image files found in directory.")
                return

            self.log_queue.put(f"\n=== Shuffling Files ===")
            self.log_queue.put(f"Directory: {directory}")
            self.log_queue.put(f"Found {len(image_files)} image files")

            # Apply seed if specified
            if self.use_seed_var.get():
                try:
                    seed = int(self.seed_var.get())
                    random.seed(seed)
                    np.random.seed(seed)
                    self.log_queue.put(f"Using seed: {seed}")
                except ValueError:
                    self.log_queue.put("Invalid seed value, using random shuffle")

            # Create shuffled list
            shuffled_files = image_files.copy()
            random.shuffle(shuffled_files)

            if self.shuffle_copy_var.get():
                # Copy to new directory
                output_dir = self.shuffle_output_var.get()
                os.makedirs(output_dir, exist_ok=True)
                self.log_queue.put(f"Copying shuffled files to: {output_dir}")

                copied_images = 0
                copied_xmls = 0

                for i, (original_name, new_position) in enumerate(zip(image_files, shuffled_files)):
                    if self.stop_flag.is_set():
                        break

                    # Generate new filename with position prefix
                    new_name = f"{i + 1:05d}_{new_position}"

                    # Copy image
                    src_img = os.path.join(directory, new_position)
                    dst_img = os.path.join(output_dir, new_name)
                    shutil.copy2(src_img, dst_img)
                    copied_images += 1

                    # Copy XML if exists
                    xml_file = Path(new_position).stem + '.xml'
                    src_xml = os.path.join(directory, xml_file)
                    if os.path.exists(src_xml):
                        new_xml_name = f"{i + 1:05d}_{xml_file}"
                        dst_xml = os.path.join(output_dir, new_xml_name)
                        shutil.copy2(src_xml, dst_xml)
                        copied_xmls += 1

                    # Update progress
                    progress = (i + 1) / len(image_files) * 100
                    self.update_progress(progress, f"Copying: {new_position} ({i + 1}/{len(image_files)})")

                self.log_queue.put(f"\n✅ Shuffle completed!")
                self.log_queue.put(f"Images copied: {copied_images}")
                self.log_queue.put(f"XML files copied: {copied_xmls}")

            else:
                # In-place shuffle using temporary names
                self.log_queue.put("Performing in-place shuffle...")

                # First pass: rename to temporary names
                temp_mapping = {}
                for i, file in enumerate(image_files):
                    if self.stop_flag.is_set():
                        break

                    temp_name = f"_temp_{i:05d}_{file}"
                    src_path = os.path.join(directory, file)
                    temp_path = os.path.join(directory, temp_name)
                    os.rename(src_path, temp_path)
                    temp_mapping[file] = temp_name

                    # Also rename XML if exists
                    xml_file = Path(file).stem + '.xml'
                    src_xml = os.path.join(directory, xml_file)
                    if os.path.exists(src_xml):
                        temp_xml = f"_temp_{i:05d}_{xml_file}"
                        temp_xml_path = os.path.join(directory, temp_xml)
                        os.rename(src_xml, temp_xml_path)

                    progress = (i + 1) / (len(image_files) * 2) * 100
                    self.update_progress(progress, f"Preparing: {file}")

                # Second pass: rename from temporary to final shuffled names
                for i, (original_file, shuffled_file) in enumerate(zip(image_files, shuffled_files)):
                    if self.stop_flag.is_set():
                        break

                    temp_name = temp_mapping[shuffled_file]
                    temp_path = os.path.join(directory, temp_name)
                    new_name = f"{i + 1:05d}_{shuffled_file}"
                    final_path = os.path.join(directory, new_name)
                    os.rename(temp_path, final_path)

                    # Also rename XML if exists
                    xml_file = Path(shuffled_file).stem + '.xml'
                    temp_xml = f"_temp_{image_files.index(shuffled_file):05d}_{xml_file}"
                    temp_xml_path = os.path.join(directory, temp_xml)
                    if os.path.exists(temp_xml_path):
                        new_xml_name = f"{i + 1:05d}_{xml_file}"
                        final_xml_path = os.path.join(directory, new_xml_name)
                        os.rename(temp_xml_path, final_xml_path)

                    progress = 50 + ((i + 1) / len(image_files) * 50)
                    self.update_progress(progress, f"Shuffling: {shuffled_file}")

                self.log_queue.put(f"\n✅ In-place shuffle completed!")
                self.log_queue.put(f"Files shuffled: {len(image_files)}")

            if not self.stop_flag.is_set():
                self.log_queue.put("Shuffle operation completed successfully!")
            else:
                self.log_queue.put("\n⚠️ Operation cancelled by user")

        except Exception as e:
            self.log_queue.put(f"❌ Error during shuffle: {str(e)}")
        finally:
            self.set_buttons_state(True)
            self.update_progress(0, "Ready")

    def browse_directory(self, var):
        directory = filedialog.askdirectory(
            title="Select Directory",
            initialdir=self.app_root
        )
        if directory:
            var.set(directory)

    def update_progress(self, value, message=""):
        self.parent.after(0, lambda: self.progress_var.set(value))
        if message:
            self.parent.after(0, lambda: self.progress_label.config(text=message))

    def set_buttons_state(self, enabled):
        state = 'normal' if enabled else 'disabled'
        self.compare_btn.config(state=state)
        self.combine_btn.config(state=state)
        self.batch_combine_btn.config(state=state)
        self.split_btn.config(state=state)
        self.move_btn.config(state=state)
        self.prefix_btn.config(state=state)
        self.shuffle_files_btn.config(state=state)

        if enabled:
            self.stop_btn.pack_forget()
        else:
            self.stop_btn.pack(pady=10)

    def stop_operation(self):
        self.stop_flag.set()
        self.log_queue.put("Stopping operation...")

    def preview_prefix_changes(self):
        directory = self.prefix_dir_var.get()
        prefix = self.prefix_var.get()

        if not directory:
            messagebox.showwarning("Warning", "Please select a directory first.")
            return

        if not prefix:
            messagebox.showwarning("Warning", "Please enter a prefix.")
            return

        try:
            # Get all files in directory
            image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = [f for f in os.listdir(directory)
                           if Path(f).suffix.lower() in image_extensions]

            if not image_files:
                messagebox.showinfo("Info", "No image files found in the selected directory.")
                return

            # Create preview window
            preview_window = tk.Toplevel(self.parent)
            preview_window.title("Preview Prefix Changes")
            preview_window.geometry("600x400")

            # Create scrolled text widget
            text_widget = scrolledtext.ScrolledText(preview_window, wrap=tk.WORD, width=70, height=20)
            text_widget.pack(padx=10, pady=10, fill='both', expand=True)

            # Add preview information
            operation = "Copy" if self.prefix_copy_var.get() else "Rename"
            text_widget.insert(tk.END, f"Operation: {operation} files with prefix '{prefix}'\n")
            text_widget.insert(tk.END, f"Directory: {directory}\n")
            text_widget.insert(tk.END, f"Include XML files: {'Yes' if self.prefix_include_xml_var.get() else 'No'}\n")
            text_widget.insert(tk.END, f"\nFound {len(image_files)} image files\n")
            text_widget.insert(tk.END, "=" * 60 + "\n\n")

            # Show sample changes (first 20 files)
            sample_files = image_files[:20]
            for file in sample_files:
                text_widget.insert(tk.END, f"📄 {file} → {prefix}{file}\n")

                # Check for associated XML
                if self.prefix_include_xml_var.get():
                    xml_file = Path(file).stem + '.xml'
                    if os.path.exists(os.path.join(directory, xml_file)):
                        text_widget.insert(tk.END, f"📋 {xml_file} → {prefix}{xml_file}\n")

                text_widget.insert(tk.END, "\n")

            if len(image_files) > 20:
                text_widget.insert(tk.END, f"\n... and {len(image_files) - 20} more files")

            text_widget.config(state='disabled')

            ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Error previewing changes: {str(e)}")

    def add_prefix_to_files(self):
        directory = self.prefix_dir_var.get()
        prefix = self.prefix_var.get()

        if not directory:
            messagebox.showwarning("Warning", "Please select a directory.")
            return

        if not prefix:
            messagebox.showwarning("Warning", "Please enter a prefix.")
            return

        # Confirm action
        operation = "copy" if self.prefix_copy_var.get() else "rename"
        if messagebox.askyesno("Confirm",
                               f"Are you sure you want to {operation} all files with prefix '{prefix}'?"):
            self.stop_flag.clear()
            self.operation_thread = threading.Thread(
                target=self._add_prefix_thread,
                args=(directory, prefix),
                daemon=True
            )
            self.operation_thread.start()

    def _add_prefix_thread(self, directory, prefix):
        try:
            self.set_buttons_state(False)
            self.update_progress(0, "Adding prefix to files...")

            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = [f for f in os.listdir(directory)
                           if Path(f).suffix.lower() in image_extensions]

            if not image_files:
                self.log_queue.put("No image files found in directory.")
                return

            self.log_queue.put(f"\n=== Adding Prefix to Files ===")
            self.log_queue.put(f"Directory: {directory}")
            self.log_queue.put(f"Prefix: {prefix}")
            self.log_queue.put(f"Operation: {'Copy' if self.prefix_copy_var.get() else 'Rename'}")
            self.log_queue.put(f"Found {len(image_files)} image files")

            processed_images = 0
            processed_xmls = 0
            errors = 0

            for i, file in enumerate(image_files):
                if self.stop_flag.is_set():
                    break

                try:
                    src_path = os.path.join(directory, file)
                    new_name = prefix + file
                    dst_path = os.path.join(directory, new_name)

                    # Check if destination already exists
                    if os.path.exists(dst_path):
                        self.log_queue.put(f"⚠️ Skipping {file}: {new_name} already exists")
                        errors += 1
                        continue

                    if self.prefix_copy_var.get():
                        shutil.copy2(src_path, dst_path)
                    else:
                        os.rename(src_path, dst_path)

                    processed_images += 1

                    # Handle associated XML file
                    if self.prefix_include_xml_var.get():
                        xml_file = Path(file).stem + '.xml'
                        src_xml = os.path.join(directory, xml_file)

                        if os.path.exists(src_xml):
                            new_xml_name = prefix + xml_file
                            dst_xml = os.path.join(directory, new_xml_name)

                            if not os.path.exists(dst_xml):
                                if self.prefix_copy_var.get():
                                    shutil.copy2(src_xml, dst_xml)
                                else:
                                    os.rename(src_xml, dst_xml)
                                processed_xmls += 1

                    # Update progress
                    progress = (i + 1) / len(image_files) * 100
                    self.update_progress(progress, f"Processing: {file} ({i + 1}/{len(image_files)})")

                except Exception as e:
                    self.log_queue.put(f"❌ Error processing {file}: {str(e)}")
                    errors += 1

            if not self.stop_flag.is_set():
                self.log_queue.put(f"\n✅ Prefix operation completed!")
                self.log_queue.put(f"Images processed: {processed_images}")
                if self.prefix_include_xml_var.get():
                    self.log_queue.put(f"XML files processed: {processed_xmls}")
                if errors > 0:
                    self.log_queue.put(f"Errors/Skipped: {errors}")
            else:
                self.log_queue.put("\n⚠️ Operation cancelled by user")

        except Exception as e:
            self.log_queue.put(f"❌ Error during prefix operation: {str(e)}")
        finally:
            self.set_buttons_state(True)
            self.update_progress(0, "Ready")

    def preview_batch_folders(self):
        parent_dir = self.parent_dir_var.get()
        if not parent_dir:
            messagebox.showwarning("Warning", "Please select a parent directory first.")
            return

        try:
            folders = []

            if self.recursive_var.get():
                # Get all subdirectories recursively
                for root, dirs, files in os.walk(parent_dir):
                    if root != parent_dir:  # Skip the parent directory itself
                        folders.append(root)
            else:
                # Get only immediate subdirectories
                folders = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                           if os.path.isdir(os.path.join(parent_dir, d))]

            if not folders:
                messagebox.showinfo("Info", "No subdirectories found in the selected directory.")
                return

            # Show preview window
            preview_window = tk.Toplevel(self.parent)
            preview_window.title("Folders to Combine")
            preview_window.geometry("500x400")

            # Create scrolled text widget
            text_widget = scrolledtext.ScrolledText(preview_window, wrap=tk.WORD, width=60, height=20)
            text_widget.pack(padx=10, pady=10, fill='both', expand=True)

            # Add folder information
            text_widget.insert(tk.END, f"Found {len(folders)} folders to combine:\n\n")

            total_files = 0
            for folder in sorted(folders):
                file_count = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])
                total_files += file_count
                text_widget.insert(tk.END, f"📁 {os.path.relpath(folder, parent_dir)} ({file_count} files)\n")

            if self.include_parent_files_var.get():
                parent_files = len([f for f in os.listdir(parent_dir) if os.path.isfile(os.path.join(parent_dir, f))])
                total_files += parent_files
                text_widget.insert(tk.END, f"\n📁 [Parent Directory] ({parent_files} files)\n")

            text_widget.insert(tk.END, f"\n\nTotal files to combine: {total_files}")
            text_widget.config(state='disabled')

            ttk.Button(preview_window, text="Close", command=preview_window.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Error previewing folders: {str(e)}")

    def batch_combine_folders(self):
        parent_dir = self.parent_dir_var.get()
        output_dir = self.batch_output_var.get()

        if not parent_dir or not output_dir:
            messagebox.showwarning("Warning", "Please select both parent and output directories.")
            return

        self.stop_flag.clear()
        self.operation_thread = threading.Thread(
            target=self._batch_combine_thread,
            args=(parent_dir, output_dir),
            daemon=True
        )
        self.operation_thread.start()

    def _batch_combine_thread(self, parent_dir, output_dir):
        try:
            self.set_buttons_state(False)
            self.update_progress(0, "Preparing batch combination...")

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            self.log_queue.put("\n=== Batch Combining Folders ===")
            self.log_queue.put(f"Parent directory: {parent_dir}")
            self.log_queue.put(f"Output directory: {output_dir}")

            # Collect all folders to process
            folders_to_process = []

            if self.recursive_var.get():
                # Get all subdirectories recursively
                for root, dirs, files in os.walk(parent_dir):
                    if root != parent_dir:  # Skip the parent directory itself
                        folders_to_process.append(root)
            else:
                # Get only immediate subdirectories
                folders_to_process = [os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
                                      if os.path.isdir(os.path.join(parent_dir, d))]

            # Include parent directory files if requested
            if self.include_parent_files_var.get():
                folders_to_process.insert(0, parent_dir)

            if not folders_to_process:
                self.log_queue.put("No folders found to combine.")
                return

            self.log_queue.put(f"Found {len(folders_to_process)} folders to combine")

            # Count total files for progress tracking
            total_files = 0
            for folder in folders_to_process:
                total_files += len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

            self.log_queue.put(f"Total files to copy: {total_files}")

            # Copy files from each folder
            copied_files = 0
            conflicts = 0

            for folder_idx, folder in enumerate(folders_to_process):
                if self.stop_flag.is_set():
                    break

                folder_name = os.path.basename(folder) if folder != parent_dir else "[Parent]"
                self.log_queue.put(f"\nProcessing folder: {folder_name}")

                files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

                for file in files:
                    if self.stop_flag.is_set():
                        break

                    src_path = os.path.join(folder, file)
                    dst_path = os.path.join(output_dir, file)

                    # Handle naming conflicts
                    if os.path.exists(dst_path):
                        conflicts += 1
                        # Add folder name prefix to avoid overwriting
                        base, ext = os.path.splitext(file)
                        safe_folder_name = os.path.basename(folder).replace(' ', '_')
                        dst_path = os.path.join(output_dir, f"{safe_folder_name}_{file}")

                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_files += 1

                        # Update progress
                        progress = (copied_files / total_files) * 100
                        self.update_progress(progress, f"Copying: {file} ({copied_files}/{total_files})")

                    except Exception as e:
                        self.log_queue.put(f"Error copying {file}: {str(e)}")

            if not self.stop_flag.is_set():
                self.log_queue.put(f"\n✅ Batch combination completed!")
                self.log_queue.put(f"Total files copied: {copied_files}")
                if conflicts > 0:
                    self.log_queue.put(f"Files renamed due to conflicts: {conflicts}")
                self.log_queue.put(f"Output directory: {output_dir}")
            else:
                self.log_queue.put("\n⚠️ Operation cancelled by user")

        except Exception as e:
            self.log_queue.put(f"❌ Error during batch combination: {str(e)}")
        finally:
            self.set_buttons_state(True)
            self.update_progress(0, "Ready")

    def count_files(self, directory):
        """Count all files in the directory including subdirectories."""
        total_files = 0
        file_types = {}

        for root, dirs, files in os.walk(directory):
            total_files += len(files)
            for file in files:
                ext = Path(file).suffix.lower()
                file_types[ext] = file_types.get(ext, 0) + 1

        return total_files, file_types

    def compare_directories(self):
        dir1 = self.dir1_var.get()
        dir2 = self.dir2_var.get()

        if not dir1 or not dir2:
            messagebox.showwarning("Warning", "Please select both directories to compare.")
            return

        self.stop_flag.clear()
        self.operation_thread = threading.Thread(target=self._compare_thread,
                                                 args=(dir1, dir2), daemon=True)
        self.operation_thread.start()

    def _compare_thread(self, dir1, dir2):
        try:
            self.set_buttons_state(False)
            self.update_progress(0, "Analyzing directories...")

            self.log_queue.put("\n=== Directory Comparison ===")

            count1, types1 = self.count_files(dir1)
            self.update_progress(50, "Analyzing second directory...")

            count2, types2 = self.count_files(dir2)
            self.update_progress(100, "Analysis complete")

            self.log_queue.put(f"\nDirectory 1: {dir1}")
            self.log_queue.put(f"Total files: {count1}")
            self.log_queue.put("File types:")
            for ext, count in sorted(types1.items()):
                self.log_queue.put(f"  {ext}: {count}")

            self.log_queue.put(f"\nDirectory 2: {dir2}")
            self.log_queue.put(f"Total files: {count2}")
            self.log_queue.put("File types:")
            for ext, count in sorted(types2.items()):
                self.log_queue.put(f"  {ext}: {count}")

            self.log_queue.put(f"\nDifference in file count: {abs(count1 - count2)}")

        finally:
            self.set_buttons_state(True)
            self.update_progress(0, "Ready")

    def combine_datasets(self):
        orig_dir = self.orig_dir_var.get()
        aug_dir = self.aug_dir_var.get()
        comb_dir = self.comb_dir_var.get()

        if not all([orig_dir, aug_dir, comb_dir]):
            messagebox.showwarning("Warning", "Please select all directories.")
            return

        self.stop_flag.clear()
        self.operation_thread = threading.Thread(target=self._combine_thread,
                                                 args=(orig_dir, aug_dir, comb_dir), daemon=True)
        self.operation_thread.start()

    def _combine_thread(self, orig_dir, aug_dir, comb_dir):
        try:
            self.set_buttons_state(False)

            # Create combined directory if it doesn't exist
            os.makedirs(comb_dir, exist_ok=True)

            self.log_queue.put("\n=== Combining Two Folders ===")
            self.log_queue.put(f"Folder 1: {orig_dir}")
            self.log_queue.put(f"Folder 2: {aug_dir}")
            self.log_queue.put(f"Output: {comb_dir}")

            # Count total files
            orig_files = [f for f in os.listdir(orig_dir) if os.path.isfile(os.path.join(orig_dir, f))]
            aug_files = [f for f in os.listdir(aug_dir) if os.path.isfile(os.path.join(aug_dir, f))]
            total_files = len(orig_files) + len(aug_files)

            # Copy original files
            copied = 0
            for i, filename in enumerate(orig_files):
                if self.stop_flag.is_set():
                    break

                src_path = os.path.join(orig_dir, filename)
                dst_path = os.path.join(comb_dir, filename)
                shutil.copy2(src_path, dst_path)
                copied += 1

                progress = (i + 1) / total_files * 100
                self.update_progress(progress, f"Copying file {i + 1}/{total_files}")

            self.log_queue.put(f"Copied {copied} files from folder 1")

            # Copy additional files
            copied = 0
            conflicts = 0
            for i, filename in enumerate(aug_files):
                if self.stop_flag.is_set():
                    break

                src_path = os.path.join(aug_dir, filename)
                dst_path = os.path.join(comb_dir, filename)

                if os.path.exists(dst_path):
                    conflicts += 1
                    # Add prefix to avoid overwriting
                    base, ext = os.path.splitext(filename)
                    dst_path = os.path.join(comb_dir, f"folder2_{filename}")

                shutil.copy2(src_path, dst_path)
                copied += 1

                progress = (len(orig_files) + i + 1) / total_files * 100
                self.update_progress(progress, f"Copying file {len(orig_files) + i + 1}/{total_files}")

            self.log_queue.put(f"Copied {copied} files from folder 2")
            if conflicts > 0:
                self.log_queue.put(f"Renamed {conflicts} files to avoid conflicts")

            if not self.stop_flag.is_set():
                self.log_queue.put(f"\n✅ Combined folders created in: {comb_dir}")
            else:
                self.log_queue.put("\n⚠️ Operation cancelled by user")

        except Exception as e:
            self.log_queue.put(f"❌ Error combining folders: {str(e)}")
        finally:
            self.set_buttons_state(True)
            self.update_progress(0, "Ready")

    def split_dataset(self):
        source_dir = self.split_source_var.get()
        output_base = self.split_output_var.get()

        if not source_dir or not output_base:
            messagebox.showwarning("Warning", "Please select source and output directories.")
            return

        try:
            train_ratio = float(self.train_ratio_var.get()) / 100
            test_ratio = float(self.test_ratio_var.get()) / 100
            valid_ratio = float(self.valid_ratio_var.get()) / 100

            if abs(train_ratio + test_ratio + valid_ratio - 1.0) > 0.01:
                messagebox.showwarning("Warning", "Ratios must sum to 100%")
                return

        except ValueError:
            messagebox.showerror("Error", "Invalid ratio values")
            return

        self.stop_flag.clear()
        self.operation_thread = threading.Thread(target=self._split_thread,
                                                 args=(source_dir, output_base, train_ratio, test_ratio, valid_ratio),
                                                 daemon=True)
        self.operation_thread.start()

    def _split_thread(self, source_dir, output_base, train_ratio, test_ratio, valid_ratio):
        try:
            self.set_buttons_state(False)
            self.update_progress(0, "Preparing to split dataset...")

            # Create output directories
            train_dir = os.path.join(output_base, "train")
            test_dir = os.path.join(output_base, "test")
            valid_dir = os.path.join(output_base, "valid")

            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(valid_dir, exist_ok=True)

            self.log_queue.put("\n=== Splitting Dataset ===")

            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = [f for f in os.listdir(source_dir)
                           if Path(f).suffix.lower() in image_extensions]

            if self.shuffle_var.get():
                np.random.shuffle(image_files)

            total_files = len(image_files)
            train_end = int(train_ratio * total_files)
            test_end = train_end + int(test_ratio * total_files)

            train_files = image_files[:train_end]
            test_files = image_files[train_end:test_end]
            valid_files = image_files[test_end:]

            # Copy files function
            def copy_files(files, destination, offset=0):
                copied_images = 0
                copied_xmls = 0
                for i, file in enumerate(files):
                    if self.stop_flag.is_set():
                        break

                    # Copy image
                    src_img = os.path.join(source_dir, file)
                    dst_img = os.path.join(destination, file)
                    shutil.copy2(src_img, dst_img)
                    copied_images += 1

                    # Copy XML if exists
                    xml_file = Path(file).stem + '.xml'
                    src_xml = os.path.join(source_dir, xml_file)
                    if os.path.exists(src_xml):
                        dst_xml = os.path.join(destination, xml_file)
                        shutil.copy2(src_xml, dst_xml)
                        copied_xmls += 1

                    progress = (offset + i + 1) / total_files * 100
                    self.update_progress(progress, f"Processing file {offset + i + 1}/{total_files}")

                return copied_images, copied_xmls

            # Copy files to directories
            train_imgs, train_xmls = copy_files(train_files, train_dir, 0)
            test_imgs, test_xmls = copy_files(test_files, test_dir, len(train_files))
            valid_imgs, valid_xmls = copy_files(valid_files, valid_dir, len(train_files) + len(test_files))

            if not self.stop_flag.is_set():
                self.log_queue.put(f"\nFiles distributed:")
                self.log_queue.put(f"Train: {train_imgs} images, {train_xmls} XMLs")
                self.log_queue.put(f"Test: {test_imgs} images, {test_xmls} XMLs")
                self.log_queue.put(f"Valid: {valid_imgs} images, {valid_xmls} XMLs")
                self.log_queue.put(f"\n✅ Dataset split completed!")
            else:
                self.log_queue.put("\n⚠️ Operation cancelled by user")

        except Exception as e:
            self.log_queue.put(f"❌ Error splitting dataset: {str(e)}")
        finally:
            self.set_buttons_state(True)
            self.update_progress(0, "Ready")

    def move_files(self):
        source_dir = self.move_source_var.get()
        dest_dir = self.move_dest_var.get()

        if not source_dir or not dest_dir:
            messagebox.showwarning("Warning", "Please select both directories.")
            return

        try:
            percentage = float(self.move_percent_var.get()) / 100
            if not 0 < percentage <= 100:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid percentage value")
            return

        self.stop_flag.clear()
        operation = self.move_copy_var.get()
        self.operation_thread = threading.Thread(target=self._move_thread,
                                                 args=(source_dir, dest_dir, percentage, operation),
                                                 daemon=True)
        self.operation_thread.start()

    def _move_thread(self, source_dir, dest_dir, percentage, operation):
        try:
            self.set_buttons_state(False)
            self.update_progress(0, "Preparing file operation...")

            os.makedirs(dest_dir, exist_ok=True)

            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            jpg_files = [f for f in os.listdir(source_dir)
                         if Path(f).suffix.lower() in image_extensions]

            num_files_to_process = int(len(jpg_files) * percentage)
            files_to_process = random.sample(jpg_files, num_files_to_process)

            self.log_queue.put(f"\n=== {'Moving' if operation == 'move' else 'Copying'} Files ===")

            processed_images = 0
            processed_xmls = 0

            for i, jpg_file in enumerate(files_to_process):
                if self.stop_flag.is_set():
                    break

                # Process image
                src_img = os.path.join(source_dir, jpg_file)
                dst_img = os.path.join(dest_dir, jpg_file)

                if operation == "move":
                    shutil.move(src_img, dst_img)
                else:
                    shutil.copy2(src_img, dst_img)
                processed_images += 1

                # Process XML if exists
                xml_file = Path(jpg_file).stem + '.xml'
                src_xml = os.path.join(source_dir, xml_file)
                if os.path.exists(src_xml):
                    dst_xml = os.path.join(dest_dir, xml_file)
                    if operation == "move":
                        shutil.move(src_xml, dst_xml)
                    else:
                        shutil.copy2(src_xml, dst_xml)
                    processed_xmls += 1

                progress = (i + 1) / num_files_to_process * 100
                self.update_progress(progress, f"Processing file {i + 1}/{num_files_to_process}")

            if not self.stop_flag.is_set():
                self.log_queue.put(
                    f"{'Moved' if operation == 'move' else 'Copied'} {processed_images} images and {processed_xmls} XML files")
                self.log_queue.put(f"✅ Operation completed!")
            else:
                self.log_queue.put("\n⚠️ Operation cancelled by user")

        except Exception as e:
            self.log_queue.put(f"❌ Error: {str(e)}")
        finally:
            self.set_buttons_state(True)
            self.update_progress(0, "Ready")





