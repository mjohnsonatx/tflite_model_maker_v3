import os
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from AnnotationGui import AnnotationTab
from AugmentDataTab import AugmentationTab
from DataCleaningGui import DataCleaningTab
from FileOperationsTab import FileOperationsTab
from FrameExtractionGui import FrameExtractionTab
from PreviewValidateTab import PreviewValidateTab
from TrainingTab import TrainingTab


class TFLiteModelMakerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("TFLite Model Maker Studio")
        self.root.geometry("1100x850")

        # Setup logging
        self.log_queue = queue.Queue()
        self.log_entry_count = 0
        self.setup_ui()
        self.process_log_queue()

    def setup_ui(self):
        # Create main container that fills the window
        main_container = ttk.Frame(self.root)
        main_container.pack(fill='both', expand=True, padx=5, pady=5)

        # Configure grid weights
        main_container.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)

        # Create main PanedWindow for resizable split
        self.main_paned = ttk.PanedWindow(main_container, orient=tk.VERTICAL)
        self.main_paned.grid(row=0, column=0, sticky='nsew')

        # Top frame for notebook (scrollable)
        self.notebook_container = ttk.Frame(self.main_paned)
        # Set minimum height to prevent collapse
        self.notebook_container.configure(height=400)
        self.main_paned.add(self.notebook_container, weight=3)

        # Create canvas and scrollbar for notebook area
        self.notebook_canvas = tk.Canvas(self.notebook_container, highlightthickness=0)
        self.notebook_scrollbar = ttk.Scrollbar(self.notebook_container, orient="vertical",
                                                command=self.notebook_canvas.yview)

        self.notebook_scroll_frame = ttk.Frame(self.notebook_canvas)
        self.notebook_scroll_frame.bind(
            "<Configure>",
            lambda e: self.notebook_canvas.configure(scrollregion=self.notebook_canvas.bbox("all"))
        )

        self.notebook_canvas_window = self.notebook_canvas.create_window(
            (0, 0),
            window=self.notebook_scroll_frame,
            anchor="nw"
        )

        self.notebook_canvas.configure(yscrollcommand=self.notebook_scrollbar.set)

        # Pack canvas and scrollbar
        self.notebook_canvas.pack(side="left", fill="both", expand=True)
        self.notebook_scrollbar.pack(side="right", fill="y")

        # Create notebook inside scrollable frame
        self.notebook = ttk.Notebook(self.notebook_scroll_frame)
        # IMPORTANT: Don't use expand=True here - this is what causes the extra space
        self.notebook.pack(fill='x', padx=5, pady=5)  # Changed from fill='both', expand=True

        # Bind canvas resize to update scroll frame width
        self.notebook_canvas.bind('<Configure>', self._on_canvas_configure)

        # Create tabs
        self.data_cleaning_frame = ttk.Frame(self.notebook)
        self.frame_extraction_frame = ttk.Frame(self.notebook)
        self.annotation_frame = ttk.Frame(self.notebook)
        self.file_operations_frame = ttk.Frame(self.notebook)
        self.preview_validate_frame = ttk.Frame(self.notebook)
        self.augmentation_frame = ttk.Frame(self.notebook)
        self.training_frame = ttk.Frame(self.notebook)
        self.inference_frame = ttk.Frame(self.notebook)

        # Add tabs to notebook
        self.notebook.add(self.data_cleaning_frame, text="Data Cleaning")
        self.notebook.add(self.frame_extraction_frame, text="Frame Extraction")
        self.notebook.add(self.annotation_frame, text="Annotation")
        self.notebook.add(self.file_operations_frame, text="File Operations")
        self.notebook.add(self.preview_validate_frame, text="Preview & Validate")
        self.notebook.add(self.augmentation_frame, text="Augmentation")
        self.notebook.add(self.training_frame, text="Training")
        self.notebook.add(self.inference_frame, text="Inference")

        # Initialize tabs - do this before creating log panel
        self.data_cleaning_tab = DataCleaningTab(self.data_cleaning_frame, self.log_queue)
        self.frame_extraction_tab = FrameExtractionTab(self.frame_extraction_frame, self.log_queue)
        self.annotation_tab = AnnotationTab(self.annotation_frame, self.log_queue)
        self.file_operations_tab = FileOperationsTab(self.file_operations_frame, self.log_queue)
        self.preview_validate_tab = PreviewValidateTab(self.preview_validate_frame, self.log_queue)
        self.augmentation_tab = AugmentationTab(self.augmentation_frame, self.log_queue)
        self.training_tab = TrainingTab(self.training_frame, self.log_queue)

        # Bottom frame for log panel
        log_container = ttk.Frame(self.main_paned)
        # Set minimum height to prevent collapse
        log_container.configure(height=200)
        self.main_paned.add(log_container, weight=2)

        # Create log panel
        self.create_log_panel(log_container)

        # Force update of all widgets before setting sash position
        self.root.update_idletasks()

        # Set initial sash position after everything is created and updated
        # Use a longer delay and check if widgets exist
        self.root.after(500, self._set_initial_sash_position)

        # Bind sash movement to check if scrollbar is needed
        self.main_paned.bind('<ButtonRelease-1>', self._check_scrollbar_visibility)
        self.main_paned.bind('<B1-Motion>', self._check_scrollbar_visibility)

        # Set up mouse wheel scrolling for all widgets
        self._setup_mousewheel_scrolling()

    def _setup_mousewheel_scrolling(self):
        """Set up mouse wheel scrolling for the notebook canvas and all child widgets"""
        # Bind mouse wheel to the canvas itself
        self.notebook_canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows
        self.notebook_canvas.bind("<Button-4>", self._on_mousewheel)  # Linux
        self.notebook_canvas.bind("<Button-5>", self._on_mousewheel)  # Linux

        # Bind mouse wheel to all child widgets of the notebook
        def bind_mousewheel_to_children(widget):
            # Bind to the widget
            widget.bind("<MouseWheel>", self._on_mousewheel)  # Windows
            widget.bind("<Button-4>", self._on_mousewheel)  # Linux
            widget.bind("<Button-5>", self._on_mousewheel)  # Linux

            # Recursively bind to all children
            for child in widget.winfo_children():
                bind_mousewheel_to_children(child)

        # Apply bindings to the notebook and all its children
        bind_mousewheel_to_children(self.notebook_scroll_frame)

        # Re-bind when tab changes to ensure new tab content is bound
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event):
        """Re-bind mouse wheel when tab changes"""
        self.root.after(10, self._rebind_mousewheel)

    def _rebind_mousewheel(self):
        """Rebind mouse wheel to ensure all widgets in current tab can scroll"""

        def bind_mousewheel_to_children(widget):
            try:
                widget.bind("<MouseWheel>", self._on_mousewheel)  # Windows
                widget.bind("<Button-4>", self._on_mousewheel)  # Linux
                widget.bind("<Button-5>", self._on_mousewheel)  # Linux

                for child in widget.winfo_children():
                    bind_mousewheel_to_children(child)
            except:
                pass  # Skip any widgets that can't be bound

        # Get current tab
        current_tab_index = self.notebook.index("current")
        current_tab = self.notebook.winfo_children()[current_tab_index]
        bind_mousewheel_to_children(current_tab)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        # Check if the notebook canvas is visible and should scroll
        if self.notebook_scrollbar.winfo_ismapped():
            if event.delta:
                # Windows and MacOS
                self.notebook_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            else:
                # Linux
                if event.num == 4:
                    self.notebook_canvas.yview_scroll(-1, "units")
                elif event.num == 5:
                    self.notebook_canvas.yview_scroll(1, "units")
        # Return "break" to prevent event from propagating
        return "break"

    def _set_initial_sash_position(self):
        """Safely set the initial sash position"""
        try:
            # Ensure the paned window is properly initialized
            if self.main_paned.winfo_exists() and self.main_paned.winfo_viewable():
                # Get the total height
                total_height = self.main_paned.winfo_height()
                if total_height > 100:  # Make sure we have a reasonable height
                    # Set to 60% for notebook
                    sash_position = int(total_height * 0.6)
                    self.main_paned.sashpos(0, sash_position)
                else:
                    # If height is still too small, try again later
                    self.root.after(100, self._set_initial_sash_position)
            else:
                # If not ready, try again
                self.root.after(100, self._set_initial_sash_position)
        except Exception as e:
            # Log any errors but don't crash
            print(f"Error setting sash position: {e}")

    def _on_canvas_configure(self, event):
        # Update the scroll frame width to match canvas
        canvas_width = event.width
        if canvas_width > 0:  # Only update if we have a valid width
            self.notebook_canvas.itemconfig(self.notebook_canvas_window, width=canvas_width)

    def _check_scrollbar_visibility(self, event=None):
        """Check if scrollbar should be visible based on content height vs visible height"""
        self.root.after(10, self._update_scrollbar_visibility)

    def _update_scrollbar_visibility(self):
        try:
            # Update canvas scroll region
            self.notebook_canvas.update_idletasks()

            # Get the height of the content and the visible area
            content_height = self.notebook_scroll_frame.winfo_reqheight()
            visible_height = self.notebook_canvas.winfo_height()

            # Show or hide scrollbar based on content
            if content_height > visible_height:
                if not self.notebook_scrollbar.winfo_ismapped():
                    self.notebook_scrollbar.pack(side="right", fill="y")
            else:
                if self.notebook_scrollbar.winfo_ismapped():
                    self.notebook_scrollbar.pack_forget()
                # Reset scroll position
                self.notebook_canvas.yview_moveto(0)

            # Re-bind mousewheel after visibility check
            self.root.after(10, self._rebind_mousewheel)
        except Exception:
            # Ignore any errors during scrollbar update
            pass

    def create_log_panel(self, parent):
        # Main log frame that fills parent
        log_main_frame = ttk.Frame(parent)
        log_main_frame.pack(fill='both', expand=True)

        # Configure grid weights
        log_main_frame.rowconfigure(0, weight=1)
        log_main_frame.columnconfigure(0, weight=1)

        log_frame = ttk.LabelFrame(log_main_frame, text="Log Output", padding="5")
        log_frame.grid(row=0, column=0, sticky='nsew')

        # Configure internal weights
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        # Create text widget with larger font
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap=tk.WORD,
            font=('Consolas', 10),
            bg='#f0f0f0',
            fg='#000000'
        )
        self.log_text.grid(row=0, column=0, sticky='nsew')

        # Button frame
        button_frame = ttk.Frame(log_frame)
        button_frame.grid(row=1, column=0, sticky='ew', pady=(5, 0))

        # Add buttons with better layout
        ttk.Button(button_frame, text="Clear Log",
                   command=self.clear_log).pack(side=tk.RIGHT, padx=(5, 0))

        ttk.Button(button_frame, text="Save Log",
                   command=self.save_log).pack(side=tk.RIGHT)

        ttk.Button(button_frame, text="Copy All",
                   command=self.copy_log).pack(side=tk.RIGHT, padx=(0, 5))

        # Add log level label
        self.log_level_label = ttk.Label(button_frame, text="Log entries: 0")
        self.log_level_label.pack(side=tk.LEFT)

    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
        self.log_entry_count = 0
        self.log_level_label.config(text="Log entries: 0")

    def save_log(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            messagebox.showinfo("Success", "Log saved successfully!")

    def copy_log(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.log_text.get(1.0, tk.END))
        messagebox.showinfo("Success", "Log copied to clipboard!")

    def process_log_queue(self):
        try:
            entries_added = 0
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + '\n')
                self.log_text.see(tk.END)
                entries_added += 1

        except queue.Empty:
            # Update log entry count if entries were added
            if entries_added > 0:
                self.log_entry_count += entries_added
                self.log_level_label.config(text=f"Log entries: {self.log_entry_count}")
        finally:
            self.root.after(100, self.process_log_queue)


if __name__ == "__main__":
    root = tk.Tk()
    app = TFLiteModelMakerGUI(root)
    root.mainloop()
