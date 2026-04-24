import os
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from xml.dom.minidom import parseString

import cv2
from PIL import Image, ImageTk

from LabelMemory import LabelMemory


class PreviewValidateTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.label_memory = LabelMemory(self.app_root)
        self.current_index = 0
        self.all_image_files = []
        self.image_files = []
        self.preview_window = None
        self.current_preview_boxes = []
        self.annotation_listbox = None
        self.annotation_window = None
        self.reannotation_image_path = None
        self.reannotation_image = None
        self.reannotation_scale = 1.0
        self.reannotation_annotations = []
        self.reannotation_rect_ids = []
        self.reannotation_text_ids = []
        self.reannotation_current_rect = None
        self.reannotation_start_x = None
        self.reannotation_start_y = None
        self.reannotation_canvas = None
        self.folder_labels_listbox = None
        self.folder_labels_summary_var = None
        self.setup_ui()
        self.label_memory.subscribe(self._on_labels_changed)

    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Section 1: Directory Selection
        ttk.Label(main_frame, text="Preview Annotations",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, columnspan=3, pady=(0, 10))

        ttk.Label(main_frame, text="Image Directory:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.dir_var, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.browse_directory).grid(row=1, column=2, padx=(5, 0))

        # Section 2: Preview Options
        options_frame = ttk.LabelFrame(main_frame, text="Preview Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 10))

        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Show labels on bounding boxes",
                        variable=self.show_labels_var).grid(row=0, column=0, sticky=tk.W)

        self.show_confidence_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Show confidence scores (if available)",
                        variable=self.show_confidence_var).grid(row=1, column=0, sticky=tk.W)

        self.auto_advance_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Auto-advance to next image",
                        variable=self.auto_advance_var).grid(row=2, column=0, sticky=tk.W)

        # Box color selection
        color_frame = ttk.Frame(options_frame)
        color_frame.grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Label(color_frame, text="Box color:").pack(side=tk.LEFT, padx=(0, 10))
        self.box_color_var = tk.StringVar(value="green")
        colors = ["green", "red", "blue", "yellow", "cyan", "magenta"]
        ttk.Combobox(color_frame, textvariable=self.box_color_var,
                     values=colors, width=10, state="readonly").pack(side=tk.LEFT)

        # Section 3: Search
        search_frame = ttk.LabelFrame(main_frame, text="Search Folder", padding="10")
        search_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 10))
        search_frame.columnconfigure(1, weight=1)

        ttk.Label(search_frame, text="Label contains:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.label_filter_var = tk.StringVar()
        label_filter_entry = ttk.Entry(search_frame, textvariable=self.label_filter_var, width=30)
        label_filter_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        label_filter_entry.bind('<Return>', lambda e: self.apply_filters())

        ttk.Label(search_frame, text="Number of labels:").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 10), pady=(8, 0)
        )
        count_frame = ttk.Frame(search_frame)
        count_frame.grid(row=1, column=1, sticky=tk.W, pady=(8, 0))
        self.count_filter_mode_var = tk.StringVar(value="Any")
        count_modes = ["Any", "Equals", "At least", "At most"]
        ttk.Combobox(count_frame, textvariable=self.count_filter_mode_var,
                     values=count_modes, width=10, state="readonly").pack(side=tk.LEFT)
        self.count_filter_value_var = tk.StringVar()
        count_filter_entry = ttk.Entry(count_frame, textvariable=self.count_filter_value_var, width=8)
        count_filter_entry.pack(side=tk.LEFT, padx=(6, 0))
        count_filter_entry.bind('<Return>', lambda e: self.apply_filters())

        search_button_frame = ttk.Frame(search_frame)
        search_button_frame.grid(row=0, column=2, rowspan=2, sticky=tk.E, padx=(10, 0))
        ttk.Button(search_button_frame, text="Apply Search",
                   command=self.apply_filters).pack(side=tk.TOP, fill=tk.X)
        ttk.Button(search_button_frame, text="Clear Search",
                   command=self.clear_filters).pack(side=tk.TOP, fill=tk.X, pady=(6, 0))

        self.search_summary_var = tk.StringVar(value="No directory loaded")
        ttk.Label(search_frame, textvariable=self.search_summary_var).grid(
            row=2, column=0, columnspan=3, sticky=tk.W, pady=(8, 0)
        )

        # Section 3b: Folder label scan
        folder_labels_frame = ttk.LabelFrame(main_frame, text="Folder Labels", padding="10")
        folder_labels_frame.grid(row=4, column=0, columnspan=3,
                                 sticky=(tk.W, tk.E), pady=(0, 10))
        folder_labels_frame.columnconfigure(0, weight=1)

        self.folder_labels_summary_var = tk.StringVar(
            value="Scan the folder to see every label present in the XML files."
        )
        ttk.Label(folder_labels_frame, textvariable=self.folder_labels_summary_var).grid(
            row=0, column=0, columnspan=2, sticky=tk.W
        )

        labels_list_frame = ttk.Frame(folder_labels_frame)
        labels_list_frame.grid(row=1, column=0, columnspan=2,
                               sticky=(tk.W, tk.E), pady=(8, 0))
        labels_list_frame.columnconfigure(0, weight=1)

        labels_scroll = ttk.Scrollbar(labels_list_frame, orient=tk.VERTICAL)
        labels_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.folder_labels_listbox = tk.Listbox(
            labels_list_frame,
            height=6,
            yscrollcommand=labels_scroll.set
        )
        self.folder_labels_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E))
        labels_scroll.config(command=self.folder_labels_listbox.yview)
        self.folder_labels_listbox.bind(
            "<<ListboxSelect>>",
            lambda e: self._apply_folder_label_filter()
        )

        folder_labels_buttons = ttk.Frame(folder_labels_frame)
        folder_labels_buttons.grid(row=2, column=0, columnspan=2,
                                   sticky=tk.W, pady=(6, 0))
        ttk.Button(folder_labels_buttons, text="Scan Folder XMLs",
                   command=self.scan_folder_labels).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(folder_labels_buttons, text="Filter by Selected",
                   command=self._apply_folder_label_filter).pack(side=tk.LEFT, padx=5)
        ttk.Button(folder_labels_buttons, text="Remember All Labels",
                   command=self._remember_folder_labels).pack(side=tk.LEFT, padx=5)

        # Section 4: Controls
        ttk.Label(main_frame, text="Controls",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=5, column=0, columnspan=3, pady=(20, 10))

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=6, column=0, columnspan=3, pady=(0, 10))

        ttk.Button(control_frame, text="Start Preview",
                   command=self.start_preview).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Previous",
                   command=self.previous_image).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Next",
                   command=self.next_image).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Stop Preview",
                   command=self.stop_preview).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Delete Image + XML",
                   command=self.delete_current).grid(row=1, column=0, padx=5, pady=(6, 0))
        ttk.Button(control_frame, text="Delete Annotation XML",
                   command=self.delete_current_annotation_file).grid(row=1, column=1, padx=5, pady=(6, 0))
        ttk.Button(control_frame, text="Re-Annotate Current",
                   command=self.reannotate_current).grid(row=1, column=2, padx=5, pady=(6, 0))

        # Info section
        info_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        info_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        info_text = """Keyboard shortcuts during preview:
- Arrow Left/Right or A/D: Navigate between images
- Delete or E: Delete current image and its XML file
- X: Delete the current XML annotation file only
- R: Re-annotate the current image
- Space: Toggle auto-advance
- Escape: Close preview window
- L: Toggle label display
- C: Change box color

Note: Deleted files are permanently removed. Use with caution!"""

        self.info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        self.info_label.pack(anchor=tk.W)

        # Statistics
        self.stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        self.stats_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        self.stats_text = tk.Text(self.stats_frame, height=8, wrap=tk.WORD)
        self.stats_text.pack(fill="both", expand=True)

        # Progress
        self.progress_label = ttk.Label(main_frame, text="")
        self.progress_label.grid(row=9, column=0, columnspan=3, pady=(10, 0))

    def browse_directory(self):
        directory = filedialog.askdirectory(
            title="Select Image Directory",
            initialdir=self.app_root
        )
        if directory:
            self.dir_var.set(directory)
            self.load_images()

    def load_images(self):
        if not self.dir_var.get():
            return

        image_extensions = {'.jpg', '.jpeg', '.png'}
        self.all_image_files = []

        for file in sorted(os.listdir(self.dir_var.get())):
            if Path(file).suffix.lower() in image_extensions:
                self.all_image_files.append(file)

        self.apply_filters()
        self.log_queue.put(f"Found {len(self.all_image_files)} images in directory")

    def _image_path(self, image_file):
        return os.path.join(self.dir_var.get(), image_file)

    def _xml_path_for_image(self, image_file):
        return os.path.join(self.dir_var.get(), Path(image_file).stem + '.xml')

    def _get_image_labels(self, image_file):
        xml_path = self._xml_path_for_image(image_file)
        if not os.path.exists(xml_path):
            return []

        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            self.log_queue.put(f"Could not parse {os.path.basename(xml_path)}: {exc}")
            return []

        labels = []
        for obj in root.findall('object'):
            label = obj.findtext('name', default='object').strip() or 'object'
            labels.append(label)

        return labels

    def _get_label_filter_terms(self):
        return [
            term.strip().lower()
            for term in self.label_filter_var.get().split(',')
            if term.strip()
        ]

    def _get_count_filter(self, show_warning=False):
        mode = self.count_filter_mode_var.get()
        raw_value = self.count_filter_value_var.get().strip()

        if not raw_value:
            return None
        if mode == "Any":
            mode = "Equals"

        try:
            value = int(raw_value)
        except ValueError:
            if show_warning:
                messagebox.showwarning("Invalid Search", "Number of labels must be a whole number.")
            return False

        if value < 0:
            if show_warning:
                messagebox.showwarning("Invalid Search", "Number of labels cannot be negative.")
            return False

        return mode, value

    def _filters_are_active(self):
        return bool(self._get_label_filter_terms()) or self._get_count_filter() is not None

    def _matches_filters(self, image_file, label_terms, count_filter):
        labels = self._get_image_labels(image_file)

        if label_terms:
            normalized_labels = [label.lower() for label in labels]
            if not any(term in label for term in label_terms for label in normalized_labels):
                return False

        if count_filter:
            mode, value = count_filter
            label_count = len(labels)
            if mode == "Equals" and label_count != value:
                return False
            if mode == "At least" and label_count < value:
                return False
            if mode == "At most" and label_count > value:
                return False

        return True

    def apply_filters(self, preserve_file=None):
        if not self.dir_var.get():
            return

        count_filter = self._get_count_filter(show_warning=True)
        if count_filter is False:
            return

        if preserve_file is None and 0 <= self.current_index < len(self.image_files):
            preserve_file = self.image_files[self.current_index]

        label_terms = self._get_label_filter_terms()
        self.image_files = [
            image_file
            for image_file in self.all_image_files
            if self._matches_filters(image_file, label_terms, count_filter)
        ]

        if preserve_file in self.image_files:
            self.current_index = self.image_files.index(preserve_file)
        else:
            self.current_index = min(self.current_index, max(0, len(self.image_files) - 1))

        self.update_statistics()
        self._update_search_summary()

        if self.preview_window:
            if self.image_files:
                self.show_current_image()
            else:
                self.stop_preview()

    def clear_filters(self):
        self.label_filter_var.set("")
        self.count_filter_mode_var.set("Any")
        self.count_filter_value_var.set("")
        self.apply_filters()

    def _update_search_summary(self):
        total = len(self.all_image_files)
        matching = len(self.image_files)

        if not total:
            self.search_summary_var.set("No images loaded")
        elif self._filters_are_active():
            self.search_summary_var.set(f"Showing {matching} of {total} images matching the current search")
        else:
            self.search_summary_var.set(f"Showing all {total} images")

    def scan_folder_labels(self):
        """Scan every XML in the loaded directory and report distinct labels."""
        if not self.dir_var.get():
            messagebox.showwarning("Warning", "Please select a directory first.")
            return

        if not self.all_image_files:
            self.load_images()

        label_counts = {}
        scanned = 0
        for image_file in self.all_image_files:
            xml_path = self._xml_path_for_image(image_file)
            if not os.path.exists(xml_path):
                continue
            scanned += 1
            for label in self._get_image_labels(image_file):
                label_counts[label] = label_counts.get(label, 0) + 1

        if self.folder_labels_listbox is not None:
            self.folder_labels_listbox.delete(0, tk.END)
            for label, count in sorted(label_counts.items()):
                self.folder_labels_listbox.insert(tk.END, f"{label}  ({count})")

        if not label_counts:
            summary = f"Scanned {scanned} XML file(s); no labels found."
        else:
            summary = (f"Scanned {scanned} XML file(s) — "
                       f"{len(label_counts)} distinct label(s), "
                       f"{sum(label_counts.values())} total object(s).")
        if self.folder_labels_summary_var is not None:
            self.folder_labels_summary_var.set(summary)
        self.log_queue.put(summary)
        return label_counts

    def _selected_folder_label(self):
        if self.folder_labels_listbox is None:
            return None
        selection = self.folder_labels_listbox.curselection()
        if not selection:
            return None
        entry = self.folder_labels_listbox.get(selection[0])
        label = entry.rsplit("(", 1)[0].strip()
        return label or None

    def _apply_folder_label_filter(self):
        label = self._selected_folder_label()
        if not label:
            return
        self.label_filter_var.set(label)
        self.apply_filters()

    def _remember_folder_labels(self):
        if self.folder_labels_listbox is None:
            return
        entries = list(self.folder_labels_listbox.get(0, tk.END))
        if not entries:
            messagebox.showinfo("Info", "Scan the folder first to gather labels.")
            return
        labels = [entry.rsplit("(", 1)[0].strip() for entry in entries]
        labels = [label for label in labels if label]
        if not labels:
            return
        self.label_memory.merge_labels(labels)
        self.log_queue.put(
            f"Remembered {len(labels)} folder label(s) for the Annotation tab."
        )

    def _on_labels_changed(self, labels, current_label):
        """React to LabelMemory updates from another tab or this tab."""
        try:
            if (hasattr(self, 'reannotation_label_combo')
                    and self.reannotation_label_combo is not None
                    and self.reannotation_label_combo.winfo_exists()):
                self.reannotation_label_combo['values'] = labels
        except tk.TclError:
            pass

    def update_statistics(self):
        if not self.dir_var.get():
            return

        total_folder_images = len(self.all_image_files)
        total_images = len(self.image_files)
        images_with_xml = 0
        total_objects = 0
        label_counts = {}

        for img_file in self.image_files:
            xml_path = self._xml_path_for_image(img_file)
            labels = self._get_image_labels(img_file)
            if os.path.exists(xml_path):
                images_with_xml += 1
            for label in labels:
                total_objects += 1
                label_counts[label] = label_counts.get(label, 0) + 1

        stats_text = f"Total images in folder: {total_folder_images}\n"
        stats_text += f"Images in current search: {total_images}\n"
        stats_text += f"Images with annotations: {images_with_xml}\n"
        stats_text += f"Images without annotations: {total_images - images_with_xml}\n"
        stats_text += f"Total objects: {total_objects}\n"

        if label_counts:
            stats_text += "\nLabel distribution:\n"
            for label, count in sorted(label_counts.items()):
                stats_text += f"  {label}: {count}\n"

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def start_preview(self):
        if not self.dir_var.get():
            messagebox.showwarning("Warning", "Please select a directory first.")
            return

        if not self.image_files:
            messagebox.showinfo("Info", "No images match the current directory/search.")
            return

        self.current_index = 0
        self.show_current_image()

    def show_current_image(self):
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("Complete", "Reached end of images.")
            return

        image_file = self.image_files[self.current_index]
        image_path = os.path.join(self.dir_var.get(), image_file)

        self.create_preview_window(image_path)

    def create_preview_window(self, image_path):
        # Close previous window if exists
        if self.preview_window:
            self.preview_window.destroy()

        self.preview_window = tk.Toplevel(self.parent)
        self.preview_window.title(f"Preview: {os.path.basename(image_path)}")
        self.current_preview_boxes = []
        self.annotation_listbox = None

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", f"Could not load image: {image_path}")
            return

        h, w = image.shape[:2]
        max_width, max_height = 800, 600
        scale = min(max_width / w, max_height / h, 1.0)
        display_width = max(1, int(round(w * scale)))
        display_height = max(1, int(round(h * scale)))

        if scale < 1.0:
            display_image = cv2.resize(image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        else:
            display_image = image.copy()

        # Parse XML and draw boxes against the display image so the overlay
        # remains crisp and aligned even when the preview is resized.
        image_file = os.path.basename(image_path)
        xml_path = self._xml_path_for_image(image_file)
        boxes = []

        if os.path.exists(xml_path):
            try:
                boxes, annotation_size = self.parse_xml(xml_path)
                self.current_preview_boxes = boxes
                display_boxes = self.scale_boxes_for_display(
                    boxes,
                    annotation_size or (w, h),
                    (display_width, display_height)
                )
                self.draw_boxes(display_image, display_boxes)
            except Exception as exc:
                self.current_preview_boxes = []
                self.log_queue.put(f"Could not preview {Path(xml_path).name}: {exc}")
        else:
            self.current_preview_boxes = []

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

        # Convert to PIL and then to PhotoImage
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        # Display image
        label = tk.Label(self.preview_window, image=photo)
        label.image = photo  # Keep a reference
        label.pack()

        # Status bar
        status_frame = ttk.Frame(self.preview_window)
        status_frame.pack(fill=tk.X, pady=5)

        status_text = f"Image {self.current_index + 1} of {len(self.image_files)}"
        if os.path.exists(xml_path):
            status_text += f" | {len(boxes)} annotation(s)"
        else:
            status_text += " | No annotations"

        ttk.Label(status_frame, text=status_text).pack()

        self._add_preview_annotation_controls(boxes)

        # Update main window progress
        self.progress_label.config(
            text=f"Viewing: {self.current_index + 1}/{len(self.image_files)}"
        )

        # Bind keyboard events
        self.preview_window.bind('<Left>', lambda e: self.previous_image())
        self.preview_window.bind('<Right>', lambda e: self.next_image())
        self.preview_window.bind('a', lambda e: self.previous_image())
        self.preview_window.bind('d', lambda e: self.next_image())
        self.preview_window.bind('<Delete>', lambda e: self.delete_current())
        self.preview_window.bind('e', lambda e: self.delete_current())
        self.preview_window.bind('x', lambda e: self.delete_current_annotation_file())
        self.preview_window.bind('r', lambda e: self.reannotate_current())
        self.preview_window.bind('<space>', lambda e: self.toggle_auto_advance())
        self.preview_window.bind('<Escape>', lambda e: self.stop_preview())
        self.preview_window.bind('l', lambda e: self.toggle_labels())
        self.preview_window.bind('c', lambda e: self.cycle_color())

        # Focus window
        self.preview_window.focus_force()

        # Auto-advance if enabled
        if self.auto_advance_var.get():
            self.preview_window.after(1000, self.auto_next)

    def _add_preview_annotation_controls(self, boxes):
        controls_frame = ttk.Frame(self.preview_window, padding="5")
        controls_frame.pack(fill=tk.X)

        ttk.Button(controls_frame, text="Previous",
                   command=self.previous_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls_frame, text="Next",
                   command=self.next_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls_frame, text="Delete Annotation XML",
                   command=self.delete_current_annotation_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls_frame, text="Re-Annotate",
                   command=self.reannotate_current).pack(side=tk.LEFT, padx=3)

        if not boxes:
            self.annotation_listbox = None
            return

        annotation_frame = ttk.LabelFrame(self.preview_window, text="Annotations", padding="5")
        annotation_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        list_frame = ttk.Frame(annotation_frame)
        list_frame.pack(fill=tk.X)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.annotation_listbox = tk.Listbox(
            list_frame,
            height=min(6, max(2, len(boxes))),
            yscrollcommand=scrollbar.set
        )
        self.annotation_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.config(command=self.annotation_listbox.yview)

        for index, box in enumerate(boxes, start=1):
            self.annotation_listbox.insert(
                tk.END,
                f"{index}. {box['label']} ({int(box['xmin'])}, {int(box['ymin'])}) "
                f"to ({int(box['xmax'])}, {int(box['ymax'])})"
            )

        self.annotation_listbox.selection_set(0)

        annotation_button_frame = ttk.Frame(annotation_frame)
        annotation_button_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Button(annotation_button_frame, text="Delete Selected Annotation",
                   command=self.delete_selected_annotation).pack(side=tk.LEFT, padx=3)

    def _parse_number(self, value, cast_type=float):
        if value is None:
            return None

        try:
            if cast_type is int:
                return int(round(float(str(value).strip())))
            return cast_type(str(value).strip())
        except (TypeError, ValueError):
            return None

    def parse_xml(self, xml_file):
        """Parse XML file to extract bounding box information."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        annotation_size = None
        size_node = root.find('size')
        if size_node is not None:
            width = self._parse_number(size_node.findtext('width'), int)
            height = self._parse_number(size_node.findtext('height'), int)
            if width and height and width > 0 and height > 0:
                annotation_size = (width, height)

        boxes = []
        skipped_boxes = 0
        for object_index, obj in enumerate(root.findall('object')):
            bbox = obj.find('bndbox')
            if bbox is None:
                skipped_boxes += 1
                continue

            xmin = self._parse_number(bbox.findtext('xmin'))
            xmax = self._parse_number(bbox.findtext('xmax'))
            ymin = self._parse_number(bbox.findtext('ymin'))
            ymax = self._parse_number(bbox.findtext('ymax'))

            if None in (xmin, xmax, ymin, ymax):
                skipped_boxes += 1
                continue

            box_info = {
                'xmin': xmin,
                'xmax': xmax,
                'ymin': ymin,
                'ymax': ymax,
                'label': obj.findtext('name', default='object').strip() or 'object',
                'object_index': object_index
            }

            # Check for confidence/score if available
            score = self._parse_number(obj.findtext('score'))
            if score is not None:
                box_info['score'] = score

            boxes.append(box_info)

        if skipped_boxes:
            self.log_queue.put(
                f"Skipped {skipped_boxes} invalid box(es) in {os.path.basename(xml_file)}"
            )

        return boxes, annotation_size

    def scale_boxes_for_display(self, boxes, source_size, display_size):
        """Scale box coordinates into the preview image coordinate space."""
        display_width, display_height = display_size
        source_width, source_height = source_size

        if source_width <= 0 or source_height <= 0:
            source_width, source_height = display_size

        scaled_boxes = []
        for box in boxes:
            scaled_box = {
                'xmin': int(round(box['xmin'] * display_width / source_width)),
                'xmax': int(round(box['xmax'] * display_width / source_width)),
                'ymin': int(round(box['ymin'] * display_height / source_height)),
                'ymax': int(round(box['ymax'] * display_height / source_height)),
                'label': box['label'],
                'object_index': box.get('object_index')
            }

            if 'score' in box:
                scaled_box['score'] = box['score']

            scaled_box['xmin'] = max(0, min(display_width - 1, scaled_box['xmin']))
            scaled_box['xmax'] = max(0, min(display_width - 1, scaled_box['xmax']))
            scaled_box['ymin'] = max(0, min(display_height - 1, scaled_box['ymin']))
            scaled_box['ymax'] = max(0, min(display_height - 1, scaled_box['ymax']))

            if scaled_box['xmax'] <= scaled_box['xmin'] or scaled_box['ymax'] <= scaled_box['ymin']:
                continue

            scaled_boxes.append(scaled_box)

        return scaled_boxes

    def draw_boxes(self, image, boxes):
        """Draw bounding boxes on image."""
        color_map = {
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255)
        }

        color = color_map.get(self.box_color_var.get(), (0, 255, 0))
        image_height, image_width = image.shape[:2]
        min_dim = max(1, min(image_width, image_height))
        box_thickness = max(2, int(round(min_dim / 300)))
        font_scale = max(0.45, min_dim / 1200)
        font_thickness = max(1, box_thickness - 1)
        padding = max(2, box_thickness)

        for box in boxes:
            # Draw rectangle
            cv2.rectangle(image,
                          (box['xmin'], box['ymin']),
                          (box['xmax'], box['ymax']),
                          color, box_thickness)

            # Draw label if enabled
            if self.show_labels_var.get():
                label_text = box['label']
                if self.show_confidence_var.get() and 'score' in box:
                    label_text += f" ({box['score']:.2f})"

                text_size, baseline = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_thickness
                )
                text_width, text_height = text_size

                background_left = max(0, min(box['xmin'], image_width - text_width - (padding * 2)))

                if box['ymin'] > text_height + (padding * 2):
                    background_top = box['ymin'] - text_height - baseline - (padding * 2)
                else:
                    background_top = min(
                        image_height - text_height - baseline - (padding * 2),
                        box['ymax'] + padding
                    )

                background_top = max(0, background_top)
                background_right = min(image_width - 1, background_left + text_width + (padding * 2))
                background_bottom = min(image_height - 1, background_top + text_height + baseline + (padding * 2))
                text_x = background_left + padding
                text_y = min(image_height - baseline - 1, background_top + text_height + padding)

                # Draw background rectangle for text
                cv2.rectangle(image,
                              (background_left, background_top),
                              (background_right, background_bottom),
                              color, -1)

                # Draw text
                cv2.putText(image, label_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (255, 255, 255), font_thickness)

        return image

    def _get_current_file(self):
        if not self.image_files:
            return None
        if self.current_index < 0 or self.current_index >= len(self.image_files):
            return None
        return self.image_files[self.current_index]

    def _write_xml_file(self, root, xml_path):
        xml_str = ET.tostring(root, encoding='utf-8')
        dom = parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="    ")
        with open(xml_path, 'w', encoding='utf-8') as file:
            file.write(pretty_xml)

    def delete_selected_annotation(self):
        current_file = self._get_current_file()
        if not current_file:
            return

        if self.annotation_listbox is None:
            messagebox.showinfo("Info", "There are no annotations to delete.")
            return

        selection = self.annotation_listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Select an annotation to delete first.")
            return

        selected_index = selection[0]
        if selected_index >= len(self.current_preview_boxes):
            return

        selected_box = self.current_preview_boxes[selected_index]
        xml_path = self._xml_path_for_image(current_file)
        if not os.path.exists(xml_path):
            messagebox.showinfo("Info", "No XML annotation file exists for this image.")
            return

        result = messagebox.askyesno(
            "Delete Annotation",
            f"Delete annotation '{selected_box['label']}' from {current_file}?"
        )
        if not result:
            return

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall('object')
            object_index = selected_box.get('object_index')
            if object_index is None:
                object_index = selected_index

            if object_index < 0 or object_index >= len(objects):
                messagebox.showerror("Error", "Could not match the selected annotation in the XML file.")
                return

            root.remove(objects[object_index])

            if root.findall('object'):
                self._write_xml_file(root, xml_path)
                self.log_queue.put(f"Deleted one annotation from {Path(xml_path).name}")
            else:
                os.remove(xml_path)
                self.log_queue.put(f"Deleted XML after removing the last annotation: {Path(xml_path).name}")

            self.apply_filters(preserve_file=current_file)

        except Exception as exc:
            messagebox.showerror("Error", f"Could not delete annotation: {exc}")

    def delete_current_annotation_file(self, confirm=True):
        current_file = self._get_current_file()
        if not current_file:
            return False

        xml_path = self._xml_path_for_image(current_file)
        if not os.path.exists(xml_path):
            messagebox.showinfo("Info", "No XML annotation file exists for this image.")
            return False

        if confirm:
            result = messagebox.askyesno(
                "Delete Annotation XML",
                f"Delete only the XML annotations for {current_file}?\n\nThe image file will remain."
            )
            if not result:
                return False

        try:
            os.remove(xml_path)
            self.log_queue.put(f"Deleted annotation XML: {Path(xml_path).name}")
            self.apply_filters(preserve_file=current_file)
            return True
        except Exception as exc:
            messagebox.showerror("Error", f"Could not delete XML annotation: {exc}")
            return False

    def _labels_for_reannotation(self, current_file):
        """Gather all labels for the re-annotation dropdown.

        Deliberately does NOT persist anything. We only mutate LabelMemory on
        save so the dropdown ordering stays stable while the user is picking a
        value. Ordering puts the active current_label first (if any), then the
        remembered labels, then anything else discovered in XML files.
        """
        ordered = []

        def _append(label):
            if label is None:
                return
            text = str(label).strip()
            if text and text not in ordered:
                ordered.append(text)

        _append(self.label_memory.current_label)
        for label in self.label_memory.labels:
            _append(label)
        for label in self._get_image_labels(current_file):
            _append(label)
        for image_file in self.all_image_files:
            if image_file == current_file:
                continue
            for label in self._get_image_labels(image_file):
                _append(label)

        return ordered

    def reannotate_current(self):
        current_file = self._get_current_file()
        if not current_file:
            return

        image_path = self._image_path(current_file)
        if not os.path.exists(image_path):
            messagebox.showerror("Error", f"Could not find image: {image_path}")
            return

        xml_path = self._xml_path_for_image(current_file)
        if os.path.exists(xml_path):
            result = messagebox.askyesno(
                "Re-Annotate Image",
                "Open a blank annotation canvas for this image?\n\n"
                "Saving will replace the current XML annotations. Canceling keeps the current XML."
            )
            if not result:
                return

        self.create_reannotation_window(image_path, self._labels_for_reannotation(current_file))

    def create_reannotation_window(self, image_path, label_choices):
        self.close_reannotation_window(refresh=False)

        self.reannotation_image_path = image_path
        self.reannotation_image = cv2.imread(image_path)
        if self.reannotation_image is None:
            messagebox.showerror("Error", f"Could not load image: {image_path}")
            return

        self.annotation_window = tk.Toplevel(self.parent)
        self.annotation_window.title(f"Re-Annotate: {os.path.basename(image_path)}")

        rgb_image = cv2.cvtColor(self.reannotation_image, cv2.COLOR_BGR2RGB)
        h, w = self.reannotation_image.shape[:2]
        max_width, max_height = 900, 650
        scale = min(max_width / w, max_height / h, 1.0)
        display_width = max(1, int(round(w * scale)))
        display_height = max(1, int(round(h * scale)))
        self.reannotation_scale = scale

        if scale < 1.0:
            pil_image = Image.fromarray(rgb_image).resize((display_width, display_height), Image.LANCZOS)
        else:
            pil_image = Image.fromarray(rgb_image)

        self.reannotation_photo_image = ImageTk.PhotoImage(pil_image)
        self.reannotation_canvas = tk.Canvas(
            self.annotation_window,
            width=display_width,
            height=display_height
        )
        self.reannotation_canvas.pack()
        self.reannotation_canvas.create_image(0, 0, anchor=tk.NW, image=self.reannotation_photo_image)

        self.reannotation_canvas.bind("<ButtonPress-1>", self._on_reannotation_mouse_down)
        self.reannotation_canvas.bind("<B1-Motion>", self._on_reannotation_mouse_drag)
        self.reannotation_canvas.bind("<ButtonRelease-1>", self._on_reannotation_mouse_up)

        control_frame = ttk.Frame(self.annotation_window, padding="5")
        control_frame.pack(fill=tk.X)

        current_label = self.label_memory.current_label
        if current_label not in label_choices:
            current_label = label_choices[0] if label_choices else ""

        self.reannotation_label_var = tk.StringVar(value=current_label)
        ttk.Label(control_frame, text="Label:").pack(side=tk.LEFT, padx=(0, 5))
        self.reannotation_label_combo = ttk.Combobox(
            control_frame,
            textvariable=self.reannotation_label_var,
            values=label_choices,
            state="readonly",
            width=24
        )
        self.reannotation_label_combo.pack(side=tk.LEFT, padx=(0, 8))
        # NOTE: do not mutate the combobox ``values`` mid-interaction. Replacing
        # the list while the user is picking an entry causes Tkinter to keep
        # the previously selected index and re-render the value at that index
        # from the new list, which was the source of the MAKE_BLACK_BOX ->
        # dumbbell -> "other" bug.

        self.reannotation_new_label_var = tk.StringVar()
        ttk.Label(control_frame, text="New:").pack(side=tk.LEFT, padx=(0, 4))
        new_label_entry = ttk.Entry(
            control_frame,
            textvariable=self.reannotation_new_label_var,
            width=16
        )
        new_label_entry.pack(side=tk.LEFT, padx=(0, 4))
        new_label_entry.bind("<Return>", lambda e: self._add_reannotation_label())
        ttk.Button(control_frame, text="Add Label",
                   command=self._add_reannotation_label).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Button(control_frame, text="Undo", command=self.undo_reannotation).pack(side=tk.LEFT, padx=3)
        ttk.Button(control_frame, text="Clear All", command=self.clear_reannotation).pack(side=tk.LEFT, padx=3)
        ttk.Button(control_frame, text="Save", command=self.save_reannotation).pack(side=tk.RIGHT, padx=3)
        ttk.Button(control_frame, text="Cancel", command=self.close_reannotation_window).pack(side=tk.RIGHT, padx=3)

        help_text = ("Click and drag to draw boxes. Choose an existing label in the dropdown, "
                     "or type a new one and press Add Label.")
        ttk.Label(self.annotation_window, text=help_text).pack(fill=tk.X, padx=8, pady=(0, 6))

        self.reannotation_annotations = []
        self.reannotation_rect_ids = []
        self.reannotation_text_ids = []
        self.reannotation_current_rect = None
        self.reannotation_start_x = None
        self.reannotation_start_y = None

        self.annotation_window.bind('u', lambda e: self.undo_reannotation())
        self.annotation_window.bind('<Escape>', lambda e: self.close_reannotation_window())
        self.annotation_window.protocol("WM_DELETE_WINDOW", self.close_reannotation_window)
        self.annotation_window.focus_force()

    def _on_reannotation_mouse_down(self, event):
        self.reannotation_start_x = event.x
        self.reannotation_start_y = event.y
        self.reannotation_current_rect = self.reannotation_canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline='red',
            width=2
        )

    def _on_reannotation_mouse_drag(self, event):
        if self.reannotation_current_rect:
            self.reannotation_canvas.coords(
                self.reannotation_current_rect,
                self.reannotation_start_x,
                self.reannotation_start_y,
                event.x,
                event.y
            )

    def _on_reannotation_mouse_up(self, event):
        if not self.reannotation_current_rect:
            return

        x1, y1, x2, y2 = self.reannotation_canvas.coords(self.reannotation_current_rect)
        if abs(x2 - x1) <= 5 or abs(y2 - y1) <= 5:
            self.reannotation_canvas.delete(self.reannotation_current_rect)
            self.reannotation_current_rect = None
            return

        label = self.reannotation_label_var.get().strip() or "object"
        self.reannotation_canvas.itemconfig(self.reannotation_current_rect, outline='green')
        text_id = self.reannotation_canvas.create_text(
            min(x1, x2),
            min(y1, y2) - 5,
            text=label,
            anchor=tk.SW,
            fill='green',
            font=('Arial', 10, 'bold')
        )

        self.reannotation_annotations.append({
            'x1': min(x1, x2),
            'y1': min(y1, y2),
            'x2': max(x1, x2),
            'y2': max(y1, y2),
            'label': label
        })
        self.reannotation_rect_ids.append(self.reannotation_current_rect)
        self.reannotation_text_ids.append(text_id)
        self.reannotation_current_rect = None
        self.log_queue.put(f"Added {label} annotation")

    def _add_reannotation_label(self):
        """Add a user-typed label to the re-annotation dropdown.

        Persists it to LabelMemory (so other tabs pick it up immediately) and
        updates the combo's ``values`` while no selection is being mutated.
        """
        new_label = self.reannotation_new_label_var.get().strip()
        if not new_label:
            return

        self.label_memory.merge_labels([new_label], current_label=new_label)
        combo = getattr(self, 'reannotation_label_combo', None)
        if combo is not None and combo.winfo_exists():
            values = list(combo['values'])
            if new_label not in values:
                values.append(new_label)
                combo['values'] = values
            self.reannotation_label_var.set(new_label)

        self.reannotation_new_label_var.set("")
        self.log_queue.put(f"Added label to memory: {new_label}")

    def undo_reannotation(self):
        if not self.reannotation_annotations:
            return

        self.reannotation_annotations.pop()
        if self.reannotation_rect_ids:
            self.reannotation_canvas.delete(self.reannotation_rect_ids.pop())
        if self.reannotation_text_ids:
            self.reannotation_canvas.delete(self.reannotation_text_ids.pop())
        self.log_queue.put("Removed last re-annotation")

    def clear_reannotation(self):
        self.reannotation_annotations = []
        for rect_id in self.reannotation_rect_ids:
            self.reannotation_canvas.delete(rect_id)
        for text_id in self.reannotation_text_ids:
            self.reannotation_canvas.delete(text_id)
        self.reannotation_rect_ids = []
        self.reannotation_text_ids = []
        self.log_queue.put("Cleared re-annotations")

    def close_reannotation_window(self, refresh=True):
        image_file = None
        if self.reannotation_image_path:
            image_file = os.path.basename(self.reannotation_image_path)

        if self.annotation_window:
            self.annotation_window.destroy()
            self.annotation_window = None

        if refresh and image_file:
            self.apply_filters(preserve_file=image_file)

    def save_reannotation(self):
        if not self.reannotation_annotations:
            messagebox.showwarning("Warning", "Draw at least one annotation before saving.")
            return

        self.label_memory.merge_labels(
            [ann['label'] for ann in self.reannotation_annotations],
            current_label=self.reannotation_label_var.get()
        )

        folder = os.path.basename(os.path.dirname(self.reannotation_image_path))
        filename = os.path.basename(self.reannotation_image_path)
        h, w, c = self.reannotation_image.shape

        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = folder or ""
        ET.SubElement(root, "filename").text = filename
        ET.SubElement(root, "path").text = filename

        source = ET.SubElement(root, "source")
        ET.SubElement(source, "database").text = "roboflow.com"

        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = str(c)

        ET.SubElement(root, "segmented").text = "0"

        valid_annotations = 0
        scale = self.reannotation_scale or 1.0
        for ann in self.reannotation_annotations:
            xmin = max(0, min(w - 1, int(round(ann['x1'] / scale))))
            ymin = max(0, min(h - 1, int(round(ann['y1'] / scale))))
            xmax = max(0, min(w - 1, int(round(ann['x2'] / scale))))
            ymax = max(0, min(h - 1, int(round(ann['y2'] / scale))))

            if xmax <= xmin or ymax <= ymin:
                continue

            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = ann['label']
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            ET.SubElement(obj, "occluded").text = "0"

            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(xmin)
            ET.SubElement(bbox, "xmax").text = str(xmax)
            ET.SubElement(bbox, "ymin").text = str(ymin)
            ET.SubElement(bbox, "ymax").text = str(ymax)
            valid_annotations += 1

        if not valid_annotations:
            messagebox.showwarning("Warning", "No valid annotations were drawn.")
            return

        xml_path = self._xml_path_for_image(filename)
        try:
            self._write_xml_file(root, xml_path)
            self.log_queue.put(f"Saved re-annotations to: {Path(xml_path).name}")
            self.close_reannotation_window(refresh=False)
            self.apply_filters(preserve_file=filename)
        except Exception as exc:
            messagebox.showerror("Error", f"Could not save re-annotations: {exc}")

    def previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_current_image()

    def next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_current_image()

    def auto_next(self):
        if self.auto_advance_var.get() and self.preview_window:
            self.next_image()

    def delete_current(self):
        if not self.image_files:
            return

        current_file = self.image_files[self.current_index]
        image_path = self._image_path(current_file)
        xml_path = self._xml_path_for_image(current_file)

        result = messagebox.askyesno(
            "Confirm Delete",
            f"Delete {current_file} and its XML file?\n\nThis cannot be undone!"
        )

        if result:
            try:
                # Delete image
                if os.path.exists(image_path):
                    os.remove(image_path)
                    self.log_queue.put(f"Deleted image: {current_file}")

                # Delete XML
                if os.path.exists(xml_path):
                    os.remove(xml_path)
                    self.log_queue.put(f"Deleted XML: {Path(current_file).stem}.xml")

                # Remove from list
                if current_file in self.all_image_files:
                    self.all_image_files.remove(current_file)
                self.image_files.pop(self.current_index)

                # Update statistics
                self.update_statistics()
                self._update_search_summary()

                # Show next image
                if self.current_index >= len(self.image_files):
                    self.current_index = max(0, len(self.image_files) - 1)

                if self.image_files:
                    self.show_current_image()
                else:
                    self.stop_preview()
                    messagebox.showinfo("Info", "No more images in directory.")

            except Exception as e:
                messagebox.showerror("Error", f"Could not delete files: {str(e)}")

    def stop_preview(self):
        if self.preview_window:
            self.preview_window.destroy()
            self.preview_window = None

    def toggle_auto_advance(self):
        self.auto_advance_var.set(not self.auto_advance_var.get())
        self.log_queue.put(f"Auto-advance: {'ON' if self.auto_advance_var.get() else 'OFF'}")

    def toggle_labels(self):
        self.show_labels_var.set(not self.show_labels_var.get())
        self.show_current_image()

    def cycle_color(self):
        colors = ["green", "red", "blue", "yellow", "cyan", "magenta"]
        current_idx = colors.index(self.box_color_var.get())
        next_idx = (current_idx + 1) % len(colors)
        self.box_color_var.set(colors[next_idx])
        self.show_current_image()
