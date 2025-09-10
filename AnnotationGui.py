import os
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from xml.dom.minidom import parseString

import cv2
from PIL import Image, ImageTk

# TODO add previous button that also shows the xml and bbox if present
class AnnotationTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.current_image = None
        self.current_image_path = None
        self.annotations = []
        self.current_annotation = None
        self.drawing = False
        self.setup_ui()

    def setup_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.parent.columnconfigure(0, weight=1)
        self.parent.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)

        # Section 1: Directory Selection
        ttk.Label(main_frame, text="Image Directory",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, columnspan=3, pady=(0, 10))

        ttk.Label(main_frame, text="Image Directory:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.image_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.image_dir_var, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.browse_image_dir).grid(row=1, column=2, padx=(5, 0))

        # Section 2: Label Management
        ttk.Label(main_frame, text="Label Management",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=2, column=0, columnspan=3, pady=(20, 10))

        label_frame = ttk.LabelFrame(main_frame, text="Object Labels", padding="10")
        label_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))

        # Label list
        label_list_frame = ttk.Frame(label_frame)
        label_list_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create listbox with scrollbar
        scrollbar = ttk.Scrollbar(label_list_frame)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.label_listbox = tk.Listbox(label_list_frame, height=6, yscrollcommand=scrollbar.set)
        self.label_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.config(command=self.label_listbox.yview)

        # Label entry and buttons
        entry_frame = ttk.Frame(label_frame)
        entry_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))

        ttk.Label(entry_frame, text="New Label:").grid(row=0, column=0, padx=(0, 5))
        self.new_label_var = tk.StringVar()
        self.label_entry = ttk.Entry(entry_frame, textvariable=self.new_label_var, width=20)
        self.label_entry.grid(row=0, column=1, padx=(0, 5))
        self.label_entry.bind('<Return>', lambda e: self.add_label())

        ttk.Button(entry_frame, text="Add", command=self.add_label).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(entry_frame, text="Remove", command=self.remove_label).grid(row=0, column=3)

        # Current label selection
        ttk.Label(label_frame, text="Current Label:").grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        self.current_label_var = tk.StringVar()
        self.current_label_combo = ttk.Combobox(label_frame, textvariable=self.current_label_var,
                                                state="readonly", width=20)
        self.current_label_combo.grid(row=2, column=1, sticky=tk.W, pady=(10, 0))

        # Add default label
        self.label_listbox.insert(tk.END, "kettlebell")
        self.update_label_combo()

        # Section 3: Annotation Controls
        ttk.Label(main_frame, text="Annotation Controls",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=4, column=0, columnspan=3, pady=(20, 10))

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=5, column=0, columnspan=3, pady=(0, 10))

        ttk.Button(control_frame, text="Start Annotation",
                   command=self.start_annotation).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Previous Image",
                   command=self.previous_image).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Next Image",
                   command=self.next_image).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Skip Image",
                   command=self.skip_image).grid(row=0, column=3, padx=5)

        # Annotation info
        info_frame = ttk.LabelFrame(main_frame, text="Annotation Info", padding="10")
        info_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        self.info_text = tk.Text(info_frame, height=4, wrap=tk.WORD)
        self.info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.update_info("Ready to start annotation. Select an image directory and click 'Start Annotation'.")

        # Progress
        self.progress_label = ttk.Label(main_frame, text="")
        self.progress_label.grid(row=7, column=0, columnspan=3, pady=(10, 0))

        # Initialize
        self.image_files = []
        self.current_index = 0
        self.ann_window = None

    def browse_image_dir(self):
        directory = filedialog.askdirectory(
            title="Select Image Directory",
            initialdir=self.app_root
        )
        if directory:
            self.image_dir_var.set(directory)
            self.load_images()

    def load_images(self):
        if not self.image_dir_var.get():
            return

        image_extensions = {'.jpg', '.jpeg', '.png'}
        self.image_files = []

        for file in sorted(os.listdir(self.image_dir_var.get())):
            if Path(file).suffix.lower() in image_extensions:
                # Check if XML already exists
                xml_path = os.path.join(self.image_dir_var.get(),
                                        Path(file).stem + '.xml')
                if not os.path.exists(xml_path):
                    self.image_files.append(file)

        self.log_queue.put(f"Found {len(self.image_files)} images without annotations")
        self.update_progress()

    def add_label(self):
        label = self.new_label_var.get().strip()
        if label and label not in self.label_listbox.get(0, tk.END):
            self.label_listbox.insert(tk.END, label)
            self.new_label_var.set("")
            self.update_label_combo()
            self.log_queue.put(f"Added label: {label}")

    def remove_label(self):
        selection = self.label_listbox.curselection()
        if selection:
            label = self.label_listbox.get(selection[0])
            self.label_listbox.delete(selection[0])
            self.update_label_combo()
            self.log_queue.put(f"Removed label: {label}")

    def update_label_combo(self):
        labels = list(self.label_listbox.get(0, tk.END))
        self.current_label_combo['values'] = labels
        if labels and not self.current_label_var.get():
            self.current_label_var.set(labels[0])

    def update_info(self, message):
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, message)

    def update_progress(self):
        if self.image_files:
            total = len(self.image_files)
            remaining = total - self.current_index
            self.progress_label.config(
                text=f"Image {self.current_index + 1} of {total} | Remaining: {remaining}"
            )
        else:
            self.progress_label.config(text="No images loaded")

    def start_annotation(self):
        if not self.image_dir_var.get():
            messagebox.showwarning("Warning", "Please select an image directory first.")
            return

        if not self.current_label_var.get():
            messagebox.showwarning("Warning", "Please select a label first.")
            return

        if not self.image_files:
            messagebox.showinfo("Info", "No images without annotations found.")
            return

        self.current_index = 0
        self.show_current_image()

    def show_current_image(self):
        if self.current_index >= len(self.image_files):
            messagebox.showinfo("Complete", "All images have been processed!")
            return

        image_file = self.image_files[self.current_index]
        self.current_image_path = os.path.join(self.image_dir_var.get(), image_file)

        # Create annotation window using Tkinter Canvas instead of OpenCV
        self.create_annotation_window()

    def create_annotation_window(self):
        # Close previous window if exists
        if self.ann_window:
            self.ann_window.destroy()

        self.ann_window = tk.Toplevel(self.parent)
        self.ann_window.title(f"Annotate: {os.path.basename(self.current_image_path)}")

        # Load image
        self.original_image = cv2.imread(self.current_image_path)
        if self.original_image is None:
            messagebox.showerror("Error", f"Could not load image: {self.current_image_path}")
            return

        # Convert BGR to RGB for display
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        # Resize for display if needed
        h, w = self.original_image.shape[:2]
        max_width, max_height = 800, 600

        scale = min(max_width / w, max_height / h, 1.0)
        new_width = int(w * scale)
        new_height = int(h * scale)

        self.display_scale = scale

        # Create PIL image
        if scale < 1.0:
            pil_image = Image.fromarray(rgb_image).resize((new_width, new_height), Image.LANCZOS)
        else:
            pil_image = Image.fromarray(rgb_image)

        self.photo_image = ImageTk.PhotoImage(pil_image)

        # Create canvas
        self.canvas = tk.Canvas(self.ann_window, width=new_width, height=new_height)
        self.canvas.pack()

        # Display image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo_image)

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Bind keyboard shortcuts
        self.ann_window.bind('n', lambda e: self.next_image())
        self.ann_window.bind('s', lambda e: self.skip_image())
        self.ann_window.bind('u', lambda e: self.undo_last())
        self.ann_window.bind('<Escape>', lambda e: self.close_annotation_window())

        # Control frame
        control_frame = ttk.Frame(self.ann_window, padding="5")
        control_frame.pack(fill=tk.X)

        ttk.Label(control_frame, text=f"Label: {self.current_label_var.get()}").pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Undo (u)", command=self.undo_last).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear All", command=self.clear_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next (n)", command=self.next_image).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Skip (s)", command=self.skip_image).pack(side=tk.RIGHT, padx=5)

        # Instructions
        self.update_info(
            f"Annotating: {os.path.basename(self.current_image_path)}\n"
            f"Current label: {self.current_label_var.get()}\n"
            "Click and drag to draw bounding box. Shortcuts: n=next, s=skip, u=undo, ESC=close"
        )

        # Reset annotations for new image
        self.annotations = []
        self.rect_ids = []
        self.text_ids = []
        self.current_rect = None
        self.start_x = None
        self.start_y = None

        # Handle window close
        self.ann_window.protocol("WM_DELETE_WINDOW", self.close_annotation_window)

        # Focus the window
        self.ann_window.focus_force()

    def on_mouse_down(self, event):
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y

        # Create rectangle
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )

    def on_mouse_drag(self, event):
        if self.drawing and self.current_rect:
            # Update rectangle
            self.canvas.coords(self.current_rect,
                               self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.drawing and self.current_rect:
            self.drawing = False

            # Finalize rectangle
            x1, y1, x2, y2 = self.canvas.coords(self.current_rect)

            # Ensure we have a valid box
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                # Change color to green
                self.canvas.itemconfig(self.current_rect, outline='green')

                # Add label text
                label_text = self.canvas.create_text(
                    x1, y1 - 5, text=self.current_label_var.get(),
                    anchor=tk.SW, fill='green', font=('Arial', 10, 'bold')
                )

                # Store annotation
                self.annotations.append({
                    'x1': min(x1, x2),
                    'y1': min(y1, y2),
                    'x2': max(x1, x2),
                    'y2': max(y1, y2),
                    'label': self.current_label_var.get()
                })

                self.rect_ids.append(self.current_rect)
                self.text_ids.append(label_text)

                self.log_queue.put(f"Added {self.current_label_var.get()} annotation")
            else:
                # Remove invalid rectangle
                self.canvas.delete(self.current_rect)

            self.current_rect = None

    def undo_last(self):
        if self.annotations:
            self.annotations.pop()
            if self.rect_ids:
                self.canvas.delete(self.rect_ids.pop())
            if self.text_ids:
                self.canvas.delete(self.text_ids.pop())
            self.log_queue.put("Removed last annotation")

    def clear_annotations(self):
        self.annotations = []
        for rect_id in self.rect_ids:
            self.canvas.delete(rect_id)
        for text_id in self.text_ids:
            self.canvas.delete(text_id)
        self.rect_ids = []
        self.text_ids = []
        self.log_queue.put("Cleared all annotations")

    def close_annotation_window(self):
        if self.ann_window:
            self.ann_window.destroy()
            self.ann_window = None

    def save_current_annotations(self):
        if not self.annotations:
            return

        # Create XML
        annotation = ET.Element("annotation")

        # Add basic info
        folder = os.path.basename(os.path.dirname(self.current_image_path))
        filename = os.path.basename(self.current_image_path)

        ET.SubElement(annotation, "folder").text = folder or ""
        ET.SubElement(annotation, "filename").text = filename
        ET.SubElement(annotation, "path").text = filename

        # Add source
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "roboflow.com"

        # Add size
        h, w, c = self.original_image.shape
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = str(c)

        ET.SubElement(annotation, "segmented").text = "0"

        # Add objects
        for ann in self.annotations:
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = ann['label']
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            ET.SubElement(obj, "occluded").text = "0"

            # Convert display coordinates back to original
            xmin = int(ann['x1'] / self.display_scale)
            ymin = int(ann['y1'] / self.display_scale)
            xmax = int(ann['x2'] / self.display_scale)
            ymax = int(ann['y2'] / self.display_scale)

            bbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bbox, "xmin").text = str(xmin)
            ET.SubElement(bbox, "xmax").text = str(xmax)
            ET.SubElement(bbox, "ymin").text = str(ymin)
            ET.SubElement(bbox, "ymax").text = str(ymax)

        # Pretty print XML
        xml_str = ET.tostring(annotation, encoding='utf-8')
        dom = parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="	")

        # Save XML
        xml_path = os.path.join(
            os.path.dirname(self.current_image_path),
            Path(self.current_image_path).stem + '.xml'
        )

        with open(xml_path, 'w') as f:
            f.write(pretty_xml)

        self.log_queue.put(f"Saved annotations to: {os.path.basename(xml_path)}")

    def next_image(self):
        self.save_current_annotations()
        self.close_annotation_window()

        self.current_index += 1
        self.update_progress()

        if self.current_index < len(self.image_files):
            self.show_current_image()
        else:
            messagebox.showinfo("Complete", "All images have been annotated!")

    def previous_image(self):
        if self.current_index > 0:
            self.close_annotation_window()

            self.current_index -= 1
            self.update_progress()
            self.show_current_image()

    def skip_image(self):
        self.close_annotation_window()

        self.current_index += 1
        self.update_progress()

        if self.current_index < len(self.image_files):
            self.show_current_image()
        else:
            messagebox.showinfo("Complete", "Reached end of images!")
