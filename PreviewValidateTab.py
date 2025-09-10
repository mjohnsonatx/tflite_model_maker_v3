import os
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk, ImageDraw


class PreviewValidateTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.current_index = 0
        self.image_files = []
        self.preview_window = None
        self.setup_ui()

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

        # Section 3: Controls
        ttk.Label(main_frame, text="Controls",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=3, column=0, columnspan=3, pady=(20, 10))

        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=4, column=0, columnspan=3, pady=(0, 10))

        ttk.Button(control_frame, text="Start Preview",
                   command=self.start_preview).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Previous",
                   command=self.previous_image).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Next",
                   command=self.next_image).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Delete Current",
                   command=self.delete_current).grid(row=0, column=3, padx=5)
        ttk.Button(control_frame, text="Stop Preview",
                   command=self.stop_preview).grid(row=0, column=4, padx=5)

        # Info section
        info_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        info_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        info_text = """Keyboard shortcuts during preview:
• Arrow Left/Right or A/D: Navigate between images
• Delete or E: Delete current image and its XML file
• Space: Toggle auto-advance
• Escape: Close preview window
• L: Toggle label display
• C: Change box color

Note: Deleted files are permanently removed. Use with caution!"""

        self.info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        self.info_label.pack(anchor=tk.W)

        # Statistics
        self.stats_frame = ttk.LabelFrame(main_frame, text="Statistics", padding="10")
        self.stats_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))

        self.stats_text = tk.Text(self.stats_frame, height=4, wrap=tk.WORD)
        self.stats_text.pack(fill="both", expand=True)

        # Progress
        self.progress_label = ttk.Label(main_frame, text="")
        self.progress_label.grid(row=7, column=0, columnspan=3, pady=(10, 0))

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
        self.image_files = []

        for file in sorted(os.listdir(self.dir_var.get())):
            if Path(file).suffix.lower() in image_extensions:
                self.image_files.append(file)

        self.update_statistics()
        self.log_queue.put(f"Found {len(self.image_files)} images in directory")

    def update_statistics(self):
        if not self.dir_var.get():
            return

        total_images = len(self.image_files)
        images_with_xml = 0
        total_objects = 0
        label_counts = {}

        for img_file in self.image_files:
            xml_path = os.path.join(self.dir_var.get(), Path(img_file).stem + '.xml')
            if os.path.exists(xml_path):
                images_with_xml += 1
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for obj in root.findall('object'):
                        total_objects += 1
                        label = obj.find('name').text
                        label_counts[label] = label_counts.get(label, 0) + 1
                except:
                    pass

        stats_text = f"Total images: {total_images}\n"
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
            messagebox.showinfo("Info", "No images found in directory.")
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

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            messagebox.showerror("Error", f"Could not load image: {image_path}")
            return

        # Parse XML and draw boxes
        xml_path = os.path.join(self.dir_var.get(),
                                Path(image_path).stem + '.xml')

        if os.path.exists(xml_path):
            boxes = self.parse_xml(xml_path)
            image = self.draw_boxes(image, boxes)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize for display if needed
        h, w = image.shape[:2]
        max_width, max_height = 800, 600
        scale = min(max_width / w, max_height / h, 1.0)

        if scale < 1.0:
            new_width = int(w * scale)
            new_height = int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_width, new_height))

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
            status_text += " | Has annotations"
        else:
            status_text += " | No annotations"

        ttk.Label(status_frame, text=status_text).pack()

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
        self.preview_window.bind('<space>', lambda e: self.toggle_auto_advance())
        self.preview_window.bind('<Escape>', lambda e: self.stop_preview())
        self.preview_window.bind('l', lambda e: self.toggle_labels())
        self.preview_window.bind('c', lambda e: self.cycle_color())

        # Focus window
        self.preview_window.focus_force()

        # Auto-advance if enabled
        if self.auto_advance_var.get():
            self.preview_window.after(1000, self.auto_next)

    def parse_xml(self, xml_file):
        """Parse XML file to extract bounding box information."""
        tree = ET.parse(xml_file)
        root = tree.getroot()

        boxes = []
        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            label = obj.find('name').text

            box_info = {
                'xmin': int(bbox.find('xmin').text),
                'xmax': int(bbox.find('xmax').text),
                'ymin': int(bbox.find('ymin').text),
                'ymax': int(bbox.find('ymax').text),
                'label': label
            }

            # Check for confidence/score if available
            score = obj.find('score')
            if score is not None:
                box_info['score'] = float(score.text)

            boxes.append(box_info)

        return boxes

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

        for box in boxes:
            # Draw rectangle
            cv2.rectangle(image,
                          (box['xmin'], box['ymin']),
                          (box['xmax'], box['ymax']),
                          color, 2)

            # Draw label if enabled
            if self.show_labels_var.get():
                label_text = box['label']
                if self.show_confidence_var.get() and 'score' in box:
                    label_text += f" ({box['score']:.2f})"

                # Calculate text position
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                text_x = box['xmin']
                text_y = box['ymin'] - 5 if box['ymin'] > 20 else box['ymax'] + 20

                # Draw background rectangle for text
                cv2.rectangle(image,
                              (text_x, text_y - text_size[1] - 2),
                              (text_x + text_size[0], text_y + 2),
                              color, -1)

                # Draw text
                cv2.putText(image, label_text,
                            (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1)

        return image

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
        image_path = os.path.join(self.dir_var.get(), current_file)
        xml_path = os.path.join(self.dir_var.get(),
                                Path(current_file).stem + '.xml')

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
                self.image_files.pop(self.current_index)

                # Update statistics
                self.update_statistics()

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
