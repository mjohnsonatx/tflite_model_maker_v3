import os
import threading
import tkinter as tk
import xml.etree.ElementTree as ET
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk


class AugmentationTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.augment_thread = None
        self.stop_flag = threading.Event()
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
        ttk.Label(main_frame, text="Data Augmentation",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=0, column=0, columnspan=3, pady=(0, 10))

        # Source directory
        ttk.Label(main_frame, text="Source Directory:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        self.source_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.source_dir_var, width=50).grid(row=1, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.browse_source).grid(row=1, column=2, padx=(5, 0))

        # Destination directory
        ttk.Label(main_frame, text="Output Directory:").grid(row=2, column=0, sticky=tk.W, padx=(0, 10))
        self.dest_dir_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.dest_dir_var, width=50).grid(row=2, column=1, sticky=(tk.W, tk.E))
        ttk.Button(main_frame, text="Browse", command=self.browse_dest).grid(row=2, column=2, padx=(5, 0))

        # Section 2: Augmentation Settings
        ttk.Label(main_frame, text="Augmentation Settings",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=3, column=0, columnspan=3, pady=(20, 10))

        settings_frame = ttk.LabelFrame(main_frame, text="Configure Augmentations", padding="10")
        settings_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        # Number of augmented images
        num_frame = ttk.Frame(settings_frame)
        num_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(num_frame, text="Number of augmented images to generate:").pack(side=tk.LEFT, padx=(0, 10))
        self.num_images_var = tk.StringVar(value="100")
        ttk.Entry(num_frame, textvariable=self.num_images_var, width=10).pack(side=tk.LEFT)
        ttk.Label(num_frame, text="(0 = augment all images in source)").pack(side=tk.LEFT, padx=(10, 0))

        # Augmentation options
        aug_options_frame = ttk.Frame(settings_frame)
        aug_options_frame.pack(fill="x", pady=(10, 0))

        # Left column
        left_frame = ttk.Frame(aug_options_frame)
        left_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 20))

        # Color augmentations
        ttk.Label(left_frame, text="Color Augmentations", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)

        self.brightness_var = tk.BooleanVar(value=True)
        brightness_frame = ttk.Frame(left_frame)
        brightness_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(brightness_frame, text="Random Brightness",
                        variable=self.brightness_var).pack(side=tk.LEFT)
        ttk.Label(brightness_frame, text="Max delta:").pack(side=tk.LEFT, padx=(20, 5))
        self.brightness_delta_var = tk.StringVar(value="0.4")
        ttk.Entry(brightness_frame, textvariable=self.brightness_delta_var, width=5).pack(side=tk.LEFT)

        self.saturation_var = tk.BooleanVar(value=True)
        saturation_frame = ttk.Frame(left_frame)
        saturation_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(saturation_frame, text="Random Saturation",
                        variable=self.saturation_var).pack(side=tk.LEFT)
        ttk.Label(saturation_frame, text="Range:").pack(side=tk.LEFT, padx=(20, 5))
        self.saturation_lower_var = tk.StringVar(value="0.2")
        ttk.Entry(saturation_frame, textvariable=self.saturation_lower_var, width=5).pack(side=tk.LEFT)
        ttk.Label(saturation_frame, text="to").pack(side=tk.LEFT, padx=5)
        self.saturation_upper_var = tk.StringVar(value="1.1")
        ttk.Entry(saturation_frame, textvariable=self.saturation_upper_var, width=5).pack(side=tk.LEFT)

        self.hue_var = tk.BooleanVar(value=True)
        hue_frame = ttk.Frame(left_frame)
        hue_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(hue_frame, text="Random Hue",
                        variable=self.hue_var).pack(side=tk.LEFT)
        ttk.Label(hue_frame, text="Max delta:").pack(side=tk.LEFT, padx=(20, 5))
        self.hue_delta_var = tk.StringVar(value="0.2")
        ttk.Entry(hue_frame, textvariable=self.hue_delta_var, width=5).pack(side=tk.LEFT)

        # Right column
        right_frame = ttk.Frame(aug_options_frame)
        right_frame.pack(side=tk.LEFT, fill="both", expand=True)

        # Geometric augmentations
        ttk.Label(right_frame, text="Geometric Augmentations", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W)

        self.flip_var = tk.BooleanVar(value=True)
        flip_frame = ttk.Frame(right_frame)
        flip_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(flip_frame, text="Random Horizontal Flip",
                        variable=self.flip_var).pack(side=tk.LEFT)
        ttk.Label(flip_frame, text="Probability:").pack(side=tk.LEFT, padx=(20, 5))
        self.flip_prob_var = tk.StringVar(value="0.5")
        ttk.Entry(flip_frame, textvariable=self.flip_prob_var, width=5).pack(side=tk.LEFT)

        self.zoom_var = tk.BooleanVar(value=True)
        zoom_frame = ttk.Frame(right_frame)
        zoom_frame.pack(fill="x", pady=2)
        ttk.Checkbutton(zoom_frame, text="Random Zoom Out",
                        variable=self.zoom_var).pack(side=tk.LEFT)
        ttk.Label(zoom_frame, text="Probability:").pack(side=tk.LEFT, padx=(20, 5))
        self.zoom_prob_var = tk.StringVar(value="0.5")
        ttk.Entry(zoom_frame, textvariable=self.zoom_prob_var, width=5).pack(side=tk.LEFT)

        zoom_range_frame = ttk.Frame(right_frame)
        zoom_range_frame.pack(fill="x", pady=2)
        ttk.Label(zoom_range_frame, text="    Zoom factor range:").pack(side=tk.LEFT)
        self.zoom_min_var = tk.StringVar(value="0.1")
        ttk.Entry(zoom_range_frame, textvariable=self.zoom_min_var, width=5).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(zoom_range_frame, text="to").pack(side=tk.LEFT, padx=5)
        self.zoom_max_var = tk.StringVar(value="0.7")
        ttk.Entry(zoom_range_frame, textvariable=self.zoom_max_var, width=5).pack(side=tk.LEFT)

        # Preview button
        preview_frame = ttk.Frame(settings_frame)
        preview_frame.pack(fill="x", pady=(20, 0))
        ttk.Button(preview_frame, text="Preview Augmentation",
                   command=self.preview_augmentation).pack()

        # Section 3: Progress
        ttk.Label(main_frame, text="Progress",
                  font=('TkDefaultFont', 12, 'bold')).grid(row=5, column=0, columnspan=3, pady=(20, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))

        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=7, column=0, columnspan=3)

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=3, pady=(20, 0))

        self.start_btn = ttk.Button(button_frame, text="Start Augmentation",
                                    command=self.start_augmentation)
        self.start_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop",
                                   command=self.stop_augmentation,
                                   state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)

    def browse_source(self):
        directory = filedialog.askdirectory(
            title="Select Source Directory",
            initialdir=self.app_root
        )
        if directory:
            self.source_dir_var.set(directory)

    def browse_dest(self):
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.app_root
        )
        if directory:
            self.dest_dir_var.set(directory)

    def preview_augmentation(self):
        if not self.source_dir_var.get():
            messagebox.showwarning("Warning", "Please select a source directory first.")
            return

        # Find first image in directory
        image_files = [f for f in os.listdir(self.source_dir_var.get())
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
            messagebox.showinfo("Info", "No images found in source directory.")
            return

        # Create preview window
        preview_window = tk.Toplevel(self.parent)
        preview_window.title("Augmentation Preview")

        # Load and augment first image
        image_path = os.path.join(self.source_dir_var.get(), image_files[0])
        xml_path = os.path.join(self.source_dir_var.get(), Path(image_files[0]).stem + '.xml')

        try:
            # Load image
            image = self.load_image(image_path)

            # Load boxes if XML exists
            if os.path.exists(xml_path):
                boxes = self.parse_xml(xml_path)
            else:
                boxes = tf.constant([], dtype=tf.float32)

            # Apply augmentations
            aug_image, aug_boxes = self.augment_image(image, boxes)

            # Convert to displayable format
            display_image = (aug_image.numpy() * 255).astype(np.uint8)

            # Draw bounding boxes if any
            if len(aug_boxes) > 0 and os.path.exists(xml_path):
                h, w = display_image.shape[:2]
                for box in aug_boxes.numpy():
                    ymin, xmin, ymax, xmax = box
                    cv2.rectangle(display_image,
                                  (int(xmin * w), int(ymin * h)),
                                  (int(xmax * w), int(ymax * h)),
                                  (0, 255, 0), 2)

            # Convert to PIL and display
            pil_image = Image.fromarray(display_image)

            # Resize for display if needed
            max_size = 600
            if pil_image.width > max_size or pil_image.height > max_size:
                pil_image.thumbnail((max_size, max_size), Image.LANCZOS)

            photo = ImageTk.PhotoImage(pil_image)

            label = tk.Label(preview_window, image=photo)
            label.image = photo
            label.pack(padx=10, pady=10)

            ttk.Label(preview_window, text=f"Preview of: {image_files[0]}").pack()
            ttk.Button(preview_window, text="Close",
                       command=preview_window.destroy).pack(pady=10)

        except Exception as e:
            messagebox.showerror("Error", f"Could not generate preview: {str(e)}")
            preview_window.destroy()

    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        return tf.cast(image, tf.float32) / 255.0

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        boxes = []
        for obj in root.findall('object'):
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            # Normalize the coordinates
            boxes.append([
                ymin / height, xmin / width,
                ymax / height, xmax / width
            ])

        return tf.constant(boxes, dtype=tf.float32)

    def update_xml(self, xml_path, new_filename, new_boxes):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Update filename
        root.find('filename').text = new_filename

        # Update path
        if root.find('path') is not None:
            root.find('path').text = new_filename

        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        # Update bounding boxes
        for i, obj in enumerate(root.findall('object')):
            if i < len(new_boxes):
                bndbox = obj.find('bndbox')
                bndbox.find('ymin').text = str(int(new_boxes[i][0] * height))
                bndbox.find('xmin').text = str(int(new_boxes[i][1] * width))
                bndbox.find('ymax').text = str(int(new_boxes[i][2] * height))
                bndbox.find('xmax').text = str(int(new_boxes[i][3] * width))

        return tree

    def random_zoom_out(self, image, boxes):
        if not self.zoom_var.get():
            return image, boxes

        if tf.random.uniform([]) > float(self.zoom_prob_var.get()):
            return image, boxes

        original_height, original_width = tf.shape(image)[0], tf.shape(image)[1]

        # Randomly choose a zoom factor
        zoom_min = float(self.zoom_min_var.get())
        zoom_max = float(self.zoom_max_var.get())
        zoom_factor = tf.random.uniform([], zoom_min, zoom_max)

        # Calculate new dimensions
        new_height = tf.cast(tf.cast(original_height, tf.float32) * zoom_factor, tf.int32)
        new_width = tf.cast(tf.cast(original_width, tf.float32) * zoom_factor, tf.int32)

        # Resize the image
        zoomed_image = tf.image.resize(image, [new_height, new_width])

        # Create a blank canvas of the original size
        canvas = tf.zeros([original_height, original_width, 3], dtype=tf.float32)

        # Calculate offsets to center the zoomed image
        offset_height = (original_height - new_height) // 2
        offset_width = (original_width - new_width) // 2

        # Place the zoomed image onto the canvas
        canvas = tf.image.pad_to_bounding_box(zoomed_image, offset_height, offset_width,
                                              original_height, original_width)

        # Adjust bounding boxes
        if len(boxes) > 0:
            scale_y = tf.cast(new_height, tf.float32) / tf.cast(original_height, tf.float32)
            scale_x = tf.cast(new_width, tf.float32) / tf.cast(original_width, tf.float32)
            offset_y = tf.cast(offset_height, tf.float32) / tf.cast(original_height, tf.float32)
            offset_x = tf.cast(offset_width, tf.float32) / tf.cast(original_width, tf.float32)

            boxes = tf.stack([
                boxes[:, 0] * scale_y + offset_y,
                boxes[:, 1] * scale_x + offset_x,
                boxes[:, 2] * scale_y + offset_y,
                boxes[:, 3] * scale_x + offset_x
            ], axis=-1)

            # Clip the boxes to ensure they're within [0, 1]
            boxes = tf.clip_by_value(boxes, 0.0, 1.0)

        return canvas, boxes

    def random_flip_horizontal(self, image, boxes):
        if not self.flip_var.get():
            return image, boxes

        if tf.random.uniform([]) > float(self.flip_prob_var.get()):
            return image, boxes

        image = tf.image.flip_left_right(image)

        if len(boxes) > 0:
            boxes = tf.stack([
                boxes[:, 0],
                1 - boxes[:, 3],
                boxes[:, 2],
                1 - boxes[:, 1]
            ], axis=-1)

        return image, boxes

    def augment_image(self, image, boxes):
        # Color augmentations
        if self.brightness_var.get():
            max_delta = float(self.brightness_delta_var.get())
            image = tf.image.random_brightness(image, max_delta=max_delta)

        if self.saturation_var.get():
            lower = float(self.saturation_lower_var.get())
            upper = float(self.saturation_upper_var.get())
            image = tf.image.random_saturation(image, lower=lower, upper=upper)

        if self.hue_var.get():
            max_delta = float(self.hue_delta_var.get())
            image = tf.image.random_hue(image, max_delta=max_delta)

        # Geometric augmentations
        image, boxes = self.random_flip_horizontal(image, boxes)
        image, boxes = self.random_zoom_out(image, boxes)

        # Ensure the pixel values are still in [0, 1] range
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, boxes

    def start_augmentation(self):
        if not self.source_dir_var.get() or not self.dest_dir_var.get():
            messagebox.showwarning("Warning", "Please select both source and output directories.")
            return

        try:
            num_images = int(self.num_images_var.get())
            if num_images < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid number of images.")
            return

        self.stop_flag.clear()
        self.augment_thread = threading.Thread(target=self._augment_thread,
                                               args=(num_images,), daemon=True)
        self.augment_thread.start()

    def _augment_thread(self, num_images):
        try:
            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')

            source_dir = self.source_dir_var.get()
            dest_dir = self.dest_dir_var.get()

            # Create output directory
            os.makedirs(dest_dir, exist_ok=True)

            # Get image files
            image_files = [f for f in os.listdir(source_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if num_images == 0:
                num_images = len(image_files)
            else:
                np.random.shuffle(image_files)

            self.log_queue.put(f"\n=== Starting Data Augmentation ===")
            self.log_queue.put(f"Processing {min(num_images, len(image_files))} images")

            processed = 0
            skipped = 0

            for i, image_file in enumerate(image_files[:num_images]):
                if self.stop_flag.is_set():
                    break

                image_path = os.path.join(source_dir, image_file)
                xml_path = os.path.join(source_dir, Path(image_file).stem + '.xml')

                if not os.path.exists(xml_path):
                    self.log_queue.put(f"XML file not found for {image_file}, skipping...")
                    skipped += 1
                    continue

                try:
                    # Load and augment image
                    image = self.load_image(image_path)
                    boxes = self.parse_xml(xml_path)
                    augmented_image, augmented_boxes = self.augment_image(image, boxes)

                    # Save augmented image
                    new_filename = f"augmented_{i}_{image_file}"
                    new_image_path = os.path.join(dest_dir, new_filename)
                    tf.keras.preprocessing.image.save_img(new_image_path, augmented_image.numpy())

                    # Update and save XML
                    new_xml_tree = self.update_xml(xml_path, new_filename, augmented_boxes.numpy())
                    new_xml_path = os.path.join(dest_dir, f"augmented_{i}_{Path(image_file).stem}.xml")
                    new_xml_tree.write(new_xml_path)

                    processed += 1

                    # Update progress
                    progress = (i + 1) / min(num_images, len(image_files)) * 100
                    self._update_progress(progress, f"Processing: {image_file}")

                except Exception as e:
                    self.log_queue.put(f"Error processing {image_file}: {str(e)}")
                    skipped += 1

            if not self.stop_flag.is_set():
                self.log_queue.put(f"\n✅ Augmentation completed!")
                self.log_queue.put(f"Processed: {processed} images")
                self.log_queue.put(f"Skipped: {skipped} images")
            else:
                self.log_queue.put("\n⚠️ Augmentation stopped by user")

        except Exception as e:
            self.log_queue.put(f"❌ Error during augmentation: {str(e)}")
        finally:
            self.start_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self._update_progress(0, "Ready")

    def stop_augmentation(self):
        self.stop_flag.set()
        self.log_queue.put("Stopping augmentation...")

    def _update_progress(self, value, message=""):
        self.parent.after(0, lambda: self.progress_var.set(value))
        if message:
            self.parent.after(0, lambda: self.status_label.config(text=message))

