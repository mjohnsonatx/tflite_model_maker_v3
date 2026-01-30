import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tkinter as tk
import xml.etree.ElementTree as ET
from collections import defaultdict
from multiprocessing import Pool, cpu_count, Manager, Process, Queue as MPQueue
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Optional, Tuple
import glob

import cv2
import numpy as np
from PIL import Image, ImageTk


def _check_xml_issues(xml_files):
    issues = []
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name = obj.find('name')
                if name is not None and name.text != 'kettlebell':
                    issues.append(xml_path)
                    break
                if obj.find('polygon') is not None:
                    issues.append(xml_path)
                    break
        except:
            issues.append(xml_path)
    return issues


class DataCleaningTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.outliers = []
        self.current_outlier_index = 0
        self.preview_window = None
        self.canvas = None
        self.show_overlay_var = tk.BooleanVar(value=False)  # Default to no overlay
        self.setup_ui()

    def setup_ui(self):
        # Main content frame - directly in parent, no canvas/scrollbar
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Section 1: Directory Selection
        ttk.Label(main_frame, text="Data Directory Selection",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(0, 10))

        dir_frame = ttk.Frame(main_frame)
        dir_frame.pack(fill='x', pady=(0, 20))

        ttk.Label(dir_frame, text="Source Directory:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.source_dir_var = tk.StringVar()
        ttk.Entry(dir_frame, textvariable=self.source_dir_var, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E))
        ttk.Button(dir_frame, text="Browse", command=self.browse_source_dir).grid(row=0, column=2, padx=(5, 0))
        dir_frame.columnconfigure(1, weight=1)

        # Section 2: Basic Cleaning Operations
        ttk.Label(main_frame, text="Basic Cleaning Operations",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(20, 10))

        # Operation checkboxes
        self.operations_frame = ttk.LabelFrame(main_frame, text="Select Operations", padding="10")
        self.operations_frame.pack(fill='x', pady=(0, 20))

        self.clean_xml_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.operations_frame, text="Clean XML files (fix labels, remove polygons)",
                        variable=self.clean_xml_var).pack(anchor='w')

        self.verify_pairs_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.operations_frame, text="Verify image-XML pairs",
                        variable=self.verify_pairs_var).pack(anchor='w')

        self.remove_orphans_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.operations_frame, text="Remove orphaned files",
                        variable=self.remove_orphans_var).pack(anchor='w')

        # Basic cleaning buttons
        basic_button_frame = ttk.Frame(main_frame)
        basic_button_frame.pack(pady=(10, 20))

        ttk.Button(basic_button_frame, text="Analyze Data",
                   command=self.analyze_data).pack(side='left', padx=5)
        ttk.Button(basic_button_frame, text="Start Basic Cleaning",
                   command=self.start_cleaning).pack(side='left', padx=5)
        ttk.Button(basic_button_frame, text="Stop",
                   command=self.stop_cleaning).pack(side='left', padx=5)

        # Section 3: XML Filename/Path Sync
        ttk.Label(main_frame, text="XML Filename/Path Synchronization",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(20, 10))

        # XML Sync frame
        self.xml_sync_frame = ttk.LabelFrame(main_frame, text="Sync Settings", padding="10")
        self.xml_sync_frame.pack(fill='x', pady=(0, 20))

        # Options for XML sync
        self.sync_recursive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.xml_sync_frame, text="Process all subfolders recursively",
                        variable=self.sync_recursive_var,
                        command=self.on_sync_recursive_changed).pack(anchor='w', pady=(0, 5))

        # Info label
        self.sync_info_label = ttk.Label(self.xml_sync_frame,
                                         text="Single folder mode: Will process current directory only",
                                         foreground='gray')
        self.sync_info_label.pack(anchor='w', pady=(0, 10))

        # Processing options
        proc_options_frame = ttk.Frame(self.xml_sync_frame)
        proc_options_frame.pack(fill='x')

        ttk.Label(proc_options_frame, text="Max concurrent threads:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.sync_max_threads_var = tk.StringVar(value="4")
        thread_spinbox = ttk.Spinbox(proc_options_frame, from_=1, to=min(8, cpu_count()),
                                     textvariable=self.sync_max_threads_var, width=10)
        thread_spinbox.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(proc_options_frame, text=f"(1-{min(8, cpu_count())} available)").grid(row=0, column=2, sticky=tk.W,
                                                                                        padx=(10, 0))

        # Sync button
        sync_button_frame = ttk.Frame(main_frame)
        sync_button_frame.pack(pady=(0, 20))

        ttk.Button(sync_button_frame, text="Sync XML filename/path",
                   command=self.start_xml_sync).pack(side='left', padx=5)
        ttk.Button(sync_button_frame, text="Stop Sync",
                   command=self.stop_xml_sync).pack(side='left', padx=5)

        # Section 5: Outlier Detection - Side by Side Layout
        ttk.Label(main_frame, text="Outlier Detection Settings",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(20, 10))

        # Create container for side-by-side layout
        outlier_container = ttk.Frame(main_frame)
        outlier_container.pack(fill='x', pady=(0, 20))

        # Left column - Bounding Box Settings
        bbox_frame = ttk.LabelFrame(outlier_container, text="Bounding Box Outliers", padding="10")
        bbox_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Size thresholds
        ttk.Label(bbox_frame, text="Box Area Thresholds:", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w',
                                                                                                   pady=(0, 5))

        size_frame = ttk.Frame(bbox_frame)
        size_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(size_frame, text="Min area (% of image):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.min_area_var = tk.StringVar(value="0.5")
        ttk.Entry(size_frame, textvariable=self.min_area_var, width=8).grid(row=0, column=1, sticky=tk.W)
        ttk.Label(size_frame, text="%").grid(row=0, column=2, sticky=tk.W)

        ttk.Label(size_frame, text="Max area (% of image):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.max_area_var = tk.StringVar(value="95")
        ttk.Entry(size_frame, textvariable=self.max_area_var, width=8).grid(row=1, column=1, sticky=tk.W)
        ttk.Label(size_frame, text="%").grid(row=1, column=2, sticky=tk.W)

        # Aspect ratio thresholds
        ttk.Label(bbox_frame, text="Aspect Ratio Thresholds:", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w',
                                                                                                       pady=(10, 5))

        aspect_frame = ttk.Frame(bbox_frame)
        aspect_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(aspect_frame, text="Min ratio (w/h):").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.min_aspect_var = tk.StringVar(value="0.1")
        ttk.Entry(aspect_frame, textvariable=self.min_aspect_var, width=8).grid(row=0, column=1, sticky=tk.W)

        ttk.Label(aspect_frame, text="Max ratio (w/h):").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.max_aspect_var = tk.StringVar(value="10")
        ttk.Entry(aspect_frame, textvariable=self.max_aspect_var, width=8).grid(row=1, column=1, sticky=tk.W)

        # Additional options
        ttk.Label(bbox_frame, text="Additional Checks:", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w',
                                                                                                 pady=(10, 5))

        self.use_statistical_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bbox_frame, text="Statistical outliers (IQR)",
                        variable=self.use_statistical_var).pack(anchor='w')

        self.check_edge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(bbox_frame, text="Boxes touching edges",
                        variable=self.check_edge_var).pack(anchor='w')

        # Right column - Image Quality Settings
        quality_frame = ttk.LabelFrame(outlier_container, text="Image Quality Outliers", padding="10")
        quality_frame.pack(side='left', fill='both', expand=True)

        # Brightness settings
        ttk.Label(quality_frame, text="Brightness:", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(0, 5))

        brightness_frame = ttk.Frame(quality_frame)
        brightness_frame.pack(fill='x', pady=(0, 10))

        self.check_brightness_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(brightness_frame, text="Enable",
                        variable=self.check_brightness_var).grid(row=0, column=0, sticky=tk.W, columnspan=2)

        ttk.Label(brightness_frame, text="Dark threshold:").grid(row=1, column=0, sticky=tk.W, padx=(20, 5))
        self.dark_threshold_var = tk.StringVar(value="30")
        ttk.Entry(brightness_frame, textvariable=self.dark_threshold_var, width=8).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(brightness_frame, text="Bright threshold:").grid(row=2, column=0, sticky=tk.W, padx=(20, 5))
        self.bright_threshold_var = tk.StringVar(value="220")
        ttk.Entry(brightness_frame, textvariable=self.bright_threshold_var, width=8).grid(row=2, column=1, sticky=tk.W)

        # Saturation settings
        ttk.Label(quality_frame, text="Saturation:", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(10, 5))

        saturation_frame = ttk.Frame(quality_frame)
        saturation_frame.pack(fill='x', pady=(0, 10))

        self.check_saturation_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(saturation_frame, text="Enable",
                        variable=self.check_saturation_var).grid(row=0, column=0, sticky=tk.W, columnspan=2)

        ttk.Label(saturation_frame, text="Low threshold:").grid(row=1, column=0, sticky=tk.W, padx=(20, 5))
        self.low_saturation_var = tk.StringVar(value="20")
        ttk.Entry(saturation_frame, textvariable=self.low_saturation_var, width=8).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(saturation_frame, text="High threshold:").grid(row=2, column=0, sticky=tk.W, padx=(20, 5))
        self.high_saturation_var = tk.StringVar(value="200")
        ttk.Entry(saturation_frame, textvariable=self.high_saturation_var, width=8).grid(row=2, column=1, sticky=tk.W)

        # Other quality checks
        ttk.Label(quality_frame, text="Other Checks:", font=('TkDefaultFont', 9, 'bold')).pack(anchor='w', pady=(10, 5))

        other_frame = ttk.Frame(quality_frame)
        other_frame.pack(fill='x')

        self.check_contrast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(other_frame, text="Low contrast",
                        variable=self.check_contrast_var).grid(row=0, column=0, sticky=tk.W)

        self.low_contrast_var = tk.StringVar(value="20")
        ttk.Entry(other_frame, textvariable=self.low_contrast_var, width=8).grid(row=0, column=1, sticky=tk.W,
                                                                                 padx=(5, 0))

        self.check_blur_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(other_frame, text="Blur detection",
                        variable=self.check_blur_var).grid(row=1, column=0, sticky=tk.W)

        self.blur_threshold_var = tk.StringVar(value="100")
        ttk.Entry(other_frame, textvariable=self.blur_threshold_var, width=8).grid(row=1, column=1, sticky=tk.W,
                                                                                   padx=(5, 0))

        # Common settings
        common_frame = ttk.LabelFrame(main_frame, text="Common Settings", padding="10")
        common_frame.pack(fill='x', pady=(0, 20))

        # Outlier folder
        folder_frame = ttk.Frame(common_frame)
        folder_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(folder_frame, text="Outlier folder name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.outlier_folder_var = tk.StringVar(value="outliers_to_review")
        ttk.Entry(folder_frame, textvariable=self.outlier_folder_var, width=30).grid(row=0, column=1, sticky=tk.W)

        # Processing settings
        proc_frame = ttk.Frame(common_frame)
        proc_frame.pack(fill='x')

        ttk.Label(proc_frame, text="Processing threads:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.num_threads_var = tk.StringVar(value=str(min(4, cpu_count())))
        thread_spinbox = ttk.Spinbox(proc_frame, from_=1, to=cpu_count(),
                                     textvariable=self.num_threads_var, width=10)
        thread_spinbox.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(proc_frame, text=f"(1-{cpu_count()} available)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))

        # Detection type selection
        self.detection_type_var = tk.StringVar(value="both")
        detection_frame = ttk.Frame(common_frame)
        detection_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(detection_frame, text="Detection type:").pack(side='left', padx=(0, 10))
        ttk.Radiobutton(detection_frame, text="Bounding boxes only",
                        variable=self.detection_type_var, value="bbox").pack(side='left', padx=5)
        ttk.Radiobutton(detection_frame, text="Image quality only",
                        variable=self.detection_type_var, value="quality").pack(side='left', padx=5)
        ttk.Radiobutton(detection_frame, text="Both",
                        variable=self.detection_type_var, value="both").pack(side='left', padx=5)

        # Outlier detection buttons
        outlier_button_frame = ttk.Frame(main_frame)
        outlier_button_frame.pack(pady=(0, 20))

        ttk.Button(outlier_button_frame, text="Detect Outliers",
                   command=self.detect_outliers).pack(side='left', padx=5)
        ttk.Button(outlier_button_frame, text="Preview Outliers",
                   command=self.preview_outliers).pack(side='left', padx=5)
        ttk.Button(outlier_button_frame, text="Move Outliers to Folder",
                   command=self.move_outliers).pack(side='left', padx=5)

        # Section 6: Results
        ttk.Label(main_frame, text="Results",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(20, 10))

        # Create notebook for results
        self.results_notebook = ttk.Notebook(main_frame)
        self.results_notebook.pack(fill='both', expand=True, pady=(0, 20))

        # Summary tab
        self.summary_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.summary_frame, text="Summary")

        self.summary_text = tk.Text(self.summary_frame, height=15, wrap=tk.WORD)
        self.summary_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Outliers list tab
        self.list_frame = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.list_frame, text="Outlier List")

        # Create treeview for outliers
        columns = ('File', 'Type', 'Category', 'Value', 'Issue')
        self.outlier_tree = ttk.Treeview(self.list_frame, columns=columns, show='tree headings', height=10)

        for col in columns:
            self.outlier_tree.heading(col, text=col)
            self.outlier_tree.column(col, width=150)

        # Scrollbars for treeview
        tree_scroll_y = ttk.Scrollbar(self.list_frame, orient='vertical', command=self.outlier_tree.yview)
        tree_scroll_x = ttk.Scrollbar(self.list_frame, orient='horizontal', command=self.outlier_tree.xview)
        self.outlier_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        self.outlier_tree.grid(row=0, column=0, sticky='nsew')
        tree_scroll_y.grid(row=0, column=1, sticky='ns')
        tree_scroll_x.grid(row=1, column=0, sticky='ew')

        self.list_frame.rowconfigure(0, weight=1)
        self.list_frame.columnconfigure(0, weight=1)

        # Bind double-click to preview
        self.outlier_tree.bind('<Double-Button-1>', self.on_tree_double_click)

        # Section 7: Progress
        ttk.Label(main_frame, text="Progress",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(20, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=(0, 10))

        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.pack()

        # Initialize
        self.cleaning_thread = None
        self.stop_flag = threading.Event()
        self.sync_stop_flag = threading.Event()
        self.manager = None
        self.progress_dict = None
        self.sync_thread = None

    def browse_source_dir(self):
        directory = filedialog.askdirectory(
            title="Select Source Directory",
            initialdir=self.app_root
        )
        if directory:
            self.source_dir_var.set(directory)

    def analyze_data(self):
        if not self.source_dir_var.get():
            messagebox.showwarning("Warning", "Please select a source directory first.")
            return

        self.log_queue.put("Starting data analysis...")
        threading.Thread(target=self._analyze_data_thread, daemon=True).start()

    def _analyze_data_thread(self):
        try:
            source_dir = self.source_dir_var.get()

            # Count files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            xml_files = []
            image_files = []

            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    ext = Path(file).suffix.lower()
                    if ext in image_extensions:
                        image_files.append(os.path.join(root, file))
                    elif ext == '.xml':
                        xml_files.append(os.path.join(root, file))

            self.log_queue.put(f"\nData Analysis Results:")
            self.log_queue.put(f"Total images: {len(image_files)}")
            self.log_queue.put(f"Total XML files: {len(xml_files)}")

            # Check for orphaned files
            image_stems = {}
            xml_stems = {}

            for img_path in image_files:
                stem = Path(img_path).stem
                image_stems[stem] = img_path

            for xml_path in xml_files:
                stem = Path(xml_path).stem
                xml_stems[stem] = xml_path

            orphaned_images = set(image_stems.keys()) - set(xml_stems.keys())
            orphaned_xmls = set(xml_stems.keys()) - set(image_stems.keys())

            self.log_queue.put(f"\nOrphaned images (no XML): {len(orphaned_images)}")
            if orphaned_images and len(orphaned_images) < 10:
                for img in list(orphaned_images)[:5]:
                    self.log_queue.put(f"  - {img}")

            self.log_queue.put(f"Orphaned XMLs (no image): {len(orphaned_xmls)}")
            if orphaned_xmls and len(orphaned_xmls) < 10:
                for xml in list(orphaned_xmls)[:5]:
                    self.log_queue.put(f"  - {xml}")

            # Check XML content
            if xml_files:
                self.log_queue.put("\nChecking XML files...")
                issues = _check_xml_issues(xml_files[:min(50, len(xml_files))])
                if issues:
                    self.log_queue.put(
                        f"Found {len(issues)} XML files with issues (checked first {min(50, len(xml_files))} files)")
                else:
                    self.log_queue.put("No XML issues found in sample")

        except Exception as e:
            self.log_queue.put(f"Error during analysis: {str(e)}")

    def start_cleaning(self):
        if not self.source_dir_var.get():
            messagebox.showwarning("Warning", "Please select a source directory first.")
            return

        self.stop_flag.clear()
        self.cleaning_thread = threading.Thread(target=self._cleaning_thread, daemon=True)
        self.cleaning_thread.start()

    def stop_cleaning(self):
        self.stop_flag.set()
        self.log_queue.put("Stopping cleaning process...")

    def _cleaning_thread(self):
        try:
            source_dir = self.source_dir_var.get()
            total_operations = sum([
                self.clean_xml_var.get(),
                self.verify_pairs_var.get(),
                self.remove_orphans_var.get()
            ])

            if total_operations == 0:
                self.log_queue.put("No operations selected!")
                return

            current_op = 0

            # Clean XML files
            if self.clean_xml_var.get() and not self.stop_flag.is_set():
                self.log_queue.put("\n=== Cleaning XML files ===")
                self._update_status("Cleaning XML files...")
                self._clean_xml_files(source_dir)
                current_op += 1
                self._update_progress((current_op / total_operations) * 100)

            # Verify image-XML pairs
            if self.verify_pairs_var.get() and not self.stop_flag.is_set():
                self.log_queue.put("\n=== Verifying image-XML pairs ===")
                self._update_status("Verifying pairs...")
                self._verify_pairs(source_dir)
                current_op += 1
                self._update_progress((current_op / total_operations) * 100)

            # Remove orphaned files
            if self.remove_orphans_var.get() and not self.stop_flag.is_set():
                self.log_queue.put("\n=== Removing orphaned files ===")
                self._update_status("Removing orphans...")
                self._remove_orphans(source_dir)
                current_op += 1
                self._update_progress((current_op / total_operations) * 100)

            if not self.stop_flag.is_set():
                self._update_status("Cleaning completed!")
                self.log_queue.put("\n✅ Data cleaning completed successfully!")

        except Exception as e:
            self.log_queue.put(f"\n❌ Error during cleaning: {str(e)}")
            self._update_status(f"Error: {str(e)}")

    def _clean_xml_files(self, directory):
        xml_files = []
        for root, dirs, files in os.walk(directory):
            xml_files.extend([os.path.join(root, f) for f in files if f.endswith('.xml')])

        total = len(xml_files)
        modified_count = 0

        for i, xml_path in enumerate(xml_files):
            if self.stop_flag.is_set():
                break

            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                modified = False

                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name is not None and name.text != 'kettlebell':
                        name.text = 'kettlebell'
                        modified = True

                    polygon = obj.find('polygon')
                    if polygon is not None:
                        obj.remove(polygon)
                        modified = True

                if modified:
                    tree.write(xml_path)
                    modified_count += 1
                    self.log_queue.put(f"Updated: {os.path.basename(xml_path)}")

                self._update_progress((i + 1) / total * 100)

            except Exception as e:
                self.log_queue.put(f"Error processing {xml_path}: {str(e)}")

        self.log_queue.put(f"Modified {modified_count} XML files")

    def _verify_pairs(self, directory):
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = {}
        xml_files = {}

        # Collect all files
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                stem = Path(file).stem
                ext = Path(file).suffix.lower()

                if ext in image_extensions:
                    image_files[stem] = filepath
                elif ext == '.xml':
                    xml_files[stem] = filepath

        # Find pairs and orphans
        paired = set(image_files.keys()) & set(xml_files.keys())
        orphaned_images = set(image_files.keys()) - set(xml_files.keys())
        orphaned_xmls = set(xml_files.keys()) - set(image_files.keys())

        self.log_queue.put(f"Found {len(paired)} valid image-XML pairs")
        self.log_queue.put(f"Found {len(orphaned_images)} images without XML")
        self.log_queue.put(f"Found {len(orphaned_xmls)} XML files without images")

    def _remove_orphans(self, directory):
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = {}
        xml_files = {}

        # Collect all files
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                stem = Path(file).stem
                ext = Path(file).suffix.lower()

                if ext in image_extensions:
                    image_files[stem] = filepath
                elif ext == '.xml':
                    xml_files[stem] = filepath

        # Find orphans
        orphaned_images = set(image_files.keys()) - set(xml_files.keys())
        orphaned_xmls = set(xml_files.keys()) - set(image_files.keys())

        removed_count = 0

        # Remove orphaned images
        for stem in orphaned_images:
            if self.stop_flag.is_set():
                break
            try:
                os.remove(image_files[stem])
                self.log_queue.put(f"Removed orphaned image: {os.path.basename(image_files[stem])}")
                removed_count += 1
            except Exception as e:
                self.log_queue.put(f"Error removing {image_files[stem]}: {str(e)}")

        # Remove orphaned XMLs
        for stem in orphaned_xmls:
            if self.stop_flag.is_set():
                break
            try:
                os.remove(xml_files[stem])
                self.log_queue.put(f"Removed orphaned XML: {os.path.basename(xml_files[stem])}")
                removed_count += 1
            except Exception as e:
                self.log_queue.put(f"Error removing {xml_files[stem]}: {str(e)}")

        self.log_queue.put(f"Removed {removed_count} orphaned files")

    # Multiprocessing worker functions with progress tracking
    @staticmethod
    def process_bbox_batch(args):
        """Process a batch of images for bbox outliers with progress tracking"""
        file_batch, source_dir, params, progress_dict, batch_id = args
        results = []

        for idx, img_file in enumerate(file_batch):
            img_path = os.path.join(source_dir, img_file)
            xml_path = os.path.join(source_dir, Path(img_file).stem + '.xml')

            if not os.path.exists(xml_path):
                # Update progress
                if progress_dict is not None:
                    progress_dict[batch_id] = idx + 1
                continue

            try:
                # Get image dimensions
                img = cv2.imread(img_path)
                if img is None:
                    if progress_dict is not None:
                        progress_dict[batch_id] = idx + 1
                    continue

                img_height, img_width = img.shape[:2]
                img_area = img_width * img_height

                # Parse XML
                tree = ET.parse(xml_path)
                root = tree.getroot()

                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)

                    box_width = xmax - xmin
                    box_height = ymax - ymin
                    box_area = box_width * box_height

                    area_percent = (box_area / img_area) * 100
                    aspect_ratio = box_width / box_height if box_height > 0 else 0

                    results.append({
                        'file': img_file,
                        'xml_path': xml_path,
                        'area_percent': area_percent,
                        'aspect_ratio': aspect_ratio,
                        'xmin': xmin, 'ymin': ymin,
                        'xmax': xmax, 'ymax': ymax,
                        'img_width': img_width,
                        'img_height': img_height,
                        'at_edge': (xmin <= 1 or ymin <= 1 or
                                    xmax >= img_width - 1 or ymax >= img_height - 1)
                    })

            except Exception as e:
                pass

            # Update progress
            if progress_dict is not None:
                progress_dict[batch_id] = idx + 1

        return results

    @staticmethod
    def process_quality_batch(args):
        """Process a batch of images for quality outliers with progress tracking"""
        file_batch, source_dir, params, progress_dict, batch_id = args
        results = []

        for idx, img_file in enumerate(file_batch):
            img_path = os.path.join(source_dir, img_file)
            xml_path = os.path.join(source_dir, Path(img_file).stem + '.xml')

            img = cv2.imread(img_path)
            if img is None:
                if progress_dict is not None:
                    progress_dict[batch_id] = idx + 1
                continue

            # Convert to different color spaces
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            # Calculate saturation properly - exclude black pixels (V=0)
            v_channel = hsv[:, :, 2]
            s_channel = hsv[:, :, 1]

            # Only consider pixels with some brightness
            valid_pixels = v_channel > 10
            if np.any(valid_pixels):
                mean_saturation = np.mean(s_channel[valid_pixels])
            else:
                mean_saturation = np.mean(s_channel)

            metrics = {
                'file': img_file,
                'xml_path': xml_path,
                'brightness': np.mean(v_channel),
                'saturation': mean_saturation,
                'contrast': np.std(gray),
                'sharpness': cv2.Laplacian(gray, cv2.CV_64F).var(),
                'hue_mean': np.mean(hsv[:, :, 0]),
                'hue_std': np.std(hsv[:, :, 0])
            }

            results.append(metrics)

            # Update progress
            if progress_dict is not None:
                progress_dict[batch_id] = idx + 1

        return results

    # Outlier Detection Methods
    def detect_outliers(self):
        if not self.source_dir_var.get():
            messagebox.showwarning("Warning", "Please select a source directory first.")
            return

        threading.Thread(target=self._detect_outliers_thread, daemon=True).start()

    def _detect_outliers_thread(self):
        try:
            self._update_status("Detecting outliers...")
            self._update_progress(0)
            self.outliers = []

            detection_type = self.detection_type_var.get()
            num_threads = int(self.num_threads_var.get())

            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png'}
            image_files = []

            for file in os.listdir(self.source_dir_var.get()):
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(file)

            total_files = len(image_files)
            self.log_queue.put(f"\n=== Outlier Detection ===")
            self.log_queue.put(f"Analyzing {total_files} images using {num_threads} threads...")

            # Split files into batches for multiprocessing
            batch_size = max(1, total_files // num_threads)
            file_batches = [image_files[i:i + batch_size] for i in range(0, total_files, batch_size)]

            bbox_outliers = []
            quality_outliers = []

            # Process bounding box outliers
            if detection_type in ['bbox', 'both']:
                self.log_queue.put("Analyzing bounding boxes...")
                bbox_outliers = self._detect_bbox_outliers_mp(file_batches, num_threads, total_files)
                self._update_progress(50 if detection_type == 'both' else 100)

            # Process quality outliers
            if detection_type in ['quality', 'both']:
                self.log_queue.put("Analyzing image quality...")
                quality_outliers = self._detect_quality_outliers_mp(file_batches, num_threads, total_files)
                self._update_progress(100)

            # Combine outliers
            self.outliers = bbox_outliers + quality_outliers

            # Remove duplicates
            unique_files = {}
            for outlier in self.outliers:
                filename = outlier['file']
                if filename not in unique_files:
                    unique_files[filename] = outlier
                else:
                    # Merge issues
                    unique_files[filename]['issues'] += "; " + outlier['issues']
                    unique_files[filename]['outlier_type'].extend(outlier['outlier_type'])

            self.outliers = list(unique_files.values())

            # Update UI
            self.update_outlier_results()

            self._update_status(f"Detection complete. Found {len(self.outliers)} outliers.")
            self.log_queue.put(f"\n✅ Found {len(self.outliers)} outliers")

        except Exception as e:
            self.log_queue.put(f"❌ Error during outlier detection: {str(e)}")
            self._update_status("Error during detection")

    def _detect_bbox_outliers_mp(self, file_batches, num_threads, total_files):
        """Detect bounding box outliers using multiprocessing with progress tracking"""
        # Create shared progress dict
        manager = Manager()
        progress_dict = manager.dict()

        # Prepare arguments for workers
        args = []
        for batch_id, batch in enumerate(file_batches):
            progress_dict[batch_id] = 0
            args.append((batch, self.source_dir_var.get(), {}, progress_dict, batch_id))

        # Process batches in parallel with progress monitoring
        with Pool(processes=num_threads) as pool:
            result = pool.map_async(DataCleaningTab.process_bbox_batch, args)

            # Monitor progress
            while not result.ready():
                processed = sum(progress_dict.values())
                progress_pct = (processed / total_files) * 100 if total_files > 0 else 0
                self._update_status(f"Analyzing bounding boxes... {processed}/{total_files}")
                self._update_progress(progress_pct * 0.5)  # First half of progress
                result.wait(0.1)

            results = result.get()

        # Flatten results
        all_box_data = []
        for batch_result in results:
            all_box_data.extend(batch_result)

        # Extract metrics for statistical analysis
        all_box_areas = [d['area_percent'] for d in all_box_data]
        all_aspect_ratios = [d['aspect_ratio'] for d in all_box_data]

        # Calculate statistical thresholds
        if self.use_statistical_var.get() and all_box_areas:
            area_q1 = np.percentile(all_box_areas, 25)
            area_q3 = np.percentile(all_box_areas, 75)
            area_iqr = area_q3 - area_q1
            iqr_mult = 1.5

            area_lower_bound = area_q1 - iqr_mult * area_iqr
            area_upper_bound = area_q3 + iqr_mult * area_iqr

            aspect_q1 = np.percentile(all_aspect_ratios, 25)
            aspect_q3 = np.percentile(all_aspect_ratios, 75)
            aspect_iqr = aspect_q3 - aspect_q1

            aspect_lower_bound = aspect_q1 - iqr_mult * aspect_iqr
            aspect_upper_bound = aspect_q3 + iqr_mult * aspect_iqr
        else:
            area_lower_bound = area_upper_bound = None
            aspect_lower_bound = aspect_upper_bound = None

        # Identify outliers
        outliers = []
        min_area = float(self.min_area_var.get())
        max_area = float(self.max_area_var.get())
        min_aspect = float(self.min_aspect_var.get())
        max_aspect = float(self.max_aspect_var.get())

        for data in all_box_data:
            issues = []
            outlier_type = []

            # Check absolute thresholds
            if data['area_percent'] < min_area:
                issues.append(f"Too small ({data['area_percent']:.1f}% < {min_area}%)")
                outlier_type.append("tiny")
            elif data['area_percent'] > max_area:
                issues.append(f"Too large ({data['area_percent']:.1f}% > {max_area}%)")
                outlier_type.append("huge")

            if data['aspect_ratio'] < min_aspect:
                issues.append(f"Too tall (aspect {data['aspect_ratio']:.2f} < {min_aspect})")
                outlier_type.append("tall")
            elif data['aspect_ratio'] > max_aspect:
                issues.append(f"Too wide (aspect {data['aspect_ratio']:.2f} > {max_aspect})")
                outlier_type.append("wide")

            # Check statistical outliers
            if self.use_statistical_var.get() and area_lower_bound is not None:
                if data['area_percent'] < area_lower_bound:
                    issues.append("Statistical outlier (small)")
                    outlier_type.append("stat_small")
                elif data['area_percent'] > area_upper_bound:
                    issues.append("Statistical outlier (large)")
                    outlier_type.append("stat_large")

                if data['aspect_ratio'] < aspect_lower_bound:
                    issues.append("Statistical outlier (aspect low)")
                    outlier_type.append("stat_aspect_low")
                elif data['aspect_ratio'] > aspect_upper_bound:
                    issues.append("Statistical outlier (aspect high)")
                    outlier_type.append("stat_aspect_high")

            # Check edge proximity
            if self.check_edge_var.get() and data['at_edge']:
                issues.append("Touching image edge")
                outlier_type.append("edge")

            if issues:
                data['issues'] = "; ".join(issues)
                data['outlier_type'] = outlier_type
                data['category'] = 'bbox'
                data['value'] = f"Area: {data['area_percent']:.1f}%, Aspect: {data['aspect_ratio']:.2f}"
                outliers.append(data)

        return outliers

    def _detect_quality_outliers_mp(self, file_batches, num_threads, total_files):
        """Detect image quality outliers using multiprocessing with progress tracking"""
        # Create shared progress dict
        manager = Manager()
        progress_dict = manager.dict()

        # Prepare arguments for workers
        args = []
        for batch_id, batch in enumerate(file_batches):
            progress_dict[batch_id] = 0
            args.append((batch, self.source_dir_var.get(), {}, progress_dict, batch_id))

        # Process batches in parallel with progress monitoring
        with Pool(processes=num_threads) as pool:
            result = pool.map_async(DataCleaningTab.process_quality_batch, args)

            # Monitor progress
            while not result.ready():
                processed = sum(progress_dict.values())
                progress_pct = (processed / total_files) * 100 if total_files > 0 else 0
                self._update_status(f"Analyzing quality... {processed}/{total_files}")
                self._update_progress(50 + (progress_pct * 0.5))  # Second half of progress
                result.wait(0.1)

            results = result.get()

        # Flatten results
        all_metrics = []
        for batch_result in results:
            all_metrics.extend(batch_result)

        # Identify outliers
        outliers = []
        dark_threshold = float(self.dark_threshold_var.get())
        bright_threshold = float(self.bright_threshold_var.get())
        low_saturation = float(self.low_saturation_var.get())
        high_saturation = float(self.high_saturation_var.get())
        low_contrast = float(self.low_contrast_var.get())
        blur_threshold = float(self.blur_threshold_var.get())

        for metrics in all_metrics:
            issues = []
            outlier_type = []
            values = []

            # Check brightness
            if self.check_brightness_var.get():
                if metrics['brightness'] < dark_threshold:
                    issues.append(f"Too dark (brightness: {metrics['brightness']:.1f})")
                    outlier_type.append("dark")
                    values.append(f"Brightness: {metrics['brightness']:.1f}")
                elif metrics['brightness'] > bright_threshold:
                    issues.append(f"Too bright (brightness: {metrics['brightness']:.1f})")
                    outlier_type.append("bright")
                    values.append(f"Brightness: {metrics['brightness']:.1f}")

            # Check saturation
            if self.check_saturation_var.get():
                if metrics['saturation'] < low_saturation:
                    issues.append(f"Low saturation ({metrics['saturation']:.1f})")
                    outlier_type.append("desaturated")
                    values.append(f"Saturation: {metrics['saturation']:.1f}")
                elif metrics['saturation'] > high_saturation:
                    issues.append(f"High saturation ({metrics['saturation']:.1f})")
                    outlier_type.append("oversaturated")
                    values.append(f"Saturation: {metrics['saturation']:.1f}")

            # Check contrast
            if self.check_contrast_var.get():
                if metrics['contrast'] < low_contrast:
                    issues.append(f"Low contrast ({metrics['contrast']:.1f})")
                    outlier_type.append("low_contrast")
                    values.append(f"Contrast: {metrics['contrast']:.1f}")

            # Check blur
            if self.check_blur_var.get():
                if metrics['sharpness'] < blur_threshold:
                    issues.append(f"Blurry image (sharpness: {metrics['sharpness']:.1f})")
                    outlier_type.append("blurry")
                    values.append(f"Sharpness: {metrics['sharpness']:.1f}")

            if issues:
                outlier_data = {
                    'file': metrics['file'],
                    'xml_path': metrics['xml_path'],
                    'issues': "; ".join(issues),
                    'outlier_type': outlier_type,
                    'category': 'quality',
                    'value': "; ".join(values),
                    'metrics': metrics
                }
                outliers.append(outlier_data)

        return outliers

    def update_outlier_results(self):
        # Update summary
        summary = f"Outlier Detection Results\n"
        summary += f"{'=' * 50}\n\n"
        summary += f"Total outliers found: {len(self.outliers)}\n\n"

        # Count by category
        bbox_outliers = [o for o in self.outliers if o.get('category') == 'bbox']
        quality_outliers = [o for o in self.outliers if o.get('category') == 'quality']

        summary += f"Bounding box outliers: {len(bbox_outliers)}\n"
        summary += f"Image quality outliers: {len(quality_outliers)}\n\n"

        # Outlier breakdown
        outlier_types = defaultdict(int)
        for outlier in self.outliers:
            for otype in outlier['outlier_type']:
                outlier_types[otype] += 1

        if outlier_types:
            summary += f"Outlier Breakdown:\n"
            for otype, count in sorted(outlier_types.items(), key=lambda x: x[1], reverse=True):
                summary += f"  {otype}: {count}\n"

        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary)

        # Update outlier list
        self.outlier_tree.delete(*self.outlier_tree.get_children())
        for outlier in self.outliers:
            self.outlier_tree.insert('', 'end', values=(
                outlier['file'],
                ', '.join(outlier['outlier_type']),
                outlier.get('category', 'unknown'),
                outlier.get('value', 'N/A'),
                outlier['issues']
            ))

    def preview_outliers(self):
        if not self.outliers:
            messagebox.showinfo("Info", "No outliers to preview. Run outlier detection first.")
            return

        self.current_outlier_index = 0
        self.initial_outlier_count = len(self.outliers)
        self.show_outlier_preview()

    def show_outlier_preview(self):
        if self.preview_window:
            self.preview_window.destroy()

        outlier = self.outliers[self.current_outlier_index]
        img_path = os.path.join(self.source_dir_var.get(), outlier['file'])

        # Update window title with counter
        self.preview_window = tk.Toplevel(self.parent)
        self.preview_window.title(
            f"Outlier Preview [{self.current_outlier_index + 1}/{len(self.outliers)}] - {outlier['file']}")

        # Create main paned window for split view
        main_paned = ttk.PanedWindow(self.preview_window, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Add prominent counter at the top
        counter_frame = ttk.Frame(self.preview_window)
        counter_frame.pack(fill=tk.X, side=tk.TOP, before=main_paned, padx=10, pady=(5, 0))

        # Large counter display
        counter_text = f"{self.current_outlier_index + 1} / {len(self.outliers)}"
        counter_label = ttk.Label(counter_frame, text=counter_text,
                                  font=('TkDefaultFont', 16, 'bold'))
        counter_label.pack(side=tk.LEFT)

        # Progress percentage
        progress_pct = ((self.current_outlier_index + 1) / len(self.outliers)) * 100
        progress_text = f"({progress_pct:.1f}% complete)"
        ttk.Label(counter_frame, text=progress_text,
                  font=('TkDefaultFont', 10)).pack(side=tk.LEFT, padx=(10, 0))

        # Visual progress bar
        progress_bar = ttk.Progressbar(counter_frame, mode='determinate',
                                       length=200, value=progress_pct)
        progress_bar.pack(side=tk.LEFT, padx=(20, 0))

        # Left panel - Image
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=3)

        # Right panel - Details
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        # Load and display image
        img = cv2.imread(img_path)
        if img is None:
            messagebox.showerror("Error", f"Could not load image: {img_path}")
            return

        # Create a copy for drawing
        display_img = img.copy()

        # Only draw annotations if overlay is enabled
        bbox_data = []
        xml_path = outlier.get('xml_path')
        if xml_path and os.path.exists(xml_path):
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()

                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    if bbox is not None:
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)

                        label = obj.find('name')
                        label_text = label.text if label is not None else "unknown"

                        # Check if this is the outlier box
                        is_outlier_box = False
                        if outlier.get('category') == 'bbox':
                            if (xmin == outlier.get('xmin') and ymin == outlier.get('ymin') and
                                    xmax == outlier.get('xmax') and ymax == outlier.get('ymax')):
                                is_outlier_box = True

                        bbox_data.append({
                            'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                            'label': label_text, 'is_outlier': is_outlier_box
                        })

                        if self.show_overlay_var.get():
                            # Draw box with different color for outlier
                            color = (0, 0, 255) if is_outlier_box else (0, 255, 0)
                            thickness = 3 if is_outlier_box else 2

                            cv2.rectangle(display_img, (xmin, ymin), (xmax, ymax), color, thickness)

                            # Add label
                            if is_outlier_box:
                                label_text += " (OUTLIER)"

                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 0.6
                            font_thickness = 2
                            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]

                            text_x = xmin
                            text_y = ymin - 5 if ymin > 20 else ymax + 20

                            cv2.rectangle(display_img,
                                          (text_x, text_y - text_size[1] - 4),
                                          (text_x + text_size[0] + 4, text_y + 4),
                                          color, -1)

                            cv2.putText(display_img, label_text,
                                        (text_x + 2, text_y),
                                        font, font_scale,
                                        (255, 255, 255), font_thickness)

            except Exception as e:
                self.log_queue.put(f"Error parsing XML for preview: {str(e)}")

        # Convert to RGB and resize for display
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        max_size = 800
        if w > max_size or h > max_size:
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))

        # Convert to PIL and display
        pil_image = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        img_label = tk.Label(left_frame, image=photo)
        img_label.image = photo
        img_label.pack(expand=True)

        # Store for toggle function
        self.current_img_label = img_label
        self.current_img_original = img
        self.current_bbox_data = bbox_data

        # Right panel contents
        # Title and toggle
        title_frame = ttk.Frame(right_frame)
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        ttk.Label(title_frame, text="Outlier Details",
                  font=('TkDefaultFont', 12, 'bold')).pack(side=tk.LEFT)

        ttk.Checkbutton(title_frame, text="Overlay",
                        variable=self.show_overlay_var,
                        command=lambda: self.toggle_preview_overlay(outlier)).pack(side=tk.RIGHT)

        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=10, pady=5)

        # Scrollable frame for details
        canvas = tk.Canvas(right_frame, highlightthickness=0, width=300)
        scrollbar = ttk.Scrollbar(right_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0))
        scrollbar.pack(side="right", fill="y", padx=(0, 10))

        # File info - Remove the duplicate counter from here since we have it at the top
        info_frame = ttk.LabelFrame(scrollable_frame, text="File Information", padding=10)
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(info_frame, text=f"File: {outlier['file']}", wraplength=250).pack(anchor=tk.W)
        ttk.Label(info_frame, text=f"Category: {outlier.get('category', 'unknown')}").pack(anchor=tk.W)

        if xml_path and os.path.exists(xml_path):
            ttk.Label(info_frame, text=f"XML: {os.path.basename(xml_path)}", wraplength=250).pack(anchor=tk.W)
        else:
            ttk.Label(info_frame, text="XML: Not found", foreground='red').pack(anchor=tk.W)

        # Issues
        issues_frame = ttk.LabelFrame(scrollable_frame, text="Issues Found", padding=10)
        issues_frame.pack(fill=tk.X, padx=10, pady=10)

        for issue in outlier['issues'].split(';'):
            issue = issue.strip()
            if issue:
                issue_label = ttk.Label(issues_frame, text=f"• {issue}", wraplength=250, foreground='red')
                issue_label.pack(anchor=tk.W, pady=2)

        # Bounding boxes details
        if bbox_data:
            bbox_frame = ttk.LabelFrame(scrollable_frame, text="Bounding Boxes", padding=10)
            bbox_frame.pack(fill=tk.X, padx=10, pady=10)

            for idx, bbox in enumerate(bbox_data, 1):
                box_subframe = ttk.Frame(bbox_frame)
                box_subframe.pack(fill=tk.X, pady=5)

                box_title = f"Box {idx}: {bbox['label']}"
                if bbox['is_outlier']:
                    box_title += " (OUTLIER)"

                ttk.Label(box_subframe, text=box_title,
                          font=('TkDefaultFont', 9, 'bold'),
                          foreground='red' if bbox['is_outlier'] else 'black').pack(anchor=tk.W)

                ttk.Label(box_subframe,
                          text=f"  Position: ({bbox['xmin']}, {bbox['ymin']}) → ({bbox['xmax']}, {bbox['ymax']})",
                          font=('Consolas', 8)).pack(anchor=tk.W)

                width = bbox['xmax'] - bbox['xmin']
                height = bbox['ymax'] - bbox['ymin']
                ttk.Label(box_subframe, text=f"  Size: {width} × {height} px",
                          font=('Consolas', 8)).pack(anchor=tk.W)

        # Quality metrics
        if outlier.get('category') == 'quality' and 'metrics' in outlier:
            metrics_frame = ttk.LabelFrame(scrollable_frame, text="Quality Metrics", padding=10)
            metrics_frame.pack(fill=tk.X, padx=10, pady=10)

            metrics = outlier['metrics']
            for key, value in metrics.items():
                if key not in ['file', 'xml_path']:
                    ttk.Label(metrics_frame, text=f"{key}:", font=('TkDefaultFont', 9, 'bold')).pack(anchor=tk.W)
                    ttk.Label(metrics_frame, text=f"  {value:.2f}", font=('Consolas', 9)).pack(anchor=tk.W)

        # Bottom controls frame
        bottom_frame = ttk.Frame(self.preview_window)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)

        # Quick stats
        stats_frame = ttk.Frame(bottom_frame)
        stats_frame.pack(fill=tk.X, pady=(0, 5))

        # Calculate how many have been processed (deleted)
        processed = len(self.outliers) - len([o for o in self.outliers if o])  # This needs adjustment
        remaining_text = f"Remaining outliers: {len(self.outliers)} | "
        if hasattr(self, 'initial_outlier_count'):
            processed = self.initial_outlier_count - len(self.outliers)
            remaining_text += f"Processed: {processed} | "

        ttk.Label(stats_frame, text=remaining_text, font=('TkDefaultFont', 9)).pack(side=tk.LEFT)

        # Improved keyboard shortcuts display
        shortcuts_frame = ttk.LabelFrame(bottom_frame, text="Keyboard Shortcuts", padding="10")
        shortcuts_frame.pack(fill=tk.X, pady=(0, 5))

        shortcuts = [
            ("Navigate:", "← → or A D", "Previous/Next image"),
            ("Actions:", "Delete or X", "Delete image + XML"),
            ("Display:", "O", "Toggle overlay on/off"),
            ("Close:", "Esc or Q", "Close preview window")
        ]

        for i, (category, keys, description) in enumerate(shortcuts):
            row_frame = ttk.Frame(shortcuts_frame)
            row_frame.pack(fill=tk.X, pady=2)

            ttk.Label(row_frame, text=category, font=('TkDefaultFont', 9, 'bold'),
                      width=12).pack(side=tk.LEFT)
            ttk.Label(row_frame, text=keys, font=('Consolas', 9),
                      foreground='darkblue', width=18).pack(side=tk.LEFT)
            ttk.Label(row_frame, text=description, font=('TkDefaultFont', 9)).pack(side=tk.LEFT)

        # Legend
        legend_frame = ttk.Frame(bottom_frame)
        legend_frame.pack(fill=tk.X)

        legend_items = [
            ("●", "green", "Normal box"),
            ("●", "red", "Outlier box")
        ]

        for symbol, color, text in legend_items:
            item_frame = ttk.Frame(legend_frame)
            item_frame.pack(side=tk.LEFT, padx=10)

            tk.Label(item_frame, text=symbol, font=('TkDefaultFont', 16),
                     foreground=color).pack(side=tk.LEFT)
            ttk.Label(item_frame, text=text, font=('TkDefaultFont', 9)).pack(side=tk.LEFT, padx=5)

        # Enhanced keyboard bindings
        self.preview_window.bind('<Left>', lambda e: self.prev_outlier())
        self.preview_window.bind('<Right>', lambda e: self.next_outlier())
        self.preview_window.bind('<a>', lambda e: self.prev_outlier())
        self.preview_window.bind('<A>', lambda e: self.prev_outlier())
        self.preview_window.bind('<d>', lambda e: self.next_outlier())
        self.preview_window.bind('<D>', lambda e: self.next_outlier())
        self.preview_window.bind('<Delete>', lambda e: self.delete_outlier_quick(outlier))
        self.preview_window.bind('<x>', lambda e: self.delete_outlier_quick(outlier))
        self.preview_window.bind('<X>', lambda e: self.delete_outlier_quick(outlier))
        self.preview_window.bind('<Escape>', lambda e: self.preview_window.destroy())
        self.preview_window.bind('<q>', lambda e: self.preview_window.destroy())
        self.preview_window.bind('<Q>', lambda e: self.preview_window.destroy())
        self.preview_window.bind('<o>', lambda e: self.toggle_overlay_key(outlier))
        self.preview_window.bind('<O>', lambda e: self.toggle_overlay_key(outlier))

        # Focus window
        self.preview_window.focus_force()
        self.preview_window.grab_set()

        # Update main window status
        self._update_status(f"Previewing outlier {self.current_outlier_index + 1} of {len(self.outliers)}")

    def toggle_overlay_key(self, outlier):
        """Toggle overlay via keyboard"""
        self.show_overlay_var.set(not self.show_overlay_var.get())
        self.toggle_preview_overlay(outlier)

    def toggle_preview_overlay(self, outlier):
        """Redraw the image with or without overlay"""
        img_path = os.path.join(self.source_dir_var.get(), outlier['file'])
        img = cv2.imread(img_path)
        if img is None:
            return

        display_img = img.copy()

        # Redraw boxes if overlay is enabled
        if self.show_overlay_var.get():
            for bbox in self.current_bbox_data:
                color = (0, 0, 255) if bbox['is_outlier'] else (0, 255, 0)
                thickness = 3 if bbox['is_outlier'] else 2

                cv2.rectangle(display_img,
                              (bbox['xmin'], bbox['ymin']),
                              (bbox['xmax'], bbox['ymax']),
                              color, thickness)

                label_text = bbox['label']
                if bbox['is_outlier']:
                    label_text += " (OUTLIER)"

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]

                text_x = bbox['xmin']
                text_y = bbox['ymin'] - 5 if bbox['ymin'] > 20 else bbox['ymax'] + 20

                cv2.rectangle(display_img,
                              (text_x, text_y - text_size[1] - 4),
                              (text_x + text_size[0] + 4, text_y + 4),
                              color, -1)

                cv2.putText(display_img, label_text,
                            (text_x + 2, text_y),
                            font, font_scale,
                            (255, 255, 255), font_thickness)

        # Update display
        img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        max_size = 800
        if w > max_size or h > max_size:
            scale = min(max_size / w, max_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_rgb = cv2.resize(img_rgb, (new_w, new_h))

        pil_image = Image.fromarray(img_rgb)
        photo = ImageTk.PhotoImage(pil_image)

        self.current_img_label.configure(image=photo)
        self.current_img_label.image = photo

    def delete_outlier_quick(self, outlier):
        """Delete both image and XML files without confirmation for speed"""
        img_path = os.path.join(self.source_dir_var.get(), outlier['file'])
        xml_path = outlier.get('xml_path')

        if not xml_path:
            xml_path = os.path.join(self.source_dir_var.get(), Path(outlier['file']).stem + '.xml')

        try:
            deleted_files = []
            errors = []

            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    deleted_files.append(f"Image: {outlier['file']}")
                except Exception as e:
                    errors.append(f"Image: {str(e)}")

            if os.path.exists(xml_path):
                try:
                    os.remove(xml_path)
                    deleted_files.append(f"XML: {os.path.basename(xml_path)}")
                except Exception as e:
                    errors.append(f"XML: {str(e)}")

            if deleted_files:
                self.log_queue.put(f"Deleted: {outlier['file']} + XML")

            if errors:
                self.log_queue.put(f"Error deleting {outlier['file']}: {'; '.join(errors)}")
            else:
                self.outliers.remove(outlier)
                self.update_outlier_results()

                if self.outliers:
                    if self.current_outlier_index >= len(self.outliers):
                        self.current_outlier_index = len(self.outliers) - 1
                    self.show_outlier_preview()
                else:
                    self.preview_window.destroy()
                    self.log_queue.put("All outliers processed!")
                    messagebox.showinfo("Complete", "All outliers have been processed!")

        except Exception as e:
            self.log_queue.put(f"Error during deletion: {str(e)}")

    def prev_outlier(self):
        """Navigate to previous outlier"""
        if self.current_outlier_index > 0:
            self.current_outlier_index -= 1
            self.show_outlier_preview()
        else:
            self.current_outlier_index = len(self.outliers) - 1
            self.show_outlier_preview()

    def next_outlier(self):
        """Navigate to next outlier"""
        if self.current_outlier_index < len(self.outliers) - 1:
            self.current_outlier_index += 1
            self.show_outlier_preview()
        else:
            self.current_outlier_index = 0
            self.show_outlier_preview()

    def delete_outlier(self, outlier):
        """Redirect to quick delete for consistency"""
        self.delete_outlier_quick(outlier)

    def move_single_outlier(self, outlier):
        outlier_dir = os.path.join(self.source_dir_var.get(), self.outlier_folder_var.get())
        os.makedirs(outlier_dir, exist_ok=True)

        try:
            img_path = os.path.join(self.source_dir_var.get(), outlier['file'])
            xml_path = outlier['xml_path']

            new_img_path = os.path.join(outlier_dir, outlier['file'])
            new_xml_path = os.path.join(outlier_dir, os.path.basename(xml_path))

            shutil.move(img_path, new_img_path)
            shutil.move(xml_path, new_xml_path)

            self.log_queue.put(f"Moved to outlier folder: {outlier['file']}")

            self.outliers.remove(outlier)
            if self.outliers:
                if self.current_outlier_index >= len(self.outliers):
                    self.current_outlier_index = len(self.outliers) - 1
                self.show_outlier_preview()
            else:
                self.preview_window.destroy()
                messagebox.showinfo("Info", "No more outliers to preview.")

        except Exception as e:
            messagebox.showerror("Error", f"Could not move files: {str(e)}")

    def move_outliers(self):
        if not self.outliers:
            messagebox.showinfo("Info", "No outliers to move. Run outlier detection first.")
            return

        result = messagebox.askyesno("Confirm Move",
                                     f"Move {len(self.outliers)} outlier files to '{self.outlier_folder_var.get()}' folder?")
        if result:
            threading.Thread(target=self._move_outliers_thread, daemon=True).start()

    def _move_outliers_thread(self):
        try:
            outlier_dir = os.path.join(self.source_dir_var.get(), self.outlier_folder_var.get())
            os.makedirs(outlier_dir, exist_ok=True)

            moved = 0
            errors = 0

            for i, outlier in enumerate(self.outliers):
                self._update_progress((i / len(self.outliers)) * 100)
                self._update_status(f"Moving file {i + 1}/{len(self.outliers)}")

                try:
                    img_path = os.path.join(self.source_dir_var.get(), outlier['file'])
                    xml_path = outlier['xml_path']

                    new_img_path = os.path.join(outlier_dir, outlier['file'])
                    new_xml_path = os.path.join(outlier_dir, os.path.basename(xml_path))

                    shutil.move(img_path, new_img_path)
                    shutil.move(xml_path, new_xml_path)
                    moved += 1

                except Exception as e:
                    self.log_queue.put(f"Error moving {outlier['file']}: {str(e)}")
                    errors += 1

            self._update_status(f"Moved {moved} files to outlier folder. Errors: {errors}")
            self._update_progress(100)
            self.log_queue.put(f"\n✅ Moved {moved} outlier files to: {outlier_dir}")

            if errors > 0:
                self.log_queue.put(f"⚠️ Failed to move {errors} files")

            self.outliers = []
            self.update_outlier_results()

        except Exception as e:
            self.log_queue.put(f"Error during move operation: {str(e)}")

    def on_tree_double_click(self, event):
        selection = self.outlier_tree.selection()
        if selection:
            item = self.outlier_tree.item(selection[0])
            filename = item['values'][0]

            for i, outlier in enumerate(self.outliers):
                if outlier['file'] == filename:
                    self.current_outlier_index = i
                    self.show_outlier_preview()
                    break

    def _update_status(self, message):
        self.parent.after(0, lambda: self.status_label.config(text=message))

    def _update_progress(self, value, message=""):
        self.parent.after(0, lambda: self.progress_var.set(value))
        if message:
            self.parent.after(0, lambda: self.status_label.config(text=message))

    def sync_xml_filenames(self) -> None:
        """
        Synchronize XML filename and path fields with actual image files.
        Process train, valid, and test folders if they exist.
        """
        try:
            source_dir = self.source_dir_var.get()
            if not source_dir:
                self.log_queue.put("No source directory specified")
                return

            # Define folders to process
            folders_to_process: List[str] = ['train', 'valid', 'test']

            total_fixed = 0
            total_skipped = 0

            # Process each folder
            for folder in folders_to_process:
                folder_path = os.path.join(source_dir, folder)

                # Check if folder exists
                if not os.path.exists(folder_path):
                    self.log_queue.put(f"Folder '{folder}' not found, checking root directory...")
                    # If subfolder doesn't exist, try the root directory itself
                    if folder == 'train' and os.path.exists(source_dir):
                        # Process root directory if no subfolders exist
                        folder_path = source_dir
                        folder = os.path.basename(source_dir)
                    else:
                        continue

                self.log_queue.put(f"\n🔍 Processing folder: {folder}")
                result = self.sync_folder(folder_path)

                total_fixed += result['fixed']
                total_skipped += result['skipped']

                self.log_queue.put(
                    f"📁 Done with folder '{folder}': Fixed={result['fixed']}, Skipped={result['skipped']}"
                )

            # If no subfolders were found, process the root directory
            if total_fixed == 0 and total_skipped == 0:
                self.log_queue.put("\n🔍 No subfolders found, processing root directory")
                result = self.sync_folder(source_dir)
                total_fixed = result['fixed']
                total_skipped = result['skipped']
                self.log_queue.put(
                    f"📁 Done with root directory: Fixed={result['fixed']}, Skipped={result['skipped']}"
                )

            self.log_queue.put(f"\n✅ XML filename sync completed: Fixed={total_fixed}, Skipped={total_skipped}")

        except Exception as e:
            self.log_queue.put(f"❌ Error during XML filename sync: {str(e)}")

    def sync_folder(self, folder_path: str) -> Dict[str, int]:
        """
        Fix <filename> and <path> fields in XML files for a given folder.

        Args:
            folder_path: Path to the folder containing XML and image files

        Returns:
            Dictionary with 'fixed' and 'skipped' counts
        """
        try:
            # Get all XML files in the folder
            xml_files = glob.glob(os.path.join(folder_path, '*.xml'))

            self.log_queue.put(f"Found {len(xml_files)} XML files in {os.path.basename(folder_path)}")

            fixed = 0
            skipped = 0

            for idx, xml_path in enumerate(xml_files, start=1):
                if self.stop_flag.is_set():
                    break

                # Get the base name without extension
                base_name = os.path.splitext(os.path.basename(xml_path))[0]

                # Find corresponding image file
                img_filename = self._find_image_file(folder_path, base_name)

                if not img_filename:
                    self.log_queue.put(f"⚠️ No image found for {base_name}")
                    skipped += 1
                    continue

                try:
                    # Parse XML file
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    modified = False

                    # Update <filename> element
                    filename_elem = root.find('filename')
                    if filename_elem is None:
                        filename_elem = ET.SubElement(root, 'filename')
                        modified = True

                    old_filename = filename_elem.text
                    if old_filename != img_filename:
                        filename_elem.text = img_filename
                        modified = True

                    # Update <path> element
                    path_elem = root.find('path')
                    if path_elem is None:
                        path_elem = ET.SubElement(root, 'path')
                        modified = True

                    # Create absolute path for the image
                    new_path = os.path.abspath(os.path.join(folder_path, img_filename))
                    old_path = path_elem.text

                    if old_path != new_path:
                        path_elem.text = new_path
                        modified = True

                    # Write back if modified
                    if modified:
                        # Write XML with proper formatting
                        self._indent_xml(root)
                        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                        fixed += 1

                        self.log_queue.put(
                            f"✅ Fixed [{idx}/{len(xml_files)}]: {os.path.basename(xml_path)} "
                            f"(filename: '{old_filename}' → '{img_filename}')"
                        )

                    # Update progress
                    progress = (idx / len(xml_files)) * 100
                    self._update_status(
                        f"Processing XML files in {os.path.basename(folder_path)}: {idx}/{len(xml_files)}")

                except ET.ParseError as e:
                    self.log_queue.put(f"❌ Error parsing {xml_path}: {e}")
                    skipped += 1
                except Exception as e:
                    self.log_queue.put(f"❌ Error fixing {xml_path}: {e}")
                    skipped += 1

            return {'fixed': fixed, 'skipped': skipped}

        except Exception as e:
            self.log_queue.put(f"❌ Error processing folder {folder_path}: {str(e)}")
            return {'fixed': 0, 'skipped': 0}

    def _find_image_file(self, directory: str, base_name: str) -> Optional[str]:
        """
        Find an image file with the given base name in the directory.

        Args:
            directory: Directory to search in
            base_name: Base filename without extension

        Returns:
            Image filename with extension if found, None otherwise
        """
        # Check for various image extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

        for ext in image_extensions:
            image_path = os.path.join(directory, base_name + ext)
            if os.path.isfile(image_path):
                return base_name + ext

        return None

    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        """
        Add pretty-printing indentation to XML elements.

        Args:
            elem: XML element to indent
            level: Current indentation level
        """
        indent = "\n" + "    " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent

    def on_sync_recursive_changed(self) -> None:
        """Update info label when recursive option changes"""
        if self.sync_recursive_var.get():
            self.sync_info_label.config(
                text="Recursive mode: Will process all subfolders (each folder in separate thread)",
                foreground='blue'
            )
        else:
            self.sync_info_label.config(
                text="Single folder mode: Will process current directory only",
                foreground='gray'
            )

    def start_xml_sync(self) -> None:
        """Start the XML sync operation with multiprocessing support"""
        if not self.source_dir_var.get():
            messagebox.showwarning("Warning", "Please select a source directory first.")
            return

        self.sync_stop_flag.clear()
        self.sync_thread = threading.Thread(target=self._xml_sync_thread, daemon=True)
        self.sync_thread.start()

    def stop_xml_sync(self) -> None:
        """Stop the XML sync operation"""
        self.sync_stop_flag.set()
        self.log_queue.put("Stopping XML sync process...")

    def _xml_sync_thread(self) -> None:
        """Main thread for XML sync with multiprocessing support"""
        try:
            source_dir = self.source_dir_var.get()
            recursive = self.sync_recursive_var.get()
            max_threads = int(self.sync_max_threads_var.get())

            self.log_queue.put("\n=== Starting XML Filename/Path Sync ===")
            self._update_status("Analyzing folders...")
            self._update_progress(0)

            if recursive:
                # Find all folders with XML files
                folders_to_process = self._find_folders_with_xml(source_dir)

                if not folders_to_process:
                    self.log_queue.put("No folders with XML files found.")
                    self._update_status("No XML files found")
                    return

                self.log_queue.put(f"Found {len(folders_to_process)} folders to process")
                self.log_queue.put(f"Using up to {max_threads} concurrent threads")

                # Process folders concurrently
                self._process_folders_concurrent(folders_to_process, max_threads)

            else:
                # Single folder processing
                self.log_queue.put(f"Processing single folder: {os.path.basename(source_dir)}")

                # Check if we need to split large folder
                xml_files = glob.glob(os.path.join(source_dir, '*.xml'))

                if len(xml_files) > 1500:
                    self.log_queue.put(f"Large folder detected ({len(xml_files)} files), using parallel processing")
                    self._process_large_folder_parallel(source_dir, xml_files)
                else:
                    result = self._sync_folder_with_progress(source_dir)
                    self.log_queue.put(f"Completed: Fixed={result['fixed']}, Skipped={result['skipped']}")

            if not self.sync_stop_flag.is_set():
                self._update_status("XML sync completed!")
                self.log_queue.put("\n✅ XML sync operation completed successfully!")
            else:
                self._update_status("XML sync stopped by user")
                self.log_queue.put("\n⚠️ XML sync operation stopped by user")

        except Exception as e:
            self.log_queue.put(f"\n❌ Error during XML sync: {str(e)}")
            self._update_status(f"Error: {str(e)}")

    def _find_folders_with_xml(self, root_dir: str) -> List[str]:
        """Find all folders containing XML files"""
        folders = []

        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Check if this directory has XML files
            xml_files = [f for f in filenames if f.endswith('.xml')]
            if xml_files:
                folders.append(dirpath)

        return folders

    def _process_folders_concurrent(self, folders: List[str], max_threads: int) -> None:
        """Process multiple folders concurrently using ThreadPoolExecutor"""
        total_fixed = 0
        total_skipped = 0
        completed = 0
        total = len(folders)

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Submit all folders for processing
            future_to_folder = {}

            for folder in folders:
                if self.sync_stop_flag.is_set():
                    break

                future = executor.submit(self._sync_folder_with_progress, folder)
                future_to_folder[future] = folder

            # Process completed futures
            for future in as_completed(future_to_folder):
                if self.sync_stop_flag.is_set():
                    executor.shutdown(wait=False)
                    break

                folder = future_to_folder[future]
                folder_name = os.path.relpath(folder, self.source_dir_var.get())

                try:
                    result = future.result()
                    total_fixed += result['fixed']
                    total_skipped += result['skipped']
                    completed += 1

                    self.log_queue.put(
                        f"[{completed}/{total}] Completed '{folder_name}': "
                        f"Fixed={result['fixed']}, Skipped={result['skipped']}"
                    )

                    # Update progress
                    progress = (completed / total) * 100
                    self._update_progress(progress)
                    self._update_status(f"Processing folders: {completed}/{total}")

                except Exception as e:
                    self.log_queue.put(f"❌ Error processing '{folder_name}': {str(e)}")
                    completed += 1

        self.log_queue.put(f"\n📊 Total results: Fixed={total_fixed}, Skipped={total_skipped}")

    def _process_large_folder_parallel(self, folder_path: str, xml_files: List[str]) -> None:
        """Process a large folder by splitting it into two parts and processing in parallel"""
        # Split files into two halves
        mid_point = len(xml_files) // 2
        batch1 = xml_files[:mid_point]
        batch2 = xml_files[mid_point:]

        self.log_queue.put(f"Splitting {len(xml_files)} files into two batches: {len(batch1)} and {len(batch2)}")

        total_fixed = 0
        total_skipped = 0

        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._process_xml_batch, folder_path, batch1, 1)
            future2 = executor.submit(self._process_xml_batch, folder_path, batch2, 2)

            for future in as_completed([future1, future2]):
                if self.sync_stop_flag.is_set():
                    executor.shutdown(wait=False)
                    break

                try:
                    result = future.result()
                    total_fixed += result['fixed']
                    total_skipped += result['skipped']

                    batch_num = 1 if future == future1 else 2
                    self.log_queue.put(
                        f"Batch {batch_num} completed: Fixed={result['fixed']}, Skipped={result['skipped']}"
                    )
                except Exception as e:
                    self.log_queue.put(f"❌ Error in batch processing: {str(e)}")

        self.log_queue.put(f"📊 Folder totals: Fixed={total_fixed}, Skipped={total_skipped}")

    def _process_xml_batch(self, folder_path: str, xml_files: List[str], batch_num: int) -> Dict[str, int]:
        """Process a batch of XML files"""
        fixed = 0
        skipped = 0

        for idx, xml_path in enumerate(xml_files, start=1):
            if self.sync_stop_flag.is_set():
                break

            # Get the base name without extension
            base_name = os.path.splitext(os.path.basename(xml_path))[0]

            # Find corresponding image file
            img_filename = self._find_image_file(folder_path, base_name)

            if not img_filename:
                skipped += 1
                continue

            try:
                # Parse XML file
                tree = ET.parse(xml_path)
                root = tree.getroot()

                modified = False

                # Update <filename> element
                filename_elem = root.find('filename')
                if filename_elem is None:
                    filename_elem = ET.SubElement(root, 'filename')
                    modified = True

                old_filename = filename_elem.text
                if old_filename != img_filename:
                    filename_elem.text = img_filename
                    modified = True

                # Update <path> element
                path_elem = root.find('path')
                if path_elem is None:
                    path_elem = ET.SubElement(root, 'path')
                    modified = True

                # Create absolute path for the image
                new_path = os.path.abspath(os.path.join(folder_path, img_filename))
                old_path = path_elem.text

                if old_path != new_path:
                    path_elem.text = new_path
                    modified = True

                # Write back if modified
                if modified:
                    self._indent_xml(root)
                    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                    fixed += 1

                    if idx % 100 == 0:  # Log progress every 100 files
                        self.log_queue.put(f"Batch {batch_num}: Processed {idx}/{len(xml_files)} files")

            except Exception as e:
                skipped += 1

        return {'fixed': fixed, 'skipped': skipped}

    def _sync_folder_with_progress(self, folder_path: str) -> Dict[str, int]:
        """Sync a folder with progress updates"""
        try:
            # Get all XML files in the folder
            xml_files = glob.glob(os.path.join(folder_path, '*.xml'))

            if not xml_files:
                return {'fixed': 0, 'skipped': 0}

            folder_name = os.path.relpath(folder_path,
                                          self.source_dir_var.get()) if self.source_dir_var.get() else os.path.basename(
                folder_path)
            self.log_queue.put(f"Processing '{folder_name}' ({len(xml_files)} XML files)")

            fixed = 0
            skipped = 0

            for idx, xml_path in enumerate(xml_files, start=1):
                if self.sync_stop_flag.is_set():
                    break

                # Get the base name without extension
                base_name = os.path.splitext(os.path.basename(xml_path))[0]

                # Find corresponding image file
                img_filename = self._find_image_file(folder_path, base_name)

                if not img_filename:
                    skipped += 1
                    continue

                try:
                    # Parse XML file
                    tree = ET.parse(xml_path)
                    root = tree.getroot()

                    modified = False

                    # Update <filename> element
                    filename_elem = root.find('filename')
                    if filename_elem is None:
                        filename_elem = ET.SubElement(root, 'filename')
                        modified = True

                    old_filename = filename_elem.text
                    if old_filename != img_filename:
                        filename_elem.text = img_filename
                        modified = True

                    # Update <path> element
                    path_elem = root.find('path')
                    if path_elem is None:
                        path_elem = ET.SubElement(root, 'path')
                        modified = True

                    # Create absolute path for the image
                    new_path = os.path.abspath(os.path.join(folder_path, img_filename))
                    old_path = path_elem.text

                    if old_path != new_path:
                        path_elem.text = new_path
                        modified = True

                    # Write back if modified
                    if modified:
                        self._indent_xml(root)
                        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
                        fixed += 1

                except Exception as e:
                    skipped += 1

            return {'fixed': fixed, 'skipped': skipped}

        except Exception as e:
            self.log_queue.put(f"❌ Error processing folder {folder_path}: {str(e)}")
            return {'fixed': 0, 'skipped': 0}
