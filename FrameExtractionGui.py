import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing


class FrameExtractionTab:
    def __init__(self, parent, log_queue):
        self.parent = parent
        self.log_queue = log_queue
        self.app_root = os.path.dirname(os.path.abspath(__file__))
        self.executor = None
        self.futures = []
        self.setup_ui()

    def setup_ui(self):
        # Main frame directly in parent - no canvas/scrollbar
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Section 1: Source Selection
        ttk.Label(main_frame, text="Video Source",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(0, 5))

        # Source type selection
        source_frame = ttk.LabelFrame(main_frame, text="Select Source Type", padding="10")
        source_frame.pack(fill='x', pady=(0, 5))

        self.source_type_var = tk.StringVar(value="directory")
        ttk.Radiobutton(source_frame, text="Process all videos in directory",
                        variable=self.source_type_var, value="directory",
                        command=self.toggle_source_type).pack(anchor='w')
        ttk.Radiobutton(source_frame, text="Process single video file",
                        variable=self.source_type_var, value="single",
                        command=self.toggle_source_type).pack(anchor='w')

        # Directory selection
        self.dir_frame = ttk.Frame(main_frame)
        self.dir_frame.pack(fill='x', pady=(5, 0))

        dir_inner = ttk.Frame(self.dir_frame)
        dir_inner.pack(fill='x')

        ttk.Label(dir_inner, text="Video Directory:").pack(side='left', padx=(0, 10))
        self.video_dir_var = tk.StringVar()
        ttk.Entry(dir_inner, textvariable=self.video_dir_var, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(dir_inner, text="Browse", command=self.browse_video_dir).pack(side='left', padx=(5, 0))

        # Single file selection
        self.file_frame = ttk.Frame(main_frame)

        file_inner = ttk.Frame(self.file_frame)
        file_inner.pack(fill='x')

        ttk.Label(file_inner, text="Video File:").pack(side='left', padx=(0, 10))
        self.video_file_var = tk.StringVar()
        ttk.Entry(file_inner, textvariable=self.video_file_var, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(file_inner, text="Browse", command=self.browse_video_file).pack(side='left', padx=(5, 0))

        # Section 2: Output Settings
        ttk.Label(main_frame, text="Output Settings",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(10, 5))

        settings_frame = ttk.LabelFrame(main_frame, text="Frame Extraction Settings", padding="10")
        settings_frame.pack(fill='x', pady=(0, 5))

        # Output directory
        output_dir_frame = ttk.Frame(settings_frame)
        output_dir_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(output_dir_frame, text="Output Directory:").pack(side='left', padx=(0, 10))
        self.output_dir_var = tk.StringVar()
        ttk.Entry(output_dir_frame, textvariable=self.output_dir_var, width=40).pack(side='left', fill='x', expand=True)
        ttk.Button(output_dir_frame, text="Browse", command=self.browse_output_dir).pack(side='left', padx=(5, 0))

        # Create directory option
        self.create_dir_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Create directory if it doesn't exist",
                        variable=self.create_dir_var).pack(anchor='w', pady=(5, 0))

        # Two-column layout for settings
        settings_columns = ttk.Frame(settings_frame)
        settings_columns.pack(fill='x', pady=(10, 0))

        # Left column
        left_col = ttk.Frame(settings_columns)
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))

        # Filename prefix
        prefix_frame = ttk.Frame(left_col)
        prefix_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(prefix_frame, text="Filename Prefix:").pack(side='left', padx=(0, 10))
        self.prefix_var = tk.StringVar(value="frame")
        ttk.Entry(prefix_frame, textvariable=self.prefix_var, width=15).pack(side='left')

        # Number of threads
        thread_frame = ttk.Frame(left_col)
        thread_frame.pack(fill='x', pady=(5, 0))

        ttk.Label(thread_frame, text="Processing Threads:").pack(side='left', padx=(0, 10))
        self.num_threads_var = tk.StringVar(value=str(min(4, multiprocessing.cpu_count())))
        thread_spinbox = ttk.Spinbox(thread_frame, from_=1, to=multiprocessing.cpu_count(),
                                     textvariable=self.num_threads_var, width=10)
        thread_spinbox.pack(side='left')

        # Right column
        right_col = ttk.Frame(settings_columns)
        right_col.pack(side='left', fill='both', expand=True)

        # Resolution
        res_frame = ttk.Frame(right_col)
        res_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(res_frame, text="Output Resolution:").pack(side='left', padx=(0, 10))
        self.resolution_var = tk.StringVar(value="original")  # Changed default to "original"
        resolution_combo = ttk.Combobox(res_frame, textvariable=self.resolution_var,
                                        values=["original", "640x480", "800x600", "1024x768", "1280x720", "1920x1080",
                                                "Custom"],
                                        width=15, state="readonly")
        resolution_combo.pack(side='left')
        resolution_combo.bind('<<ComboboxSelected>>', self.on_resolution_change)

        # Custom resolution inputs
        self.custom_res_frame = ttk.Frame(right_col)

        ttk.Label(self.custom_res_frame, text="  Width:").pack(side='left', padx=(0, 5))
        self.custom_width_var = tk.StringVar(value="800")
        ttk.Entry(self.custom_res_frame, textvariable=self.custom_width_var, width=8).pack(side='left', padx=(0, 10))
        ttk.Label(self.custom_res_frame, text="Height:").pack(side='left', padx=(0, 5))
        self.custom_height_var = tk.StringVar(value="600")
        ttk.Entry(self.custom_res_frame, textvariable=self.custom_height_var, width=8).pack(side='left')

        # Aspect ratio preservation option
        self.preserve_aspect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right_col, text="Preserve aspect ratio (fit within bounds)",
                        variable=self.preserve_aspect_var).pack(anchor='w', pady=(5, 0))

        # Interpolation method
        interp_frame = ttk.Frame(settings_frame)
        interp_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(interp_frame, text="Interpolation Method:").pack(side='left', padx=(0, 10))
        self.interpolation_var = tk.StringVar(value="INTER_LINEAR")
        self.interpolation_combo = ttk.Combobox(interp_frame, textvariable=self.interpolation_var,
                                                values=["INTER_NEAREST", "INTER_LINEAR", "INTER_CUBIC",
                                                        "INTER_AREA", "INTER_LANCZOS4"],
                                                width=20, state="readonly")
        self.interpolation_combo.pack(side='left')
        ttk.Button(interp_frame, text="?", width=3,
                   command=self.show_interpolation_info).pack(side='left', padx=(5, 0))

        # Frame extraction options
        extraction_frame = ttk.LabelFrame(settings_frame, text="Frame Selection", padding="5")
        extraction_frame.pack(fill='x', pady=(10, 0))

        self.extract_all_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(extraction_frame, text="Extract all frames",
                        variable=self.extract_all_var, value=True).pack(anchor='w')

        nth_frame_container = ttk.Frame(extraction_frame)
        nth_frame_container.pack(anchor='w')

        ttk.Radiobutton(nth_frame_container, text="Extract every",
                        variable=self.extract_all_var, value=False).pack(side='left')
        self.nth_frame_var = tk.StringVar(value="5")
        ttk.Entry(nth_frame_container, textvariable=self.nth_frame_var, width=5).pack(side='left', padx=(5, 5))
        ttk.Label(nth_frame_container, text="frames").pack(side='left')

        # Section 3: Progress - REFACTORED TO SINGLE AREA
        ttk.Label(main_frame, text="Progress",
                  font=('TkDefaultFont', 12, 'bold')).pack(anchor='w', pady=(10, 5))

        progress_container = ttk.Frame(main_frame)
        progress_container.pack(fill='both', expand=True, pady=(0, 5))

        # Overall progress at top
        overall_frame = ttk.LabelFrame(progress_container, text="Overall Progress", padding="10")
        overall_frame.pack(fill='x', pady=(0, 5))

        self.overall_progress_var = tk.DoubleVar()
        self.overall_progress_bar = ttk.Progressbar(overall_frame, variable=self.overall_progress_var, maximum=100)
        self.overall_progress_bar.pack(fill='x', pady=(0, 5))

        self.overall_status_label = ttk.Label(overall_frame, text="Ready")
        self.overall_status_label.pack()

        # Individual videos in scrollable area below
        videos_label_frame = ttk.LabelFrame(progress_container, text="Individual Videos", padding="5")
        videos_label_frame.pack(fill='both', expand=True)

        # Scrollable frame for video progress bars
        self.videos_canvas = tk.Canvas(videos_label_frame, height=150)
        videos_scrollbar = ttk.Scrollbar(videos_label_frame, orient="vertical", command=self.videos_canvas.yview)
        self.videos_scroll_frame = ttk.Frame(self.videos_canvas)

        self.videos_scroll_frame.bind(
            "<Configure>",
            lambda e: self.videos_canvas.configure(scrollregion=self.videos_canvas.bbox("all"))
        )

        self.videos_canvas.create_window((0, 0), window=self.videos_scroll_frame, anchor="nw")
        self.videos_canvas.configure(yscrollcommand=videos_scrollbar.set)

        self.videos_canvas.pack(side="left", fill="both", expand=True)
        videos_scrollbar.pack(side="right", fill="y")

        # Placeholder text
        self.placeholder_label = ttk.Label(self.videos_scroll_frame, text="No videos processing...",
                                           foreground="gray")
        self.placeholder_label.pack(pady=20)

        # Video info label
        self.video_info_label = ttk.Label(main_frame, text="", foreground="blue")
        self.video_info_label.pack(pady=(5, 0))

        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=(10, 0))

        self.analyze_button = ttk.Button(
            button_frame,
            text="Analyze Videos",
            command=self.analyze_videos
        )
        self.analyze_button.pack(side='left', padx=5)

        self.start_button = ttk.Button(
            button_frame,
            text="Start Extraction",
            command=self.start_extraction
        )
        self.start_button.pack(side='left', padx=5)

        self.stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self.stop_extraction,
            state='disabled'
        )
        self.stop_button.pack(side='left', padx=5)

        # Initialize
        self.stop_flag = threading.Event()
        self.video_progress_bars = {}
        self.video_progress_vars = {}
        self.video_status_labels = {}
        self.total_videos = 0
        self.completed_videos = 0

        # Set initial visibility
        self.toggle_source_type()

    def toggle_source_type(self):
        if self.source_type_var.get() == "directory":
            self.dir_frame.pack(fill='x', pady=(5, 0))
            self.file_frame.pack_forget()
        else:
            self.file_frame.pack(fill='x', pady=(5, 0))
            self.dir_frame.pack_forget()

    def browse_video_dir(self):
        directory = filedialog.askdirectory(
            title="Select Video Directory",
            initialdir=self.app_root
        )
        if directory:
            self.video_dir_var.set(directory)

    def browse_video_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=self.app_root,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_file_var.set(filename)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self.app_root
        )
        if directory:
            self.output_dir_var.set(directory)

    def on_resolution_change(self, event=None):
        if self.resolution_var.get() == "Custom":
            self.custom_res_frame.pack(fill='x', pady=(5, 0))
        else:
            self.custom_res_frame.pack_forget()

    def show_interpolation_info(self):
        info_window = tk.Toplevel(self.parent)
        info_window.title("Interpolation Methods")
        info_window.geometry("600x400")

        info_text = scrolledtext.ScrolledText(info_window, wrap=tk.WORD, padx=10, pady=10)
        info_text.pack(fill='both', expand=True)

        interpolation_info = """
INTERPOLATION METHODS FOR IMAGE RESIZING:

• INTER_NEAREST (Nearest Neighbor)
  - Fastest method, lowest quality
  - Best for: Pixel art, images with hard edges
  - Use when: Speed is critical, quality less important

• INTER_LINEAR (Bilinear) - RECOMMENDED
  - Good balance of speed and quality
  - Best for: General purpose, most video frames
  - Use when: Standard frame extraction

• INTER_CUBIC (Bicubic)
  - Higher quality, slower than linear
  - Best for: Upscaling, smooth gradients
  - Use when: Quality is important, slight upscaling

• INTER_AREA
  - Best for downscaling (shrinking images)
  - Best for: Reducing image size
  - Use when: Output smaller than input

• INTER_LANCZOS4 (Lanczos)
  - Highest quality, slowest
  - Best for: High-quality results, professional work
  - Use when: Maximum quality needed

For frame extraction, INTER_LINEAR or INTER_AREA (for downscaling) are typically best choices.
"""
        info_text.insert(1.0, interpolation_info)
        info_text.config(state='disabled')

        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)

    def analyze_videos(self):
        if self.source_type_var.get() == "directory":
            if not self.video_dir_var.get():
                messagebox.showwarning("Warning", "Please select a video directory first.")
                return
            self._analyze_directory()
        else:
            if not self.video_file_var.get():
                messagebox.showwarning("Warning", "Please select a video file first.")
                return
            self._analyze_single_file()

    def _analyze_directory(self):
        video_dir = self.video_dir_var.get()
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = []

        for file in os.listdir(video_dir):
            if Path(file).suffix.lower() in video_extensions:
                video_files.append(os.path.join(video_dir, file))

        if not video_files:
            self.log_queue.put("No video files found in the selected directory.")
            return

        # Get frame extraction settings
        extract_all = self.extract_all_var.get()
        nth_frame = 1 if extract_all else int(self.nth_frame_var.get() or 1)

        self.log_queue.put(f"\nFound {len(video_files)} video files:")
        self.log_queue.put(f"Extraction setting: Every {nth_frame} frame(s)\n")

        total_frames = 0
        total_to_extract = 0

        for video_path in video_files:
            frames, info = self._get_video_info(video_path)
            if frames > 0:
                frames_to_extract = frames // nth_frame
                self.log_queue.put(f"  • {os.path.basename(video_path)}: {info}")
                self.log_queue.put(f"    → Will extract {frames_to_extract} of {frames} frames")
                total_frames += frames
                total_to_extract += frames_to_extract

        self.log_queue.put(f"\nTotal frames: {total_frames}")
        self.log_queue.put(f"Total frames to extract: {total_to_extract}")
        self.video_info_label.config(
            text=f"Total videos: {len(video_files)}, Frames to extract: {total_to_extract}/{total_frames}"
        )

    def _analyze_single_file(self):
        video_path = self.video_file_var.get()
        frames, info = self._get_video_info(video_path)

        if frames > 0:
            # Calculate how many frames will be extracted based on settings
            extract_all = self.extract_all_var.get()
            nth_frame = 1 if extract_all else int(self.nth_frame_var.get() or 1)
            frames_to_extract = frames // nth_frame

            self.log_queue.put(f"\nVideo file: {os.path.basename(video_path)}")
            self.log_queue.put(f"Info: {info}")
            self.log_queue.put(f"Extraction setting: Every {nth_frame} frame(s)")
            self.log_queue.put(f"Frames to extract: {frames_to_extract} of {frames} total frames")

            self.video_info_label.config(
                text=f"Single video: {frames} total frames, {frames_to_extract} to extract (every {nth_frame} frame)"
            )
        else:
            self.log_queue.put(f"Could not analyze video: {os.path.basename(video_path)}")

    def _get_video_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()

            info = f"{frame_count} frames, {width}x{height}, {fps:.1f} fps, {duration:.1f}s"
            return frame_count, info
        return 0, "Could not open"

    def calculate_aspect_ratio_size(self, orig_width, orig_height, target_width, target_height):
        """Calculate size that fits within target while preserving aspect ratio"""
        # Calculate scaling factors
        width_scale = target_width / orig_width
        height_scale = target_height / orig_height

        # Use the smaller scale to ensure the image fits within bounds
        scale = min(width_scale, height_scale)

        # Calculate new dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)

        return new_width, new_height

    def start_extraction(self):
        if not self.output_dir_var.get():
            messagebox.showwarning("Warning", "Please select an output directory.")
            return

        if self.source_type_var.get() == "directory" and not self.video_dir_var.get():
            messagebox.showwarning("Warning", "Please select a video directory.")
            return

        if self.source_type_var.get() == "single" and not self.video_file_var.get():
            messagebox.showwarning("Warning", "Please select a video file.")
            return

        # Create output directory if needed
        if self.create_dir_var.get() and not os.path.exists(self.output_dir_var.get()):
            try:
                os.makedirs(self.output_dir_var.get())
                self.log_queue.put(f"Created output directory: {self.output_dir_var.get()}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory: {str(e)}")
                return

        # Update button states
        self.start_button.config(state='disabled')
        self.analyze_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Clear previous progress bars
        for widget in self.videos_scroll_frame.winfo_children():
            widget.destroy()
        self.video_progress_bars.clear()
        self.video_progress_vars.clear()
        self.video_status_labels.clear()

        # Reset counters
        self.completed_videos = 0
        self.total_videos = 0

        self.stop_flag.clear()
        threading.Thread(target=self._start_multi_threaded_extraction, daemon=True).start()

    def stop_extraction(self):
        self.stop_flag.set()
        self.log_queue.put("Stopping extraction process...")
        if self.executor:
            self.executor.shutdown(wait=False)
            self.futures = []

        # Reset button states
        self.start_button.config(state='normal')
        self.analyze_button.config(state='normal')
        self.stop_button.config(state='disabled')

    def _start_multi_threaded_extraction(self):
        try:
            output_dir = self.output_dir_var.get()
            prefix = self.prefix_var.get() or "frame"
            num_threads = int(self.num_threads_var.get())

            # Get target resolution and aspect ratio setting
            preserve_aspect = self.preserve_aspect_var.get()
            use_original = self.resolution_var.get() == "original"

            if not use_original:
                if self.resolution_var.get() == "Custom":
                    target_width = int(self.custom_width_var.get())
                    target_height = int(self.custom_height_var.get())
                else:
                    width, height = self.resolution_var.get().split('x')
                    target_width = int(width)
                    target_height = int(height)
            else:
                target_width = None
                target_height = None

            # Get interpolation method
            interpolation_map = {
                "INTER_NEAREST": cv2.INTER_NEAREST,
                "INTER_LINEAR": cv2.INTER_LINEAR,
                "INTER_CUBIC": cv2.INTER_CUBIC,
                "INTER_AREA": cv2.INTER_AREA,
                "INTER_LANCZOS4": cv2.INTER_LANCZOS4
            }
            interpolation = interpolation_map.get(self.interpolation_var.get(), cv2.INTER_LINEAR)

            # Get frame extraction settings - FIX: Use correct value
            extract_all = self.extract_all_var.get()
            nth_frame = 1 if extract_all else int(self.nth_frame_var.get() or 1)

            # Get video files
            if self.source_type_var.get() == "directory":
                video_dir = self.video_dir_var.get()
                video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
                video_files = [os.path.join(video_dir, f) for f in os.listdir(video_dir)
                               if Path(f).suffix.lower() in video_extensions]
            else:
                video_files = [self.video_file_var.get()]

            if not video_files:
                self.log_queue.put("No video files found!")
                return

            self.total_videos = len(video_files)
            self.completed_videos = 0

            self.log_queue.put(f"\n=== Starting Multi-threaded Frame Extraction ===")
            if use_original:
                self.log_queue.put(f"Output resolution: Original (no resizing)")
            else:
                self.log_queue.put(f"Output resolution: {target_width}x{target_height}")
                self.log_queue.put(f"Preserve aspect ratio: {preserve_aspect}")
                self.log_queue.put(f"Interpolation: {self.interpolation_var.get()}")
            self.log_queue.put(f"Prefix: '{prefix}'")
            self.log_queue.put(f"Extracting every {nth_frame} frame(s)")  # FIX: Now shows correct value
            self.log_queue.put(f"Using {num_threads} threads\n")

            # Create progress bars for each video
            for video_path in video_files:
                video_name = os.path.basename(video_path)
                self._create_video_progress_bar(video_name)

            # Update overall status
            self._update_overall_status(f"Processing 0/{self.total_videos} videos...")

            # Create ThreadPoolExecutor
            self.executor = ThreadPoolExecutor(max_workers=num_threads)

            # Create a dictionary to map futures to video paths
            future_to_video = {}

            # Submit extraction tasks
            for video_path in video_files:
                future = self.executor.submit(
                    self._extract_frames_from_video_threaded,
                    video_path, output_dir, target_width, target_height,
                    interpolation, nth_frame, prefix, preserve_aspect, use_original
                )
                future_to_video[future] = video_path

            # Monitor completion
            for future in as_completed(future_to_video):
                if self.stop_flag.is_set():
                    break

                video_path = future_to_video[future]
                self.completed_videos += 1

                # Update overall progress - FIX: Now updates correctly
                overall_progress = (self.completed_videos / self.total_videos) * 100
                self._update_overall_progress(overall_progress)
                self._update_overall_status(f"Processing {self.completed_videos}/{self.total_videos} videos...")

                try:
                    result = future.result()
                    if result['success']:
                        self.log_queue.put(
                            f"✅ {result['video_name']}: Extracted {result['saved_count']} frames"
                        )
                    else:
                        self.log_queue.put(
                            f"❌ {result['video_name']}: {result['error']}"
                        )
                except Exception as e:
                    video_name = os.path.basename(video_path)
                    self.log_queue.put(f"❌ Error processing {video_name}: {str(e)}")

            if not self.stop_flag.is_set():
                self._update_overall_status(f"Completed! Processed {self.completed_videos}/{self.total_videos} videos")
                self.log_queue.put("\n✅ Frame extraction completed successfully!")
            else:
                self._update_overall_status("Extraction stopped by user")

        except Exception as e:
            self.log_queue.put(f"\n❌ Error during extraction: {str(e)}")
            import traceback
            self.log_queue.put(traceback.format_exc())
            self._update_overall_status(f"Error: {str(e)}")
        finally:
            if self.executor:
                self.executor.shutdown(wait=True)
            # Reset button states
            self.parent.after(0, lambda: self.start_button.config(state='normal'))
            self.parent.after(0, lambda: self.analyze_button.config(state='normal'))
            self.parent.after(0, lambda: self.stop_button.config(state='disabled'))

    def _create_video_progress_bar(self, video_name):
        """Create progress bar for individual video"""
        # Remove placeholder on first video
        if hasattr(self, 'placeholder_label') and self.placeholder_label.winfo_exists():
            self.placeholder_label.destroy()

        frame = ttk.Frame(self.videos_scroll_frame)
        frame.pack(fill='x', padx=5, pady=3)

        # Video name label
        name_to_display = video_name[:50] + "..." if len(video_name) > 50 else video_name
        ttk.Label(frame, text=name_to_display, font=('TkDefaultFont', 9)).pack(anchor='w')

        # Progress bar
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill='x', pady=(2, 2))

        # Status label
        status_label = ttk.Label(frame, text="Waiting...", font=('TkDefaultFont', 8))
        status_label.pack(anchor='w')

        # Store references
        self.video_progress_vars[video_name] = progress_var
        self.video_progress_bars[video_name] = progress_bar
        self.video_status_labels[video_name] = status_label

    def _extract_frames_from_video_threaded(self, video_path, output_dir, target_width, target_height,
                                            interpolation, nth_frame, prefix, preserve_aspect, use_original):
        """Extract frames from a single video (runs in thread)"""
        video_name = Path(video_path).stem
        video_basename = os.path.basename(video_path)
        result = {
            'video_name': video_basename,
            'success': False,
            'saved_count': 0,
            'error': None
        }

        try:
            # Update status
            self._update_video_status(video_basename, "Starting...")

            # Create output directory
            if self.source_type_var.get() == "single":
                video_output_dir = output_dir
            else:
                video_output_dir = os.path.join(output_dir, video_name)
                if not os.path.exists(video_output_dir):
                    os.makedirs(video_output_dir)

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                result['error'] = "Cannot open video"
                self._update_video_status(video_basename, "Failed: Cannot open")
                return result

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Get original dimensions for aspect ratio calculation
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calculate actual output size if preserving aspect ratio
            if not use_original and preserve_aspect and target_width and target_height:
                actual_width, actual_height = self.calculate_aspect_ratio_size(
                    orig_width, orig_height, target_width, target_height
                )
                self.log_queue.put(
                    f"  {video_basename}: {orig_width}x{orig_height} → {actual_width}x{actual_height} "
                    f"(within {target_width}x{target_height})"
                )
            elif not use_original:
                actual_width, actual_height = target_width, target_height
            else:
                actual_width, actual_height = orig_width, orig_height

            frame_count = 0
            saved_count = 0

            while True:
                if self.stop_flag.is_set():
                    result['error'] = "Stopped by user"
                    break

                success, frame = cap.read()
                if not success:
                    break

                # FIX: Check if this frame should be extracted
                if frame_count % nth_frame == 0:
                    try:
                        # Resize frame only if not using original
                        if not use_original:
                            resized = cv2.resize(frame, (actual_width, actual_height),
                                                 interpolation=interpolation)
                        else:
                            resized = frame

                        # Save frame with prefix
                        filename = f"{video_name}_{prefix}_{frame_count:05d}.jpg"
                        filepath = os.path.join(video_output_dir, filename)

                        if cv2.imwrite(filepath, resized):
                            saved_count += 1
                        else:
                            self.log_queue.put(f"❌ Failed to save: {filename}")

                    except Exception as e:
                        self.log_queue.put(f"❌ Error processing frame {frame_count}: {str(e)}")

                frame_count += 1

                # Update progress every 10 frames to reduce UI updates
                if frame_count % 10 == 0:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    self._update_video_progress(video_basename, progress)
                    self._update_video_status(
                        video_basename,
                        f"Frame {frame_count}/{total_frames} (saved: {saved_count})"
                    )

            cap.release()

            result['success'] = True
            result['saved_count'] = saved_count

            # Final update
            self._update_video_progress(video_basename, 100)
            self._update_video_status(video_basename, f"✅ Completed: {saved_count} frames saved")

        except Exception as e:
            result['error'] = str(e)
            self._update_video_status(video_basename, f"❌ Error: {str(e)[:30]}")
            import traceback
            self.log_queue.put(f"Full error for {video_basename}:\n{traceback.format_exc()}")

        return result

    def _update_video_progress(self, video_name, progress):
        """Update progress bar for specific video"""
        if video_name in self.video_progress_vars:
            self.parent.after(0, lambda: self.video_progress_vars[video_name].set(progress))

    def _update_video_status(self, video_name, status):
        """Update status label for specific video"""
        if video_name in self.video_status_labels:
            self.parent.after(0, lambda: self.video_status_labels[video_name].config(text=status))

    def _update_overall_progress(self, value):
        """Update overall progress bar"""
        self.parent.after(0, lambda: self.overall_progress_var.set(value))

    def _update_overall_status(self, message):
        """Update overall status label"""
        self.parent.after(0, lambda: self.overall_status_label.config(text=message))
