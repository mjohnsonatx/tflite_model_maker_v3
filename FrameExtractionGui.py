import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from typing import Optional, Dict
import multiprocessing

from frame_extraction_logic import (
    FrameExtractionEngine,
    ExtractionSettings,
    InterpolationMethod,
    ExtractionSession,
    VideoInfo
)


class VideoAnalysisDisplay(ttk.LabelFrame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, text="Video Analysis Results", padding="10", **kwargs)

        self._info_text = tk.StringVar(value="No videos analyzed yet")

        self._label = ttk.Label(
            self,
            textvariable=self._info_text,
            font=('TkDefaultFont', 10),
            wraplength=600,
            justify='left'
        )
        self._label.pack(fill='x', expand=True)

        self._details_frame = ttk.Frame(self)
        self._details_frame.pack(fill='x', pady=(5, 0))

        self._video_listbox: Optional[tk.Listbox] = None

    def clear(self):
        self._info_text.set("No videos analyzed yet")
        for widget in self._details_frame.winfo_children():
            widget.destroy()
        self._video_listbox = None

    def show_single_video(self, info: VideoInfo, frames_to_extract: int, nth_frame: int):
        self._info_text.set(
            f"Video: {info.name}\n"
            f"Resolution: {info.resolution_str}\n"
            f"Duration: {info.duration:.1f}s at {info.fps:.1f} fps\n"
            f"Total frames: {info.frame_count}\n"
            f"Frames to extract: {frames_to_extract} (every {nth_frame} frame)"
        )

    def show_multiple_videos(
            self,
            videos: list,
            total_frames: int,
            total_to_extract: int,
            nth_frame: int
    ):
        self._info_text.set(
            f"Found {len(videos)} videos | "
            f"Total frames: {total_frames} | "
            f"To extract: {total_to_extract} (every {nth_frame} frame)"
        )

        for widget in self._details_frame.winfo_children():
            widget.destroy()

        if len(videos) <= 10:
            list_frame = ttk.Frame(self._details_frame)
            list_frame.pack(fill='x', pady=(5, 0))

            scrollbar = ttk.Scrollbar(list_frame)
            scrollbar.pack(side='right', fill='y')

            self._video_listbox = tk.Listbox(
                list_frame,
                height=min(len(videos), 5),
                yscrollcommand=scrollbar.set,
                font=('TkDefaultFont', 9)
            )
            self._video_listbox.pack(side='left', fill='x', expand=True)
            scrollbar.config(command=self._video_listbox.yview)

            for info in videos:
                frames_extract = info.frame_count // nth_frame
                self._video_listbox.insert(
                    tk.END,
                    f"{info.name} - {info.resolution_str} - {frames_extract}/{info.frame_count} frames"
                )


class ProgressDisplay(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        overall_frame = ttk.LabelFrame(self, text="Overall Progress", padding="10")
        overall_frame.pack(fill='x', pady=(0, 5))

        self._overall_progress_var = tk.DoubleVar()
        self._overall_progress_bar = ttk.Progressbar(
            overall_frame,
            variable=self._overall_progress_var,
            maximum=100
        )
        self._overall_progress_bar.pack(fill='x', pady=(0, 5))

        self._overall_status_label = ttk.Label(overall_frame, text="Ready")
        self._overall_status_label.pack()

        videos_frame = ttk.LabelFrame(self, text="Individual Videos", padding="5")
        videos_frame.pack(fill='both', expand=True)

        self._videos_canvas = tk.Canvas(videos_frame, height=150)
        scrollbar = ttk.Scrollbar(
            videos_frame,
            orient="vertical",
            command=self._videos_canvas.yview
        )

        self._scroll_frame = ttk.Frame(self._videos_canvas)
        self._scroll_frame.bind(
            "<Configure>",
            lambda e: self._videos_canvas.configure(
                scrollregion=self._videos_canvas.bbox("all")
            )
        )

        self._videos_canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        self._videos_canvas.configure(yscrollcommand=scrollbar.set)

        self._videos_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._placeholder = ttk.Label(
            self._scroll_frame,
            text="No videos processing...",
            foreground="gray"
        )
        self._placeholder.pack(pady=20)

        self._video_widgets: Dict[str, dict] = {}

    def reset(self):
        for widget in self._scroll_frame.winfo_children():
            widget.destroy()
        self._video_widgets.clear()

        self._placeholder = ttk.Label(
            self._scroll_frame,
            text="No videos processing...",
            foreground="gray"
        )
        self._placeholder.pack(pady=20)

        self._overall_progress_var.set(0)
        self._overall_status_label.config(text="Ready")

    def add_video(self, video_name: str):
        if self._placeholder and self._placeholder.winfo_exists():
            self._placeholder.destroy()
            self._placeholder = None

        frame = ttk.Frame(self._scroll_frame)
        frame.pack(fill='x', padx=5, pady=3)

        display_name = video_name[:50] + "..." if len(video_name) > 50 else video_name
        ttk.Label(frame, text=display_name, font=('TkDefaultFont', 9)).pack(anchor='w')

        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill='x', pady=(2, 2))

        status_label = ttk.Label(frame, text="Waiting...", font=('TkDefaultFont', 8))
        status_label.pack(anchor='w')

        self._video_widgets[video_name] = {
            'frame': frame,
            'progress_var': progress_var,
            'progress_bar': progress_bar,
            'status_label': status_label
        }

    def update_video_progress(self, video_name: str, progress: float, status: str):
        if video_name in self._video_widgets:
            widgets = self._video_widgets[video_name]
            widgets['progress_var'].set(progress)
            widgets['status_label'].config(text=status)

    def update_overall_progress(self, progress: float, status: str):
        self._overall_progress_var.set(progress)
        self._overall_status_label.config(text=status)


class FrameExtractionTab:
    def __init__(self, parent, log_queue):
        self._parent = parent
        self._log_queue = log_queue
        self._app_root = os.path.dirname(os.path.abspath(__file__))

        self._engine = FrameExtractionEngine(log_callback=self._log)
        self._engine.set_progress_callbacks(
            video_progress=self._on_video_progress,
            overall_progress=self._on_overall_progress
        )

        self._pending_session: Optional[ExtractionSession] = None

        self._setup_ui()

    def _log(self, message: str):
        if self._log_queue:
            self._log_queue.put(message)

    def _setup_ui(self):
        main_frame = ttk.Frame(self._parent, padding="10")
        main_frame.pack(fill="both", expand=True)

        self._build_source_section(main_frame)
        self._build_analysis_section(main_frame)
        self._build_output_section(main_frame)
        self._build_progress_section(main_frame)
        self._build_button_section(main_frame)

        self._toggle_source_type()
        self._toggle_nth_frame_entry()

    def _build_source_section(self, parent):
        ttk.Label(
            parent,
            text="Video Source",
            font=('TkDefaultFont', 12, 'bold')
        ).pack(anchor='w', pady=(0, 5))

        source_frame = ttk.LabelFrame(parent, text="Select Source Type", padding="10")
        source_frame.pack(fill='x', pady=(0, 5))

        self._source_type_var = tk.StringVar(value="directory")
        ttk.Radiobutton(
            source_frame,
            text="Process all videos in directory",
            variable=self._source_type_var,
            value="directory",
            command=self._toggle_source_type
        ).pack(anchor='w')
        ttk.Radiobutton(
            source_frame,
            text="Process single video file",
            variable=self._source_type_var,
            value="single",
            command=self._toggle_source_type
        ).pack(anchor='w')

        self._dir_frame = ttk.Frame(parent)
        dir_inner = ttk.Frame(self._dir_frame)
        dir_inner.pack(fill='x')

        ttk.Label(dir_inner, text="Video Directory:").pack(side='left', padx=(0, 10))
        self._video_dir_var = tk.StringVar()
        ttk.Entry(dir_inner, textvariable=self._video_dir_var, width=50).pack(
            side='left', fill='x', expand=True
        )
        ttk.Button(dir_inner, text="Browse", command=self._browse_video_dir).pack(
            side='left', padx=(5, 0)
        )

        self._file_frame = ttk.Frame(parent)
        file_inner = ttk.Frame(self._file_frame)
        file_inner.pack(fill='x')

        ttk.Label(file_inner, text="Video File:").pack(side='left', padx=(0, 10))
        self._video_file_var = tk.StringVar()
        ttk.Entry(file_inner, textvariable=self._video_file_var, width=50).pack(
            side='left', fill='x', expand=True
        )
        ttk.Button(file_inner, text="Browse", command=self._browse_video_file).pack(
            side='left', padx=(5, 0)
        )

    def _build_analysis_section(self, parent):
        self._analysis_display = VideoAnalysisDisplay(parent)
        self._analysis_display.pack(fill='x', pady=(5, 10))

    def _build_output_section(self, parent):
        ttk.Label(
            parent,
            text="Output Settings",
            font=('TkDefaultFont', 12, 'bold')
        ).pack(anchor='w', pady=(10, 5))

        settings_frame = ttk.LabelFrame(
            parent,
            text="Frame Extraction Settings",
            padding="10"
        )
        settings_frame.pack(fill='x', pady=(0, 5))

        output_dir_frame = ttk.Frame(settings_frame)
        output_dir_frame.pack(fill='x', pady=(0, 5))

        ttk.Label(output_dir_frame, text="Output Directory:").pack(side='left', padx=(0, 10))
        self._output_dir_var = tk.StringVar()
        ttk.Entry(output_dir_frame, textvariable=self._output_dir_var, width=40).pack(
            side='left', fill='x', expand=True
        )
        ttk.Button(output_dir_frame, text="Browse", command=self._browse_output_dir).pack(
            side='left', padx=(5, 0)
        )

        self._create_dir_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            settings_frame,
            text="Create directory if it doesn't exist",
            variable=self._create_dir_var
        ).pack(anchor='w', pady=(5, 0))

        settings_columns = ttk.Frame(settings_frame)
        settings_columns.pack(fill='x', pady=(10, 0))

        left_col = ttk.Frame(settings_columns)
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))

        prefix_frame = ttk.Frame(left_col)
        prefix_frame.pack(fill='x', pady=(0, 5))
        ttk.Label(prefix_frame, text="Filename Prefix:").pack(side='left', padx=(0, 10))
        self._prefix_var = tk.StringVar(value="frame")
        ttk.Entry(prefix_frame, textvariable=self._prefix_var, width=15).pack(side='left')

        thread_frame = ttk.Frame(left_col)
        thread_frame.pack(fill='x', pady=(5, 0))
        ttk.Label(thread_frame, text="Processing Threads:").pack(side='left', padx=(0, 10))

        max_threads = multiprocessing.cpu_count()
        self._num_threads_var = tk.StringVar(value=str(min(4, max_threads)))
        self._thread_spinbox = ttk.Spinbox(
            thread_frame,
            from_=1,
            to=max_threads,
            textvariable=self._num_threads_var,
            width=10,
            validate='key',
            validatecommand=(self._parent.register(self._validate_thread_count), '%P')
        )
        self._thread_spinbox.pack(side='left')

        gpu_frame = ttk.Frame(left_col)
        gpu_frame.pack(fill='x', pady=(5, 0))
        self._use_gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            gpu_frame,
            text="Use GPU acceleration (if available)",
            variable=self._use_gpu_var
        ).pack(anchor='w')

        quality_frame = ttk.Frame(left_col)
        quality_frame.pack(fill='x', pady=(5, 0))
        ttk.Label(quality_frame, text="JPEG Quality:").pack(side='left', padx=(0, 10))
        self._jpeg_quality_var = tk.StringVar(value="95")
        quality_spinbox = ttk.Spinbox(
            quality_frame,
            from_=1,
            to=100,
            textvariable=self._jpeg_quality_var,
            width=10
        )
        quality_spinbox.pack(side='left')

        right_col = ttk.Frame(settings_columns)
        right_col.pack(side='left', fill='both', expand=True)

        res_frame = ttk.Frame(right_col)
        res_frame.pack(fill='x', pady=(0, 5))
        ttk.Label(res_frame, text="Output Resolution:").pack(side='left', padx=(0, 10))

        self._resolution_var = tk.StringVar(value="original")
        resolution_combo = ttk.Combobox(
            res_frame,
            textvariable=self._resolution_var,
            values=[
                "original", "640x480", "800x600", "1024x768",
                "1280x720", "1920x1080", "Custom"
            ],
            width=15,
            state="readonly"
        )
        resolution_combo.pack(side='left')
        resolution_combo.bind('<<ComboboxSelected>>', self._on_resolution_change)

        self._custom_res_frame = ttk.Frame(right_col)
        ttk.Label(self._custom_res_frame, text="  Width:").pack(side='left', padx=(0, 5))
        self._custom_width_var = tk.StringVar(value="800")
        ttk.Entry(
            self._custom_res_frame,
            textvariable=self._custom_width_var,
            width=8
        ).pack(side='left', padx=(0, 10))
        ttk.Label(self._custom_res_frame, text="Height:").pack(side='left', padx=(0, 5))
        self._custom_height_var = tk.StringVar(value="600")
        ttk.Entry(
            self._custom_res_frame,
            textvariable=self._custom_height_var,
            width=8
        ).pack(side='left')

        self._preserve_aspect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            right_col,
            text="Preserve aspect ratio (fit within bounds)",
            variable=self._preserve_aspect_var
        ).pack(anchor='w', pady=(5, 0))

        interp_frame = ttk.Frame(settings_frame)
        interp_frame.pack(fill='x', pady=(10, 0))

        ttk.Label(interp_frame, text="Interpolation Method:").pack(side='left', padx=(0, 10))
        self._interpolation_var = tk.StringVar(value="INTER_LINEAR")
        ttk.Combobox(
            interp_frame,
            textvariable=self._interpolation_var,
            values=[m.name for m in InterpolationMethod],
            width=20,
            state="readonly"
        ).pack(side='left')
        ttk.Button(
            interp_frame,
            text="?",
            width=3,
            command=self._show_interpolation_info
        ).pack(side='left', padx=(5, 0))

        extraction_frame = ttk.LabelFrame(
            settings_frame,
            text="Frame Selection",
            padding="5"
        )
        extraction_frame.pack(fill='x', pady=(10, 0))

        self._extract_all_var = tk.BooleanVar(value=True)
        ttk.Radiobutton(
            extraction_frame,
            text="Extract all frames",
            variable=self._extract_all_var,
            value=True,
            command=self._toggle_nth_frame_entry
        ).pack(anchor='w')

        nth_frame_container = ttk.Frame(extraction_frame)
        nth_frame_container.pack(anchor='w')

        ttk.Radiobutton(
            nth_frame_container,
            text="Extract every",
            variable=self._extract_all_var,
            value=False,
            command=self._toggle_nth_frame_entry
        ).pack(side='left')

        self._nth_frame_var = tk.StringVar(value="5")
        self._nth_frame_entry = ttk.Entry(
            nth_frame_container,
            textvariable=self._nth_frame_var,
            width=5
        )
        self._nth_frame_entry.pack(side='left', padx=(5, 5))
        ttk.Label(nth_frame_container, text="frames").pack(side='left')

    def _build_progress_section(self, parent):
        ttk.Label(
            parent,
            text="Progress",
            font=('TkDefaultFont', 12, 'bold')
        ).pack(anchor='w', pady=(10, 5))

        self._progress_display = ProgressDisplay(parent)
        self._progress_display.pack(fill='both', expand=True, pady=(0, 5))

    def _build_button_section(self, parent):
        button_frame = ttk.Frame(parent)
        button_frame.pack(pady=(10, 0))

        self._analyze_button = ttk.Button(
            button_frame,
            text="Analyze Videos",
            command=self._analyze_videos
        )
        self._analyze_button.pack(side='left', padx=5)

        self._start_button = ttk.Button(
            button_frame,
            text="Start Extraction",
            command=self._start_extraction
        )
        self._start_button.pack(side='left', padx=5)

        self._stop_button = ttk.Button(
            button_frame,
            text="Stop",
            command=self._stop_extraction,
            state='disabled'
        )
        self._stop_button.pack(side='left', padx=5)

    def _validate_thread_count(self, value: str) -> bool:
        if value == "":
            return True
        try:
            num = int(value)
            max_threads = multiprocessing.cpu_count()
            return 1 <= num <= max_threads
        except ValueError:
            return False

    def _toggle_source_type(self):
        if self._source_type_var.get() == "directory":
            self._file_frame.pack_forget()
            self._dir_frame.pack(fill='x', pady=(5, 0), before=self._analysis_display)
        else:
            self._dir_frame.pack_forget()
            self._file_frame.pack(fill='x', pady=(5, 0), before=self._analysis_display)

    def _toggle_nth_frame_entry(self):
        if self._extract_all_var.get():
            self._nth_frame_entry.config(state='disabled')
        else:
            self._nth_frame_entry.config(state='normal')

    def _browse_video_dir(self):
        directory = filedialog.askdirectory(
            title="Select Video Directory",
            initialdir=self._app_root
        )
        if directory:
            self._video_dir_var.set(directory)

    def _browse_video_file(self):
        filename = filedialog.askopenfilename(
            title="Select Video File",
            initialdir=self._app_root,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self._video_file_var.set(filename)

    def _browse_output_dir(self):
        directory = filedialog.askdirectory(
            title="Select Output Directory",
            initialdir=self._app_root
        )
        if directory:
            self._output_dir_var.set(directory)

    def _on_resolution_change(self, event=None):
        if self._resolution_var.get() == "Custom":
            self._custom_res_frame.pack(fill='x', pady=(5, 0))
        else:
            self._custom_res_frame.pack_forget()

    def _show_interpolation_info(self):
        info_window = tk.Toplevel(self._parent)
        info_window.title("Interpolation Methods")
        info_window.geometry("600x400")

        info_text = scrolledtext.ScrolledText(
            info_window,
            wrap=tk.WORD,
            padx=10,
            pady=10
        )
        info_text.pack(fill='both', expand=True)

        interpolation_info = """
INTERPOLATION METHODS FOR IMAGE RESIZING:

INTER_NEAREST (Nearest Neighbor)
  - Fastest method, lowest quality
  - Best for: Pixel art, images with hard edges
  - Use when: Speed is critical, quality less important

INTER_LINEAR (Bilinear) - RECOMMENDED
  - Good balance of speed and quality
  - Best for: General purpose, most video frames
  - Use when: Standard frame extraction

INTER_CUBIC (Bicubic)
  - Higher quality, slower than linear
  - Best for: Upscaling, smooth gradients
  - Use when: Quality is important, slight upscaling

INTER_AREA
  - Best for downscaling (shrinking images)
  - Best for: Reducing image size
  - Use when: Output smaller than input

INTER_LANCZOS4 (Lanczos)
  - Highest quality, slowest
  - Best for: High-quality results, professional work
  - Use when: Maximum quality needed

For frame extraction, INTER_LINEAR or INTER_AREA (for downscaling)
are typically best choices.
"""
        info_text.insert(1.0, interpolation_info)
        info_text.config(state='disabled')

        ttk.Button(info_window, text="Close", command=info_window.destroy).pack(pady=10)

    def _get_nth_frame(self) -> int:
        if self._extract_all_var.get():
            return 1
        try:
            return max(1, int(self._nth_frame_var.get() or 1))
        except ValueError:
            return 1

    def _get_video_paths(self) -> list:
        if self._source_type_var.get() == "directory":
            video_dir = self._video_dir_var.get()
            if not video_dir:
                return []
            return self._engine.find_videos_in_directory(video_dir)
        else:
            video_file = self._video_file_var.get()
            return [video_file] if video_file else []

    def _analyze_videos(self):
        video_paths = self._get_video_paths()

        if not video_paths:
            source = "directory" if self._source_type_var.get() == "directory" else "file"
            messagebox.showwarning("Warning", f"Please select a video {source} first.")
            return

        nth_frame = self._get_nth_frame()
        videos, total_frames, total_to_extract = self._engine.analyze_videos(
            video_paths, nth_frame
        )

        if not videos:
            self._log("No valid_FINAL_CLEAN_DONISH video files found.")
            self._analysis_display.clear()
            return

        if len(videos) == 1:
            self._analysis_display.show_single_video(
                videos[0], total_to_extract, nth_frame
            )
            self._log(f"Analyzed: {videos[0].name} - {videos[0].summary()}")
        else:
            self._analysis_display.show_multiple_videos(
                videos, total_frames, total_to_extract, nth_frame
            )
            self._log(f"Analyzed {len(videos)} videos, {total_to_extract} frames to extract")

    def _build_extraction_settings(self) -> ExtractionSettings:
        use_original = self._resolution_var.get() == "original"
        target_width = None
        target_height = None

        if not use_original:
            if self._resolution_var.get() == "Custom":
                target_width = int(self._custom_width_var.get())
                target_height = int(self._custom_height_var.get())
            else:
                width, height = self._resolution_var.get().split('x')
                target_width = int(width)
                target_height = int(height)

        thread_count = int(self._num_threads_var.get() or 4)
        max_threads = multiprocessing.cpu_count()
        thread_count = max(1, min(thread_count, max_threads))

        jpeg_quality = int(self._jpeg_quality_var.get() or 95)
        jpeg_quality = max(1, min(100, jpeg_quality))

        return ExtractionSettings(
            output_dir=self._output_dir_var.get(),
            prefix=self._prefix_var.get() or "frame",
            num_threads=thread_count,
            target_width=target_width,
            target_height=target_height,
            preserve_aspect=self._preserve_aspect_var.get(),
            use_original_resolution=use_original,
            interpolation=InterpolationMethod[self._interpolation_var.get()],
            nth_frame=self._get_nth_frame(),
            create_subdirs=self._source_type_var.get() == "directory",
            use_gpu=self._use_gpu_var.get(),
            jpeg_quality=jpeg_quality
        )

    def _start_extraction(self):
        if not self._output_dir_var.get():
            messagebox.showwarning("Warning", "Please select an output directory.")
            return

        video_paths = self._get_video_paths()
        if not video_paths:
            source = "directory" if self._source_type_var.get() == "directory" else "file"
            messagebox.showwarning("Warning", f"Please select a video {source}.")
            return

        output_dir = self._output_dir_var.get()
        if self._create_dir_var.get() and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                self._log(f"Created output directory: {output_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory: {e}")
                return

        self._start_button.config(state='disabled')
        self._analyze_button.config(state='disabled')
        self._stop_button.config(state='normal')

        self._progress_display.reset()

        for video_path in video_paths:
            self._progress_display.add_video(os.path.basename(video_path))

        settings = self._build_extraction_settings()

        self._log(f"Starting extraction: {len(video_paths)} videos, {settings.num_threads} threads")
        self._engine.start_extraction(video_paths, settings)

    def _stop_extraction(self):
        self._log("Stopping extraction...")
        session = self._engine.stop_extraction()
        self._pending_session = session

        self._reset_buttons()

        if session and (session.created_files or session.created_directories):
            self._prompt_cleanup(session)

    def _prompt_cleanup(self, session: ExtractionSession):
        file_count = len(session.created_files)
        dir_count = len(session.created_directories)

        message = (
            f"Extraction was cancelled.\n\n"
            f"Created during this session:\n"
            f"  - {file_count} files\n"
            f"  - {dir_count} directories\n\n"
            f"Would you like to delete these files and directories?"
        )

        if messagebox.askyesno("Cleanup", message):
            files_removed, dirs_removed = self._engine.cleanup_session(session)
            self._log(f"Cleanup: removed {files_removed} files and {dirs_removed} directories")
            messagebox.showinfo(
                "Cleanup Complete",
                f"Removed {files_removed} files and {dirs_removed} directories."
            )
        else:
            self._log("Cleanup skipped by user")

        self._pending_session = None

    def _reset_buttons(self):
        self._parent.after(0, lambda: self._start_button.config(state='normal'))
        self._parent.after(0, lambda: self._analyze_button.config(state='normal'))
        self._parent.after(0, lambda: self._stop_button.config(state='disabled'))

    def _on_video_progress(self, video_name: str, progress: float, status: str):
        self._parent.after(
            0,
            lambda: self._progress_display.update_video_progress(video_name, progress, status)
        )

    def _on_overall_progress(self, progress: float, status: str):
        self._parent.after(
            0,
            lambda: self._progress_display.update_overall_progress(progress, status)
        )

        if progress >= 100 or "Error" in status or "Done" in status:
            self._reset_buttons()
