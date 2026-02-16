
import os
import shutil
import time
import subprocess
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Callable, Dict
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import cv2


class InterpolationMethod(Enum):
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_AREA = cv2.INTER_AREA
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4


@dataclass
class VideoInfo:
    path: str
    name: str
    frame_count: int
    width: int
    height: int
    fps: float
    duration: float
    rotation: int = 0

    @property
    def resolution_str(self) -> str:
        return f"{self.width}x{self.height}"

    def summary(self) -> str:
        rotation_note = f", rotation: {self.rotation} deg" if self.rotation else ""
        return (
            f"{self.frame_count} frames, {self.resolution_str}, "
            f"{self.fps:.1f} fps, {self.duration:.1f}s{rotation_note}"
        )


@dataclass
class ExtractionSettings:
    output_dir: str
    prefix: str = "frame"
    num_threads: int = 4
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    preserve_aspect: bool = True
    use_original_resolution: bool = True
    interpolation: InterpolationMethod = InterpolationMethod.INTER_LINEAR
    nth_frame: int = 1
    create_subdirs: bool = True
    use_gpu: bool = True
    jpeg_quality: int = 95


@dataclass
class ExtractionResult:
    video_name: str
    success: bool
    saved_count: int
    error: Optional[str] = None
    output_directory: Optional[str] = None
    elapsed_time: float = 0.0
    frames_per_second: float = 0.0


@dataclass
class ExtractionSession:
    videos: List[str] = field(default_factory=list)
    created_directories: List[str] = field(default_factory=list)
    created_files: List[str] = field(default_factory=list)
    completed: bool = False
    cancelled: bool = False


@dataclass
class TimingStats:
    read_time: float = 0.0
    resize_time: float = 0.0
    encode_time: float = 0.0
    write_time: float = 0.0
    total_time: float = 0.0
    frame_count: int = 0

    def summary(self) -> str:
        if self.frame_count == 0:
            return "No frames processed"

        avg_read = (self.read_time / self.frame_count) * 1000
        avg_resize = (self.resize_time / self.frame_count) * 1000
        avg_encode = (self.encode_time / self.frame_count) * 1000
        avg_write = (self.write_time / self.frame_count) * 1000
        fps = self.frame_count / self.total_time if self.total_time > 0 else 0

        return (
            f"  Read:   {avg_read:.2f}ms\n"
            f"  Resize: {avg_resize:.2f}ms\n"
            f"  Encode: {avg_encode:.2f}ms\n"
            f"  Write:  {avg_write:.2f}ms\n"
            f"  Total:  {self.total_time:.2f}s ({fps:.1f} fps)"
        )


def _get_video_rotation(video_path: str) -> int:
    ffprobe_candidates = [
        "ffprobe",
        r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
    ]

    ffprobe_path = None
    for candidate in ffprobe_candidates:
        try:
            result = subprocess.run(
                [candidate, "-version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                ffprobe_path = candidate
                break
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            continue

    if not ffprobe_path:
        return 0

    try:
        result = subprocess.run(
            [
                ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                video_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return 0

        data = json.loads(result.stdout)

        for stream in data.get("streams", []):
            if stream.get("codec_type") != "video":
                continue

            for side_data in stream.get("side_data_list", []):
                if side_data.get("side_data_type") == "Display Matrix":
                    rotation = abs(int(side_data.get("rotation", 0)))
                    if rotation in (90, 180, 270):
                        return rotation

            tags = stream.get("tags", {})
            rotation = abs(int(tags.get("rotation", 0)))
            if rotation in (90, 180, 270):
                return rotation

        return 0

    except Exception:
        return 0


def _parse_ffmpeg_progress(line: str) -> Optional[Dict[str, str]]:
    """
    FFmpeg stderr progress lines look like this:
      frame=  245 fps= 82 q=-0.0 Lsize=   44853kB time=00:00:08.17 bitrate=44938kbits/s speed=2.74x
    or during active encoding (no L prefix, uses = not :):
      frame=  245 fps= 82 q=-0.0 size=   44853kB time=00:00:08.17 bitrate=44938kbits/s speed=2.74x

    We only care about frame count and time for progress reporting.
    Returns a dict with 'frame' and 'time_seconds' keys, or None if the line
    doesn't match the expected format.
    """
    frame_match = re.search(r'frame=\s*(\d+)', line)
    time_match = re.search(r'time=(\d+):(\d+):(\d+\.\d+)', line)

    if not frame_match or not time_match:
        return None

    hours = int(time_match.group(1))
    minutes = int(time_match.group(2))
    seconds = float(time_match.group(3))
    time_seconds = hours * 3600 + minutes * 60 + seconds

    return {
        'frame': int(frame_match.group(1)),
        'time_seconds': time_seconds
    }


class FFmpegGPUDecoder:
    _ffmpeg_path: Optional[str] = None
    _gpu_decode_available: bool = False
    _checked: bool = False

    @classmethod
    def _find_ffmpeg(cls) -> Optional[str]:
        candidates = [
            "ffmpeg",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]

        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "-version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return candidate
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue

        return None

    @classmethod
    def check_availability(cls, log: Callable[[str], None]) -> Tuple[bool, str]:
        if cls._checked:
            return cls._gpu_decode_available, cls._ffmpeg_path or "not found"

        cls._checked = True
        cls._ffmpeg_path = cls._find_ffmpeg()

        if not cls._ffmpeg_path:
            log("FFmpeg not found. GPU decoding unavailable.")
            log("  Install from: https://www.gyan.dev/ffmpeg/builds/")
            log("  Download the 'full' build and add its bin folder to PATH.")
            return False, "FFmpeg not found"

        try:
            result = subprocess.run(
                [cls._ffmpeg_path, "-decoders"],
                capture_output=True,
                text=True,
                timeout=5
            )
            has_h264_cuvid = "h264_cuvid" in result.stdout
            has_hevc_cuvid = "hevc_cuvid" in result.stdout

            if has_h264_cuvid or has_hevc_cuvid:
                cls._gpu_decode_available = True
                decoders = []
                if has_h264_cuvid:
                    decoders.append("h264_cuvid")
                if has_hevc_cuvid:
                    decoders.append("hevc_cuvid")
                log(f"FFmpeg GPU decoders available: {', '.join(decoders)}")
                return True, cls._ffmpeg_path
            else:
                log("FFmpeg found but no CUDA decoders.")
                log("  You need the 'full' build from https://www.gyan.dev/ffmpeg/builds/")
                return False, "No CUDA decoders"

        except Exception as e:
            log(f"FFmpeg decoder check failed: {e}")
            return False, str(e)

    @classmethod
    def _build_filter_graph(
            cls,
            nth_frame: int,
            target_width: Optional[int],
            target_height: Optional[int],
            rotation: int
    ) -> Tuple[str, bool]:
        transpose_map = {
            90: "transpose=1",
            180: "transpose=1,transpose=1",
            270: "transpose=3",
        }

        needs_cpu_filters = (nth_frame > 1) or (rotation in transpose_map)
        needs_scaling = target_width is not None and target_height is not None

        filters = []

        if needs_cpu_filters:
            if nth_frame > 1:
                filters.append(f"select='mod(n\\,{nth_frame}).eq(0)'")

            if rotation in transpose_map:
                filters.append(transpose_map[rotation])

            if needs_scaling:
                filters.append(f"scale={target_width}:{target_height}")

            filters.append("format=bgr24")
            return ",".join(filters), True

        else:
            if needs_scaling:
                filters.append(f"scale_cuda={target_width}:{target_height}")

            filters.append("hwdownload")
            filters.append("format=bgr24")
            return ",".join(filters), False

    @classmethod
    def extract_frames_gpu(
            cls,
            video_path: str,
            output_dir: str,
            prefix: str,
            video_name: str,
            nth_frame: int,
            target_width: Optional[int],
            target_height: Optional[int],
            jpeg_quality: int,
            rotation: int,
            total_frames: int,
            duration: float,
            stop_flag: threading.Event,
            progress_callback: Optional[Callable[[str, float, str], None]],
            video_basename: str,
            log: Callable[[str], None]
    ) -> Tuple[int, float]:
        if not cls._ffmpeg_path or not cls._gpu_decode_available:
            raise RuntimeError("FFmpeg GPU decoding not available")

        start_time = time.perf_counter()

        output_pattern = os.path.join(
            output_dir,
            f"{video_name}_{prefix}_%05d.jpg"
        )

        qscale = max(1, min(31, (100 - jpeg_quality) // 3))
        filter_graph, needs_cpu_input = cls._build_filter_graph(
            nth_frame, target_width, target_height, rotation
        )

        cmd = [cls._ffmpeg_path]

        if needs_cpu_input:
            cmd.extend(["-i", video_path])
        else:
            cmd.extend(["-hwaccel", "cuda"])
            cmd.extend(["-hwaccel_output_format", "cuda"])
            cmd.extend(["-i", video_path])

        cmd.extend(["-vf", filter_graph])

        if nth_frame > 1:
            cmd.extend(["-vsync", "0"])

        cmd.extend(["-qscale:v", str(qscale)])
        cmd.extend(["-y"])
        cmd.append(output_pattern)

        log(f"  FFmpeg cmd: {' '.join(cmd)}")
        log(f"  Filter graph: {filter_graph}")
        if rotation:
            log(f"  Applying rotation correction: {rotation} degrees")

        # The number of frames ffmpeg will actually write after select filtering.
        # This is what we use to calculate progress percentage from the output
        # frame count, not from the input frame count which includes skipped frames.
        expected_output_frames = total_frames // nth_frame if nth_frame > 1 else total_frames

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            last_progress_update = 0.0
            stderr_tail = []  # keeps last N lines for error reporting

            # stderr is read line by line in this thread. FFmpeg flushes a new
            # progress line roughly once per second (or on each frame for slow
            # encodes). Each line either contains progress info we can parse, or
            # is stream/codec metadata we ignore.
            for line in process.stderr:
                stderr_tail.append(line)
                if len(stderr_tail) > 50:
                    stderr_tail.pop(0)

                if stop_flag.is_set():
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    elapsed = time.perf_counter() - start_time
                    saved = cls._count_output_files(output_dir, video_name, prefix)
                    return saved, elapsed

                progress_info = _parse_ffmpeg_progress(line)
                if not progress_info:
                    continue

                # Throttle UI updates to ~10 per second to avoid hammering tkinter
                now = time.perf_counter()
                if now - last_progress_update < 0.1:
                    continue
                last_progress_update = now

                frames_written = progress_info['frame']
                current_time = progress_info['time_seconds']

                # When using select, ffmpeg reports the INPUT frame counter, not
                # output. We calculate output frames from elapsed time ratio
                # against total duration instead.
                if nth_frame > 1 and duration > 0:
                    progress_pct = min(99.9, (current_time / duration) * 100)
                    estimated_saved = int(frames_written / nth_frame)
                else:
                    progress_pct = min(99.9, (frames_written / expected_output_frames) * 100) if expected_output_frames > 0 else 0
                    estimated_saved = frames_written

                elapsed = now - start_time
                current_fps = estimated_saved / elapsed if elapsed > 0 else 0

                if progress_callback:
                    progress_callback(
                        video_basename,
                        progress_pct,
                        f"Frame {estimated_saved}/{expected_output_frames} | {current_fps:.1f} fps"
                    )

            # Process has finished, wait for clean exit
            process.wait()
            elapsed = time.perf_counter() - start_time

            if process.returncode != 0:
                log(f"  FFmpeg stderr: {''.join(stderr_tail[-20:])}")
                raise RuntimeError(f"FFmpeg exited with code {process.returncode}")

            saved = cls._count_output_files(output_dir, video_name, prefix)

            if saved == 0:
                log(f"  Warning: 0 frames produced. FFmpeg output:")
                log(f"  {''.join(stderr_tail[-20:])}")

            return saved, elapsed

        except FileNotFoundError:
            raise RuntimeError(f"FFmpeg not found at: {cls._ffmpeg_path}")

    @classmethod
    def _count_output_files(cls, output_dir: str, video_name: str, prefix: str) -> int:
        pattern = f"{video_name}_{prefix}_"
        count = 0
        for f in os.listdir(output_dir):
            if f.startswith(pattern) and f.endswith(".jpg"):
                count += 1
        return count


class FrameExtractionEngine:
    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

    def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
        self._log = log_callback or (lambda x: None)
        self._stop_flag = threading.Event()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._current_session: Optional[ExtractionSession] = None
        self._progress_callback: Optional[Callable[[str, float, str], None]] = None
        self._overall_progress_callback: Optional[Callable[[float, str], None]] = None

        self._ffmpeg_available, self._ffmpeg_path = FFmpegGPUDecoder.check_availability(self._log)

    def set_progress_callbacks(
        self,
        video_progress: Callable[[str, float, str], None],
        overall_progress: Callable[[float, str], None]
    ):
        self._progress_callback = video_progress
        self._overall_progress_callback = overall_progress

    def analyze_video(self, video_path: str) -> Optional[VideoInfo]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            rotation = _get_video_rotation(video_path)

            return VideoInfo(
                path=video_path,
                name=os.path.basename(video_path),
                frame_count=frame_count,
                width=width,
                height=height,
                fps=fps,
                duration=duration,
                rotation=rotation
            )
        finally:
            cap.release()

    def find_videos_in_directory(self, directory: str) -> List[str]:
        videos = []
        for file in os.listdir(directory):
            if Path(file).suffix.lower() in self.VIDEO_EXTENSIONS:
                videos.append(os.path.join(directory, file))
        return videos

    def analyze_videos(
        self,
        video_paths: List[str],
        nth_frame: int = 1
    ) -> Tuple[List[VideoInfo], int, int]:
        results = []
        total_frames = 0
        total_to_extract = 0

        for path in video_paths:
            info = self.analyze_video(path)
            if info:
                results.append(info)
                total_frames += info.frame_count
                total_to_extract += info.frame_count // nth_frame

        return results, total_frames, total_to_extract

    def calculate_aspect_ratio_size(
        self,
        orig_width: int,
        orig_height: int,
        target_width: int,
        target_height: int
    ) -> Tuple[int, int]:
        width_scale = target_width / orig_width
        height_scale = target_height / orig_height
        scale = min(width_scale, height_scale)
        return int(orig_width * scale), int(orig_height * scale)

    def start_extraction(
        self,
        video_paths: List[str],
        settings: ExtractionSettings
    ) -> None:
        self._stop_flag.clear()
        self._current_session = ExtractionSession(videos=video_paths.copy())

        if settings.create_subdirs and not os.path.exists(settings.output_dir):
            os.makedirs(settings.output_dir)
            self._current_session.created_directories.append(settings.output_dir)

        thread = threading.Thread(
            target=self._run_extraction,
            args=(video_paths, settings),
            daemon=True
        )
        thread.start()

    def stop_extraction(self) -> ExtractionSession:
        self._stop_flag.set()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)

        session = self._current_session
        if session:
            session.cancelled = True
        return session

    def cleanup_session(self, session: ExtractionSession) -> Tuple[int, int]:
        files_removed = 0
        dirs_removed = 0

        for file_path in session.created_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    files_removed += 1
            except Exception as e:
                self._log(f"Failed to remove file {file_path}: {e}")

        for dir_path in reversed(session.created_directories):
            try:
                if os.path.exists(dir_path):
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                    else:
                        shutil.rmtree(dir_path)
                    dirs_removed += 1
            except Exception as e:
                self._log(f"Failed to remove directory {dir_path}: {e}")

        return files_removed, dirs_removed

    def _run_extraction(
        self,
        video_paths: List[str],
        settings: ExtractionSettings
    ) -> None:
        total_videos = len(video_paths)
        completed_videos = 0
        safe_threads = min(settings.num_threads, len(video_paths), os.cpu_count() or 4)

        use_gpu = settings.use_gpu and self._ffmpeg_available

        if use_gpu:
            safe_threads = 1

        overall_start = time.perf_counter()

        self._log("=" * 60)
        self._log("EXTRACTION STARTED")
        self._log("=" * 60)
        self._log(f"Videos:     {total_videos}")
        self._log(f"Threads:    {safe_threads}")
        self._log(f"Decode:     {'FFmpeg GPU (CUDA)' if use_gpu else 'OpenCV CPU'}")
        self._log(f"Resolution: {'Original' if settings.use_original_resolution else f'{settings.target_width}x{settings.target_height}'}")
        self._log(f"Every:      {settings.nth_frame} frame(s)")
        self._log(f"Quality:    {settings.jpeg_quality}")
        self._log("-" * 60)

        try:
            self._executor = ThreadPoolExecutor(max_workers=safe_threads)
            future_to_video: Dict[Future, str] = {}

            for video_path in video_paths:
                if self._stop_flag.is_set():
                    break

                future = self._executor.submit(
                    self._extract_single_video,
                    video_path,
                    settings,
                    use_gpu
                )
                future_to_video[future] = video_path

            total_frames_extracted = 0

            for future in as_completed(future_to_video):
                if self._stop_flag.is_set():
                    break

                video_path = future_to_video[future]
                completed_videos += 1

                overall_progress = (completed_videos / total_videos) * 100
                if self._overall_progress_callback:
                    self._overall_progress_callback(
                        overall_progress,
                        f"Processing {completed_videos}/{total_videos} videos..."
                    )

                try:
                    result = future.result()
                    total_frames_extracted += result.saved_count
                    status = "OK" if result.success else f"FAILED: {result.error}"
                    self._log(
                        f"[{completed_videos}/{total_videos}] {result.video_name}: "
                        f"{status} - {result.saved_count} frames in {result.elapsed_time:.2f}s "
                        f"({result.frames_per_second:.1f} fps)"
                    )
                except Exception as e:
                    self._log(f"[{completed_videos}/{total_videos}] {os.path.basename(video_path)}: {e}")

            if self._current_session:
                self._current_session.completed = not self._stop_flag.is_set()

            overall_elapsed = time.perf_counter() - overall_start
            overall_fps = total_frames_extracted / overall_elapsed if overall_elapsed > 0 else 0

            self._log("-" * 60)
            self._log("SUMMARY")
            self._log("-" * 60)
            self._log(f"Total time:      {overall_elapsed:.2f}s")
            self._log(f"Videos:          {completed_videos}/{total_videos}")
            self._log(f"Frames:          {total_frames_extracted}")
            self._log(f"Throughput:      {overall_fps:.1f} frames/sec")
            if completed_videos > 0:
                self._log(f"Avg per video:   {overall_elapsed / completed_videos:.2f}s")
            self._log("=" * 60)

            if not self._stop_flag.is_set() and self._overall_progress_callback:
                self._overall_progress_callback(
                    100,
                    f"Done! {total_frames_extracted} frames in {overall_elapsed:.1f}s ({overall_fps:.1f} fps)"
                )

        except Exception as e:
            self._log(f"Extraction error: {e}")
            if self._overall_progress_callback:
                self._overall_progress_callback(0, f"Error: {e}")
        finally:
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

    def _extract_single_video(
        self,
        video_path: str,
        settings: ExtractionSettings,
        use_gpu: bool
    ) -> ExtractionResult:
        video_name = Path(video_path).stem
        video_basename = os.path.basename(video_path)

        result = ExtractionResult(
            video_name=video_basename,
            success=False,
            saved_count=0
        )

        try:
            if self._progress_callback:
                path_label = "GPU" if use_gpu else "CPU"
                self._progress_callback(video_basename, 0, f"Starting ({path_label})...")

            if settings.create_subdirs:
                video_output_dir = os.path.join(settings.output_dir, video_name)
            else:
                video_output_dir = settings.output_dir

            if not os.path.exists(video_output_dir):
                os.makedirs(video_output_dir)
                if self._current_session:
                    self._current_session.created_directories.append(video_output_dir)

            result.output_directory = video_output_dir

            info = self.analyze_video(video_path)
            rotation = info.rotation if info else 0
            total_frames = info.frame_count if info else 0
            duration = info.duration if info else 0.0

            target_w, target_h = None, None
            if not settings.use_original_resolution and info:
                if settings.preserve_aspect and settings.target_width and settings.target_height:
                    effective_w = info.height if rotation in (90, 270) else info.width
                    effective_h = info.width if rotation in (90, 270) else info.height
                    target_w, target_h = self.calculate_aspect_ratio_size(
                        effective_w, effective_h,
                        settings.target_width, settings.target_height
                    )
                else:
                    target_w = settings.target_width
                    target_h = settings.target_height

            if use_gpu:
                result = self._extract_gpu(
                    video_path, video_output_dir, settings,
                    video_name, video_basename, target_w, target_h,
                    rotation, total_frames, duration, result
                )
            else:
                result = self._extract_cpu(
                    video_path, video_output_dir, settings,
                    video_name, video_basename, target_w, target_h, rotation, result
                )

        except Exception as e:
            result.error = str(e)
            self._log(f"  {video_basename}: {e}")
            if self._progress_callback:
                self._progress_callback(video_basename, 0, f"Error: {str(e)[:40]}")

        return result

    def _extract_gpu(
        self,
        video_path: str,
        output_dir: str,
        settings: ExtractionSettings,
        video_name: str,
        video_basename: str,
        target_w: Optional[int],
        target_h: Optional[int],
        rotation: int,
        total_frames: int,
        duration: float,
        result: ExtractionResult
    ) -> ExtractionResult:
        if self._progress_callback:
            self._progress_callback(video_basename, 0, "FFmpeg GPU decoding...")

        try:
            saved, elapsed = FFmpegGPUDecoder.extract_frames_gpu(
                video_path=video_path,
                output_dir=output_dir,
                prefix=settings.prefix,
                video_name=video_name,
                nth_frame=settings.nth_frame,
                target_width=target_w,
                target_height=target_h,
                jpeg_quality=settings.jpeg_quality,
                rotation=rotation,
                total_frames=total_frames,
                duration=duration,
                stop_flag=self._stop_flag,
                progress_callback=self._progress_callback,
                video_basename=video_basename,
                log=self._log
            )

            result.success = not self._stop_flag.is_set()
            result.saved_count = saved
            result.elapsed_time = elapsed
            result.frames_per_second = saved / elapsed if elapsed > 0 else 0

            if self._current_session:
                pattern = f"{video_name}_{settings.prefix}_"
                for f in os.listdir(output_dir):
                    if f.startswith(pattern) and f.endswith(".jpg"):
                        self._current_session.created_files.append(
                            os.path.join(output_dir, f)
                        )

            if self._progress_callback:
                status = f"Done: {saved} frames ({result.frames_per_second:.1f} fps)" if result.success else "Cancelled"
                self._progress_callback(video_basename, 100, status)

        except Exception as e:
            result.error = str(e)
            self._log(f"  {video_basename} GPU failed: {e}")
            self._log(f"  Falling back to CPU")
            return self._extract_cpu(
                video_path, output_dir, settings,
                video_name, video_basename, target_w, target_h, rotation,
                ExtractionResult(video_name=video_basename, success=False, saved_count=0)
            )

        return result

    def _extract_cpu(
        self,
        video_path: str,
        output_dir: str,
        settings: ExtractionSettings,
        video_name: str,
        video_basename: str,
        target_w: Optional[int],
        target_h: Optional[int],
        rotation: int,
        result: ExtractionResult
    ) -> ExtractionResult:
        stats = TimingStats()
        video_start = time.perf_counter()

        opencv_rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            result.error = "Cannot open video"
            return result

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            needs_resize = target_w is not None and target_h is not None
            needs_rotation = rotation in opencv_rotation_map
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]

            frame_count = 0
            saved_count = 0

            while not self._stop_flag.is_set():
                t0 = time.perf_counter()
                success, frame = cap.read()
                stats.read_time += time.perf_counter() - t0

                if not success:
                    break

                if frame_count % settings.nth_frame == 0:
                    t0 = time.perf_counter()
                    if needs_rotation:
                        frame = cv2.rotate(frame, opencv_rotation_map[rotation])
                    if needs_resize:
                        frame = cv2.resize(
                            frame,
                            (target_w, target_h),
                            interpolation=settings.interpolation.value
                        )
                    stats.resize_time += time.perf_counter() - t0

                    filename = f"{video_name}_{settings.prefix}_{frame_count:05d}.jpg"
                    filepath = os.path.join(output_dir, filename)

                    t0 = time.perf_counter()
                    success_encode, encoded = cv2.imencode('.jpg', frame, encode_params)
                    stats.encode_time += time.perf_counter() - t0

                    if success_encode:
                        t0 = time.perf_counter()
                        with open(filepath, 'wb') as f:
                            f.write(encoded.tobytes())
                        stats.write_time += time.perf_counter() - t0

                        saved_count += 1
                        stats.frame_count += 1

                        if self._current_session:
                            self._current_session.created_files.append(filepath)

                frame_count += 1

                if frame_count % 30 == 0 and self._progress_callback:
                    progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    elapsed = time.perf_counter() - video_start
                    current_fps = saved_count / elapsed if elapsed > 0 else 0
                    self._progress_callback(
                        video_basename,
                        progress,
                        f"Frame {frame_count}/{total_frames} | {saved_count} saved | {current_fps:.1f} fps"
                    )

            result.success = not self._stop_flag.is_set()
            result.saved_count = saved_count
            result.elapsed_time = time.perf_counter() - video_start
            result.frames_per_second = saved_count / result.elapsed_time if result.elapsed_time > 0 else 0
            stats.total_time = result.elapsed_time

            if result.success and saved_count > 50:
                self._log(f"  {video_basename} CPU timing (avg/frame):")
                self._log(stats.summary())

            if self._progress_callback:
                status = f"Done: {saved_count} frames ({result.frames_per_second:.1f} fps)" if result.success else "Cancelled"
                self._progress_callback(video_basename, 100, status)

        finally:
            cap.release()

        return result

# import os
# import re
# import shutil
# import time
# import subprocess
# from pathlib import Path
# from typing import List, Optional, Tuple, Callable, Dict
# from dataclasses import dataclass, field
# from enum import Enum
# import threading
# from concurrent.futures import ThreadPoolExecutor, as_completed, Future
# import cv2
#
#
# class InterpolationMethod(Enum):
#     INTER_NEAREST = cv2.INTER_NEAREST
#     INTER_LINEAR = cv2.INTER_LINEAR
#     INTER_CUBIC = cv2.INTER_CUBIC
#     INTER_AREA = cv2.INTER_AREA
#     INTER_LANCZOS4 = cv2.INTER_LANCZOS4
#
#
# @dataclass
# class VideoInfo:
#     path: str
#     name: str
#     frame_count: int
#     width: int
#     height: int
#     fps: float
#     duration: float
#     rotation: int = 0
#
#     @property
#     def resolution_str(self) -> str:
#         return f"{self.width}x{self.height}"
#
#     def summary(self) -> str:
#         rotation_note = f", rotation: {self.rotation} deg" if self.rotation else ""
#         return (
#             f"{self.frame_count} frames, {self.resolution_str}, "
#             f"{self.fps:.1f} fps, {self.duration:.1f}s{rotation_note}"
#         )
#
#
# @dataclass
# class ExtractionSettings:
#     output_dir: str
#     prefix: str = "frame"
#     num_threads: int = 4
#     target_width: Optional[int] = None
#     target_height: Optional[int] = None
#     preserve_aspect: bool = True
#     use_original_resolution: bool = True
#     interpolation: InterpolationMethod = InterpolationMethod.INTER_LINEAR
#     nth_frame: int = 1
#     create_subdirs: bool = True
#     use_gpu: bool = True
#     jpeg_quality: int = 95
#
#
# @dataclass
# class ExtractionResult:
#     video_name: str
#     success: bool
#     saved_count: int
#     error: Optional[str] = None
#     output_directory: Optional[str] = None
#     elapsed_time: float = 0.0
#     frames_per_second: float = 0.0
#
#
# @dataclass
# class ExtractionSession:
#     videos: List[str] = field(default_factory=list)
#     created_directories: List[str] = field(default_factory=list)
#     created_files: List[str] = field(default_factory=list)
#     completed: bool = False
#     cancelled: bool = False
#
#
# @dataclass
# class TimingStats:
#     read_time: float = 0.0
#     resize_time: float = 0.0
#     encode_time: float = 0.0
#     write_time: float = 0.0
#     total_time: float = 0.0
#     frame_count: int = 0
#
#     def summary(self) -> str:
#         if self.frame_count == 0:
#             return "No frames processed"
#
#         avg_read = (self.read_time / self.frame_count) * 1000
#         avg_resize = (self.resize_time / self.frame_count) * 1000
#         avg_encode = (self.encode_time / self.frame_count) * 1000
#         avg_write = (self.write_time / self.frame_count) * 1000
#         fps = self.frame_count / self.total_time if self.total_time > 0 else 0
#
#         return (
#             f"  Read:   {avg_read:.2f}ms\n"
#             f"  Resize: {avg_resize:.2f}ms\n"
#             f"  Encode: {avg_encode:.2f}ms\n"
#             f"  Write:  {avg_write:.2f}ms\n"
#             f"  Total:  {self.total_time:.2f}s ({fps:.1f} fps)"
#         )
#
#
# def _get_video_rotation(video_path: str) -> int:
#     ffprobe_candidates = [
#         "ffprobe",
#         r"C:\Program Files\ffmpeg\bin\ffprobe.exe",
#         r"C:\ffmpeg\bin\ffprobe.exe",
#     ]
#
#     ffprobe_path = None
#     for candidate in ffprobe_candidates:
#         try:
#             result = subprocess.run(
#                 [candidate, "-version"],
#                 capture_output=True,
#                 text=True,
#                 timeout=5
#             )
#             if result.returncode == 0:
#                 ffprobe_path = candidate
#                 break
#         except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
#             continue
#
#     if not ffprobe_path:
#         return 0
#
#     try:
#         result = subprocess.run(
#             [
#                 ffprobe_path,
#                 "-v", "quiet",
#                 "-print_format", "json",
#                 "-show_streams",
#                 video_path
#             ],
#             capture_output=True,
#             text=True,
#             timeout=10
#         )
#
#         if result.returncode != 0:
#             return 0
#
#         import json
#         data = json.loads(result.stdout)
#
#         for stream in data.get("streams", []):
#             if stream.get("codec_type") != "video":
#                 continue
#
#             for side_data in stream.get("side_data_list", []):
#                 if side_data.get("side_data_type") == "Display Matrix":
#                     rotation = abs(int(side_data.get("rotation", 0)))
#                     if rotation in (90, 180, 270):
#                         return rotation
#
#             tags = stream.get("tags", {})
#             rotation = abs(int(tags.get("rotation", 0)))
#             if rotation in (90, 180, 270):
#                 return rotation
#
#         return 0
#
#     except Exception:
#         return 0
#
#
# class FFmpegProgressParser:
#     """
#     Parses FFmpeg stderr output to extract progress information.
#
#     FFmpeg outputs lines like:
#         frame=  123 fps= 45 q=2.0 size=N/A time=00:00:04.10 ...
#
#     We extract the frame number to calculate progress percentage.
#     """
#
#     FRAME_PATTERN = re.compile(r'frame=\s*(\d+)')
#     FPS_PATTERN = re.compile(r'fps=\s*([\d.]+)')
#
#     def __init__(self, total_frames: int, nth_frame: int = 1):
#         self._total_frames = total_frames
#         self._nth_frame = nth_frame
#         self._expected_output_frames = total_frames // nth_frame if nth_frame > 1 else total_frames
#         self._current_frame = 0
#         self._current_fps = 0.0
#         self._last_update_time = 0.0
#         self._update_interval = 0.1
#
#     def parse_line(self, line: str) -> Optional[Tuple[int, float, float]]:
#         """
#         Parse a line of FFmpeg output.
#
#         Returns (current_frame, progress_percent, fps) if progress info found,
#         None otherwise.
#         """
#         frame_match = self.FRAME_PATTERN.search(line)
#         if not frame_match:
#             return None
#
#         self._current_frame = int(frame_match.group(1))
#
#         fps_match = self.FPS_PATTERN.search(line)
#         if fps_match:
#             try:
#                 self._current_fps = float(fps_match.group(1))
#             except ValueError:
#                 pass
#
#         if self._expected_output_frames > 0:
#             progress = min(100.0, (self._current_frame / self._expected_output_frames) * 100)
#         else:
#             progress = 0.0
#
#         return self._current_frame, progress, self._current_fps
#
#     def should_update(self) -> bool:
#         current_time = time.perf_counter()
#         if current_time - self._last_update_time >= self._update_interval:
#             self._last_update_time = current_time
#             return True
#         return False
#
#
# class FFmpegGPUDecoder:
#     _ffmpeg_path: Optional[str] = None
#     _gpu_decode_available: bool = False
#     _checked: bool = False
#
#     @classmethod
#     def _find_ffmpeg(cls) -> Optional[str]:
#         candidates = [
#             "ffmpeg",
#             r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
#             r"C:\ffmpeg\bin\ffmpeg.exe",
#         ]
#
#         for candidate in candidates:
#             try:
#                 result = subprocess.run(
#                     [candidate, "-version"],
#                     capture_output=True,
#                     text=True,
#                     timeout=5
#                 )
#                 if result.returncode == 0:
#                     return candidate
#             except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
#                 continue
#
#         return None
#
#     @classmethod
#     def check_availability(cls, log: Callable[[str], None]) -> Tuple[bool, str]:
#         if cls._checked:
#             return cls._gpu_decode_available, cls._ffmpeg_path or "not found"
#
#         cls._checked = True
#         cls._ffmpeg_path = cls._find_ffmpeg()
#
#         if not cls._ffmpeg_path:
#             log("FFmpeg not found. GPU decoding unavailable.")
#             log("  Install from: https://www.gyan.dev/ffmpeg/builds/")
#             log("  Download the 'full' build and add its bin folder to PATH.")
#             return False, "FFmpeg not found"
#
#         try:
#             result = subprocess.run(
#                 [cls._ffmpeg_path, "-decoders"],
#                 capture_output=True,
#                 text=True,
#                 timeout=5
#             )
#             has_h264_cuvid = "h264_cuvid" in result.stdout
#             has_hevc_cuvid = "hevc_cuvid" in result.stdout
#
#             if has_h264_cuvid or has_hevc_cuvid:
#                 cls._gpu_decode_available = True
#                 decoders = []
#                 if has_h264_cuvid:
#                     decoders.append("h264_cuvid")
#                 if has_hevc_cuvid:
#                     decoders.append("hevc_cuvid")
#                 log(f"FFmpeg GPU decoders available: {', '.join(decoders)}")
#                 return True, cls._ffmpeg_path
#             else:
#                 log("FFmpeg found but no CUDA decoders.")
#                 log("  You need the 'full' build from https://www.gyan.dev/ffmpeg/builds/")
#                 return False, "No CUDA decoders"
#
#         except Exception as e:
#             log(f"FFmpeg decoder check failed: {e}")
#             return False, str(e)
#
#     @classmethod
#     def _build_filter_graph(
#             cls,
#             nth_frame: int,
#             target_width: Optional[int],
#             target_height: Optional[int],
#             rotation: int
#     ) -> Tuple[str, bool]:
#         transpose_map = {
#             90: "transpose=1",
#             180: "transpose=1,transpose=1",
#             270: "transpose=3",
#         }
#
#         needs_cpu_filters = (nth_frame > 1) or (rotation in transpose_map)
#         needs_scaling = target_width is not None and target_height is not None
#
#         filters = []
#
#         if needs_cpu_filters:
#             if nth_frame > 1:
#                 filters.append(f"select='mod(n\\,{nth_frame}).eq(0)'")
#
#             if rotation in transpose_map:
#                 filters.append(transpose_map[rotation])
#
#             if needs_scaling:
#                 filters.append(f"scale={target_width}:{target_height}")
#
#             filters.append("format=bgr24")
#             return ",".join(filters), True
#
#         else:
#             if needs_scaling:
#                 filters.append(f"scale_cuda={target_width}:{target_height}")
#
#             filters.append("hwdownload")
#             filters.append("format=bgr24")
#             return ",".join(filters), False
#
#     @classmethod
#     def extract_frames_gpu(
#             cls,
#             video_path: str,
#             output_dir: str,
#             prefix: str,
#             video_name: str,
#             nth_frame: int,
#             target_width: Optional[int],
#             target_height: Optional[int],
#             jpeg_quality: int,
#             rotation: int,
#             total_frames: int,
#             stop_flag: threading.Event,
#             progress_callback: Optional[Callable[[str, float, str], None]],
#             video_basename: str,
#             log: Callable[[str], None]
#     ) -> Tuple[int, float]:
#         if not cls._ffmpeg_path or not cls._gpu_decode_available:
#             raise RuntimeError("FFmpeg GPU decoding not available")
#
#         start_time = time.perf_counter()
#
#         output_pattern = os.path.join(
#             output_dir,
#             f"{video_name}_{prefix}_%05d.jpg"
#         )
#
#         qscale = max(1, min(31, (100 - jpeg_quality) // 3))
#         filter_graph, needs_cpu_input = cls._build_filter_graph(
#             nth_frame, target_width, target_height, rotation
#         )
#
#         cmd = [cls._ffmpeg_path]
#
#         # Add progress flag to get frame-by-frame updates
#         cmd.extend(["-progress", "pipe:1"])
#
#         if needs_cpu_input:
#             cmd.extend(["-i", video_path])
#         else:
#             cmd.extend(["-hwaccel", "cuda"])
#             cmd.extend(["-hwaccel_output_format", "cuda"])
#             cmd.extend(["-i", video_path])
#
#         cmd.extend(["-vf", filter_graph])
#
#         if nth_frame > 1:
#             cmd.extend(["-vsync", "0"])
#
#         cmd.extend(["-qscale:v", str(qscale)])
#         cmd.extend(["-y"])
#         cmd.append(output_pattern)
#
#         log(f"  FFmpeg cmd: {' '.join(cmd)}")
#         log(f"  Filter graph: {filter_graph}")
#         if rotation:
#             log(f"  Applying rotation correction: {rotation} degrees")
#
#         progress_parser = FFmpegProgressParser(total_frames, nth_frame)
#         stderr_lines = []
#         current_frame = 0
#
#         try:
#             process = subprocess.Popen(
#                 cmd,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE,
#                 text=True,
#                 bufsize=1
#             )
#
#             def read_stderr():
#                 try:
#                     for line in process.stderr:
#                         stderr_lines.append(line)
#                 except Exception:
#                     pass
#
#             stderr_thread = threading.Thread(target=read_stderr, daemon=True)
#             stderr_thread.start()
#
#             expected_frames = total_frames // nth_frame if nth_frame > 1 else total_frames
#             last_progress_update = time.perf_counter()
#
#             while process.poll() is None:
#                 if stop_flag.is_set():
#                     process.terminate()
#                     try:
#                         process.wait(timeout=5)
#                     except subprocess.TimeoutExpired:
#                         process.kill()
#                         process.wait()
#                     elapsed = time.perf_counter() - start_time
#                     saved = cls._count_output_files(output_dir, video_name, prefix)
#                     return saved, elapsed
#
#                 # Read progress output from stdout (due to -progress pipe:1)
#                 try:
#                     line = process.stdout.readline()
#                     if line:
#                         # FFmpeg progress output format: key=value
#                         if line.startswith("frame="):
#                             try:
#                                 current_frame = int(line.split("=")[1].strip())
#                             except (ValueError, IndexError):
#                                 pass
#                 except Exception:
#                     pass
#
#                 # Update progress at regular intervals
#                 current_time = time.perf_counter()
#                 if current_time - last_progress_update >= 0.2:
#                     last_progress_update = current_time
#
#                     if progress_callback and expected_frames > 0:
#                         progress_pct = min(99.0, (current_frame / expected_frames) * 100)
#                         elapsed = current_time - start_time
#                         fps = current_frame / elapsed if elapsed > 0 else 0
#                         progress_callback(
#                             video_basename,
#                             progress_pct,
#                             f"Frame {current_frame}/{expected_frames} | {fps:.1f} fps"
#                         )
#
#             stderr_thread.join(timeout=2)
#
#             elapsed = time.perf_counter() - start_time
#             stderr_output = ''.join(stderr_lines)
#
#             if process.returncode != 0:
#                 log(f"  FFmpeg stderr: {stderr_output[-1000:]}")
#                 raise RuntimeError(f"FFmpeg exited with code {process.returncode}")
#
#             saved = cls._count_output_files(output_dir, video_name, prefix)
#
#             if saved == 0:
#                 log(f"  Warning: 0 frames produced. FFmpeg output:")
#                 log(f"  {stderr_output[-1000:]}")
#
#             return saved, elapsed
#
#         except FileNotFoundError:
#             raise RuntimeError(f"FFmpeg not found at: {cls._ffmpeg_path}")
#
#     @classmethod
#     def _count_output_files(cls, output_dir: str, video_name: str, prefix: str) -> int:
#         pattern = f"{video_name}_{prefix}_"
#         count = 0
#         for f in os.listdir(output_dir):
#             if f.startswith(pattern) and f.endswith(".jpg"):
#                 count += 1
#         return count
#
#
# class FrameExtractionEngine:
#     VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
#
#     def __init__(self, log_callback: Optional[Callable[[str], None]] = None):
#         self._log = log_callback or (lambda x: None)
#         self._stop_flag = threading.Event()
#         self._executor: Optional[ThreadPoolExecutor] = None
#         self._current_session: Optional[ExtractionSession] = None
#         self._progress_callback: Optional[Callable[[str, float, str], None]] = None
#         self._overall_progress_callback: Optional[Callable[[float, str], None]] = None
#
#         self._ffmpeg_available, self._ffmpeg_path = FFmpegGPUDecoder.check_availability(self._log)
#
#     def set_progress_callbacks(
#             self,
#             video_progress: Callable[[str, float, str], None],
#             overall_progress: Callable[[float, str], None]
#     ):
#         self._progress_callback = video_progress
#         self._overall_progress_callback = overall_progress
#
#     def analyze_video(self, video_path: str) -> Optional[VideoInfo]:
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             return None
#
#         try:
#             frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             duration = frame_count / fps if fps > 0 else 0
#             rotation = _get_video_rotation(video_path)
#
#             return VideoInfo(
#                 path=video_path,
#                 name=os.path.basename(video_path),
#                 frame_count=frame_count,
#                 width=width,
#                 height=height,
#                 fps=fps,
#                 duration=duration,
#                 rotation=rotation
#             )
#         finally:
#             cap.release()
#
#     def find_videos_in_directory(self, directory: str) -> List[str]:
#         videos = []
#         for file in os.listdir(directory):
#             if Path(file).suffix.lower() in self.VIDEO_EXTENSIONS:
#                 videos.append(os.path.join(directory, file))
#         return videos
#
#     def analyze_videos(
#             self,
#             video_paths: List[str],
#             nth_frame: int = 1
#     ) -> Tuple[List[VideoInfo], int, int]:
#         results = []
#         total_frames = 0
#         total_to_extract = 0
#
#         for path in video_paths:
#             info = self.analyze_video(path)
#             if info:
#                 results.append(info)
#                 total_frames += info.frame_count
#                 total_to_extract += info.frame_count // nth_frame
#
#         return results, total_frames, total_to_extract
#
#     def calculate_aspect_ratio_size(
#             self,
#             orig_width: int,
#             orig_height: int,
#             target_width: int,
#             target_height: int
#     ) -> Tuple[int, int]:
#         width_scale = target_width / orig_width
#         height_scale = target_height / orig_height
#         scale = min(width_scale, height_scale)
#         return int(orig_width * scale), int(orig_height * scale)
#
#     def start_extraction(
#             self,
#             video_paths: List[str],
#             settings: ExtractionSettings
#     ) -> None:
#         self._stop_flag.clear()
#         self._current_session = ExtractionSession(videos=video_paths.copy())
#
#         if settings.create_subdirs and not os.path.exists(settings.output_dir):
#             os.makedirs(settings.output_dir)
#             self._current_session.created_directories.append(settings.output_dir)
#
#         thread = threading.Thread(
#             target=self._run_extraction,
#             args=(video_paths, settings),
#             daemon=True
#         )
#         thread.start()
#
#     def stop_extraction(self) -> ExtractionSession:
#         self._stop_flag.set()
#         if self._executor:
#             self._executor.shutdown(wait=False, cancel_futures=True)
#
#         session = self._current_session
#         if session:
#             session.cancelled = True
#         return session
#
#     def cleanup_session(self, session: ExtractionSession) -> Tuple[int, int]:
#         files_removed = 0
#         dirs_removed = 0
#
#         for file_path in session.created_files:
#             try:
#                 if os.path.exists(file_path):
#                     os.remove(file_path)
#                     files_removed += 1
#             except Exception as e:
#                 self._log(f"Failed to remove file {file_path}: {e}")
#
#         for dir_path in reversed(session.created_directories):
#             try:
#                 if os.path.exists(dir_path):
#                     if not os.listdir(dir_path):
#                         os.rmdir(dir_path)
#                     else:
#                         shutil.rmtree(dir_path)
#                     dirs_removed += 1
#             except Exception as e:
#                 self._log(f"Failed to remove directory {dir_path}: {e}")
#
#         return files_removed, dirs_removed
#
#     def _run_extraction(
#             self,
#             video_paths: List[str],
#             settings: ExtractionSettings
#     ) -> None:
#         total_videos = len(video_paths)
#         completed_videos = 0
#         safe_threads = min(settings.num_threads, len(video_paths), os.cpu_count() or 4)
#
#         use_gpu = settings.use_gpu and self._ffmpeg_available
#
#         if use_gpu:
#             safe_threads = 1
#
#         overall_start = time.perf_counter()
#
#         self._log("=" * 60)
#         self._log("EXTRACTION STARTED")
#         self._log("=" * 60)
#         self._log(f"Videos:     {total_videos}")
#         self._log(f"Threads:    {safe_threads}")
#         self._log(f"Decode:     {'FFmpeg GPU (CUDA)' if use_gpu else 'OpenCV CPU'}")
#         self._log(
#             f"Resolution: {'Original' if settings.use_original_resolution else f'{settings.target_width}x{settings.target_height}'}")
#         self._log(f"Every:      {settings.nth_frame} frame(s)")
#         self._log(f"Quality:    {settings.jpeg_quality}")
#         self._log("-" * 60)
#
#         try:
#             self._executor = ThreadPoolExecutor(max_workers=safe_threads)
#             future_to_video: Dict[Future, str] = {}
#
#             for video_path in video_paths:
#                 if self._stop_flag.is_set():
#                     break
#
#                 future = self._executor.submit(
#                     self._extract_single_video,
#                     video_path,
#                     settings,
#                     use_gpu
#                 )
#                 future_to_video[future] = video_path
#
#             total_frames_extracted = 0
#
#             for future in as_completed(future_to_video):
#                 if self._stop_flag.is_set():
#                     break
#
#                 video_path = future_to_video[future]
#                 completed_videos += 1
#
#                 overall_progress = (completed_videos / total_videos) * 100
#                 if self._overall_progress_callback:
#                     self._overall_progress_callback(
#                         overall_progress,
#                         f"Processing {completed_videos}/{total_videos} videos..."
#                     )
#
#                 try:
#                     result = future.result()
#                     total_frames_extracted += result.saved_count
#                     status = "OK" if result.success else f"FAILED: {result.error}"
#                     self._log(
#                         f"[{completed_videos}/{total_videos}] {result.video_name}: "
#                         f"{status} - {result.saved_count} frames in {result.elapsed_time:.2f}s "
#                         f"({result.frames_per_second:.1f} fps)"
#                     )
#                 except Exception as e:
#                     self._log(f"[{completed_videos}/{total_videos}] {os.path.basename(video_path)}: {e}")
#
#             if self._current_session:
#                 self._current_session.completed = not self._stop_flag.is_set()
#
#             overall_elapsed = time.perf_counter() - overall_start
#             overall_fps = total_frames_extracted / overall_elapsed if overall_elapsed > 0 else 0
#
#             self._log("-" * 60)
#             self._log("SUMMARY")
#             self._log("-" * 60)
#             self._log(f"Total time:      {overall_elapsed:.2f}s")
#             self._log(f"Videos:          {completed_videos}/{total_videos}")
#             self._log(f"Frames:          {total_frames_extracted}")
#             self._log(f"Throughput:      {overall_fps:.1f} frames/sec")
#             if completed_videos > 0:
#                 self._log(f"Avg per video:   {overall_elapsed / completed_videos:.2f}s")
#             self._log("=" * 60)
#
#             if not self._stop_flag.is_set() and self._overall_progress_callback:
#                 self._overall_progress_callback(
#                     100,
#                     f"Done! {total_frames_extracted} frames in {overall_elapsed:.1f}s ({overall_fps:.1f} fps)"
#                 )
#
#         except Exception as e:
#             self._log(f"Extraction error: {e}")
#             if self._overall_progress_callback:
#                 self._overall_progress_callback(0, f"Error: {e}")
#         finally:
#             if self._executor:
#                 self._executor.shutdown(wait=True)
#                 self._executor = None
#
#     def _extract_single_video(
#             self,
#             video_path: str,
#             settings: ExtractionSettings,
#             use_gpu: bool
#     ) -> ExtractionResult:
#         video_name = Path(video_path).stem
#         video_basename = os.path.basename(video_path)
#
#         result = ExtractionResult(
#             video_name=video_basename,
#             success=False,
#             saved_count=0
#         )
#
#         try:
#             if self._progress_callback:
#                 path_label = "GPU" if use_gpu else "CPU"
#                 self._progress_callback(video_basename, 0, f"Starting ({path_label})...")
#
#             if settings.create_subdirs:
#                 video_output_dir = os.path.join(settings.output_dir, video_name)
#             else:
#                 video_output_dir = settings.output_dir
#
#             if not os.path.exists(video_output_dir):
#                 os.makedirs(video_output_dir)
#                 if self._current_session:
#                     self._current_session.created_directories.append(video_output_dir)
#
#             result.output_directory = video_output_dir
#
#             info = self.analyze_video(video_path)
#             rotation = info.rotation if info else 0
#             total_frames = info.frame_count if info else 0
#
#             target_w, target_h = None, None
#             if not settings.use_original_resolution and info:
#                 if settings.preserve_aspect and settings.target_width and settings.target_height:
#                     effective_w = info.height if rotation in (90, 270) else info.width
#                     effective_h = info.width if rotation in (90, 270) else info.height
#                     target_w, target_h = self.calculate_aspect_ratio_size(
#                         effective_w, effective_h,
#                         settings.target_width, settings.target_height
#                     )
#                 else:
#                     target_w = settings.target_width
#                     target_h = settings.target_height
#
#             if use_gpu:
#                 result = self._extract_gpu(
#                     video_path, video_output_dir, settings,
#                     video_name, video_basename, target_w, target_h,
#                     rotation, total_frames, result
#                 )
#             else:
#                 result = self._extract_cpu(
#                     video_path, video_output_dir, settings,
#                     video_name, video_basename, target_w, target_h, rotation, result
#                 )
#
#         except Exception as e:
#             result.error = str(e)
#             self._log(f"  {video_basename}: {e}")
#             if self._progress_callback:
#                 self._progress_callback(video_basename, 0, f"Error: {str(e)[:40]}")
#
#         return result
#
#     def _extract_gpu(
#             self,
#             video_path: str,
#             output_dir: str,
#             settings: ExtractionSettings,
#             video_name: str,
#             video_basename: str,
#             target_w: Optional[int],
#             target_h: Optional[int],
#             rotation: int,
#             total_frames: int,
#             result: ExtractionResult
#     ) -> ExtractionResult:
#         if self._progress_callback:
#             self._progress_callback(video_basename, 5, "FFmpeg GPU decoding...")
#
#         try:
#             saved, elapsed = FFmpegGPUDecoder.extract_frames_gpu(
#                 video_path=video_path,
#                 output_dir=output_dir,
#                 prefix=settings.prefix,
#                 video_name=video_name,
#                 nth_frame=settings.nth_frame,
#                 target_width=target_w,
#                 target_height=target_h,
#                 jpeg_quality=settings.jpeg_quality,
#                 rotation=rotation,
#                 total_frames=total_frames,
#                 stop_flag=self._stop_flag,
#                 progress_callback=self._progress_callback,
#                 video_basename=video_basename,
#                 log=self._log
#             )
#
#             result.success = not self._stop_flag.is_set()
#             result.saved_count = saved
#             result.elapsed_time = elapsed
#             result.frames_per_second = saved / elapsed if elapsed > 0 else 0
#
#             if self._current_session:
#                 pattern = f"{video_name}_{settings.prefix}_"
#                 for f in os.listdir(output_dir):
#                     if f.startswith(pattern) and f.endswith(".jpg"):
#                         self._current_session.created_files.append(
#                             os.path.join(output_dir, f)
#                         )
#
#             if self._progress_callback:
#                 status = f"Done: {saved} frames ({result.frames_per_second:.1f} fps)" if result.success else "Cancelled"
#                 self._progress_callback(video_basename, 100, status)
#
#         except Exception as e:
#             result.error = str(e)
#             self._log(f"  {video_basename} GPU failed: {e}")
#             self._log(f"  Falling back to CPU")
#             return self._extract_cpu(
#                 video_path, output_dir, settings,
#                 video_name, video_basename, target_w, target_h, rotation,
#                 ExtractionResult(video_name=video_basename, success=False, saved_count=0)
#             )
#
#         return result
#
#     def _extract_cpu(
#             self,
#             video_path: str,
#             output_dir: str,
#             settings: ExtractionSettings,
#             video_name: str,
#             video_basename: str,
#             target_w: Optional[int],
#             target_h: Optional[int],
#             rotation: int,
#             result: ExtractionResult
#     ) -> ExtractionResult:
#         stats = TimingStats()
#         video_start = time.perf_counter()
#
#         opencv_rotation_map = {
#             90: cv2.ROTATE_90_CLOCKWISE,
#             180: cv2.ROTATE_180,
#             270: cv2.ROTATE_90_COUNTERCLOCKWISE,
#         }
#
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             result.error = "Cannot open video"
#             return result
#
#         try:
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#             needs_resize = target_w is not None and target_h is not None
#             needs_rotation = rotation in opencv_rotation_map
#             encode_params = [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
#
#             frame_count = 0
#             saved_count = 0
#
#             while not self._stop_flag.is_set():
#                 t0 = time.perf_counter()
#                 success, frame = cap.read()
#                 stats.read_time += time.perf_counter() - t0
#
#                 if not success:
#                     break
#
#                 if frame_count % settings.nth_frame == 0:
#                     t0 = time.perf_counter()
#                     if needs_rotation:
#                         frame = cv2.rotate(frame, opencv_rotation_map[rotation])
#                     if needs_resize:
#                         frame = cv2.resize(
#                             frame,
#                             (target_w, target_h),
#                             interpolation=settings.interpolation.value
#                         )
#                     stats.resize_time += time.perf_counter() - t0
#
#                     filename = f"{video_name}_{settings.prefix}_{frame_count:05d}.jpg"
#                     filepath = os.path.join(output_dir, filename)
#
#                     t0 = time.perf_counter()
#                     success_encode, encoded = cv2.imencode('.jpg', frame, encode_params)
#                     stats.encode_time += time.perf_counter() - t0
#
#                     if success_encode:
#                         t0 = time.perf_counter()
#                         with open(filepath, 'wb') as f:
#                             f.write(encoded.tobytes())
#                         stats.write_time += time.perf_counter() - t0
#
#                         saved_count += 1
#                         stats.frame_count += 1
#
#                         if self._current_session:
#                             self._current_session.created_files.append(filepath)
#
#                 frame_count += 1
#
#                 if frame_count % 30 == 0 and self._progress_callback:
#                     progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
#                     elapsed = time.perf_counter() - video_start
#                     current_fps = saved_count / elapsed if elapsed > 0 else 0
#                     self._progress_callback(
#                         video_basename,
#                         progress,
#                         f"Frame {frame_count}/{total_frames} | {saved_count} saved | {current_fps:.1f} fps"
#                     )
#
#             result.success = not self._stop_flag.is_set()
#             result.saved_count = saved_count
#             result.elapsed_time = time.perf_counter() - video_start
#             result.frames_per_second = saved_count / result.elapsed_time if result.elapsed_time > 0 else 0
#             stats.total_time = result.elapsed_time
#
#             if result.success and saved_count > 50:
#                 self._log(f"  {video_basename} CPU timing (avg/frame):")
#                 self._log(stats.summary())
#
#             if self._progress_callback:
#                 status = f"Done: {saved_count} frames ({result.frames_per_second:.1f} fps)" if result.success else "Cancelled"
#                 self._progress_callback(video_basename, 100, status)
#
#         finally:
#             cap.release()
#
#         return result
