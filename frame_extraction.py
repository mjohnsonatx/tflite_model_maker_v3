import cv2
import os


def extract_frames(video_path: str, output_dir: str, prefix: str = "frame"):
    target_width = 800
    target_height = 600

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Skipping: cannot open video file: {video_path}")
        return

    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame is None or frame.size == 0:
            print(f"⚠️ Empty frame at index {frame_count} in {video_name}")
            continue

        try:
            resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            filename_prefix = os.path.basename(video_output_dir)
            filename = os.path.join(video_output_dir, f"{filename_prefix}_{prefix}_{frame_count:05d}.jpg")
            success_write = cv2.imwrite(filename, resized)
            if not success_write:
                print(f"❌ Failed to write frame {frame_count} of {video_name}")
            frame_count += 1
        except Exception as e:
            print(f"❌ Error processing frame {frame_count}: {e}")

    cap.release()
    print(f"✅ {video_name}: Extracted and resized {frame_count} frames to {video_output_dir}")


def process_videos_in_directory(input_dir: str, output_dir: str, prefix: str = "frame"):
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

    if not video_files:
        print(f"🚫 No MP4 videos found in {input_dir}")
        return

    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        extract_frames(video_path, output_dir, prefix)


if __name__ == "__main__":
    process_videos_in_directory('kettlebell_videos', 'sliced_kb_videos_into_frames', 'output')
