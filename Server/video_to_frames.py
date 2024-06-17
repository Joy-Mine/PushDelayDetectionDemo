import cv2
import sys
import os

def video_to_frames(video_path, frames_dir):
    # 创建帧保存目录
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(frames_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames to {frames_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python video_to_frames.py <video_path> <frames_dir>")
    else:
        video_path = sys.argv[1]
        frames_dir = sys.argv[2]
        video_to_frames(video_path, frames_dir)
