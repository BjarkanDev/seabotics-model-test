from ultralytics import YOLO
import cv2
import os

def main():
    model = YOLO("model_v2/YOLO26n_buoy_detector_v2.pt")
    cap = cv2.VideoCapture("dataset/test_videos/20261004_094612.mp4")
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()

    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
    os.makedirs("dataset/inferred_videos", exist_ok=True)
    video_writer = cv2.VideoWriter("dataset/inferred_videos/YOLO26n_buoy_detector_v2_real.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)
        video_writer.write(results[0].plot())

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
