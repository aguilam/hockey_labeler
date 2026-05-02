import cv2


def normalize_video():
    print("Starting video normalization")
    cap = cv2.VideoCapture("input/raw_match.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("input/match.mp4", fourcc, 20.0, (640, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        out.write(cv2.resize(frame, (640, 480)))

    cap.release()
    out.release()
    print("Finished video normalization")
