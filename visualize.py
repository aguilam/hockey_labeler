from math import nan
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd


def build_heatmap(data: pd.DataFrame, heatmap_name: str):
    bboxes = []
    for _, row in data.iterrows():
        bboxes.append(
            [
                row["x"],
                row["y_max"],
                row["width"],
                row["height"],
            ]
        )

    if len(bboxes) == 0:
        feet_points = np.zeros((0, 4), dtype=np.float64)
    else:
        feet_points = np.array(bboxes, dtype=np.float64)

    H, _, _ = np.histogram2d(
        feet_points[:, 0],
        feet_points[:, 1],
        bins=(256, 256),
        range=[[0, 640], [0, 480]],
    )
    H_smooth = gaussian_filter(H.T, sigma=2)
    H_log = np.log1p(H_smooth)
    H_norm = H_log / H_log.max() * 255
    H_uint8 = H_norm.astype(np.uint8)
    color_map = cv2.applyColorMap(H_uint8, cv2.COLORMAP_HOT)
    overlay_image = cv2.resize(color_map, (640, 480), interpolation=cv2.INTER_LINEAR)
    court_photo = cv2.imread("input/overlay_image.png")
    court_photo_resized = cv2.resize(court_photo, (640, 480))
    dst = cv2.addWeighted(court_photo_resized, 0.6, overlay_image, 0.4, 0.0)
    cv2.imwrite(f"output/{heatmap_name}.png", dst)


def build_trajectory(trajectories: list):
    overlay_img = np.zeros((480, 640, 3), np.uint8)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output/Trajectory.mp4", fourcc, 30, (640, 480))

    court_photo = cv2.imread("input/overlay_image.png")
    court_photo_resized = cv2.resize(court_photo, (640, 480))

    last_point = []
    for _, points in enumerate(trajectories):
        if last_point:
            x_prev, y_prev = last_point
            _, x_first, y_first, _, _ = points[0]
            cv2.line(
                overlay_img,
                (int(x_prev), int(y_prev)),
                (int(x_first), int(y_first)),
                (0, 255, 0),
                2,
            )
        for i in range(1, len(points)):
            _, x_prev, y_prev, _, _ = points[i - 1]
            _, x_cur, y_cur, _, _ = points[i]
            cv2.line(
                overlay_img,
                (int(x_prev), int(y_prev)),
                (int(x_cur), int(y_cur)),
                (0, 255, 0),
                2,
            )
        for _, x, y, _, _ in points:
            cv2.circle(overlay_img, (int(x), int(y)), 4, (0, 0, 255), -1)

        dst = cv2.addWeighted(court_photo_resized, 0.6, overlay_img, 0.4, 0.0)
        out.write(dst)
        last_point = points[-1][1:3]
        cv2.imwrite("output/trajectory.png", dst)
    out.release()


def build_video_visualization(data: pd.DataFrame):
    colors = [(0, 0, 0), (0, 0, 255), (0, 225, 255), (255, 225, 100), (0, 255, 0)]
    teams = ["", "Gamblers", "Ghosts"]
    video = cv2.VideoCapture("input/match.mp4")
    video_out = cv2.VideoWriter(
        "output/final_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        30,
        frameSize=(640, 480),
    )
    for key, group in data.groupby("frame_offset"):
        video.set(cv2.CAP_PROP_POS_FRAMES, key)
        ret, frame = video.read()
        shapes = np.zeros_like(frame, np.uint8)
        boxes_to_draw = []
        if not ret or frame is None:
            continue
        for _, box in group.iterrows():
            box_color = colors[box["class_id"]]
            if box["mask_points"] is not np.nan and box["mask_points"] is not None:
                pts = [
                    np.array(polygon, dtype=np.int32) for polygon in box["mask_points"]
                ]
                cv2.fillPoly(shapes, pts, box_color)
            else:
                cv2.rectangle(
                    shapes,
                    (box["x_min"], box["y_min"]),
                    (box["x_max"], box["y_max"]),
                    box_color,
                    -1,
                )
                boxes_to_draw.append((box, box_color))
        alpha = 0.4
        blended = cv2.addWeighted(frame, 1.0 - alpha, shapes, alpha, 0.0)
        mask = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY) > 0
        dst = np.where(mask[:, :, None], blended, frame).astype(np.uint8)
        for box, box_color in boxes_to_draw:
            cv2.rectangle(
                dst,
                (int(box["x_min"]), int(box["y_min"])),
                (int(box["x_max"]), int(box["y_max"])),
                box_color,
                2,
            )
        for _, box in group.iterrows():
            box_id = str(box["track_id"])
            class_title = (
                "#" + box_id + "" + box["class"] + " " + teams[int(box["player_team"])]
                if box["class"] == "player" and pd.notna(box["player_team"])
                else box["class"]
            )
            cv2.putText(
                dst,
                class_title,
                (box["x_min"], box["y_min"] - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.3,
                color=(0, 0, 0),
            )
        video_out.write(dst)
    video.release()
    video_out.release()
