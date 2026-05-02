import cv2
import pandas as pd
import numpy as np
from ultralytics.trackers import BYTETracker
from argparse import Namespace
from ultralytics.engine.results import Boxes
from pathlib import Path
from collections import defaultdict


def numpy_iou_box(box_a, box_b):
    box_a = np.array(box_a)
    box_b = np.array(box_b)

    x_min = max(box_a[0], box_b[0])
    y_min = max(box_a[1], box_b[1])
    x_max = min(box_a[2], box_b[2])
    y_max = min(box_a[3], box_b[3])

    intersection_area = max(0, x_max - x_min) * max(0, y_max - y_min)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = (box_a_area + box_b_area) - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def track_object_id(data: pd.DataFrame) -> pd.DataFrame:
    print("Starting object tracking...")
    if Path("output/tracked_data.json").exists():
        print("Skipping object tracking")
        print("Finished object tracking")
        return pd.read_json("output/tracked_data.json", orient="split")
    filtered_df = data.copy()
    filtered_df.drop(["class"], axis=1, inplace=True)
    sorted_df = filtered_df.sort_values(by="frame_offset", ascending=True)
    args = Namespace(track_buffer=60, track_high_thresh=0.6, track_low_thresh=0.1)
    _default_args = {
        "match_thresh": 0.5,
        "max_time_lost": 120,
        "new_track_thresh": 0.6,
        "fuse_score": False,
        "track_tentative": False,
        "use_byte": True,
    }

    for k, v in _default_args.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    tracker = BYTETracker(args, frame_rate=20)
    grouped_df = sorted_df.groupby("frame_offset")
    tracking_results = defaultdict(list)
    procent = grouped_df.ngroups / 100
    tracked_frames = 0
    for frame_id, group in grouped_df:
        group = group.copy()
        if group.empty:
            continue

        if "confidence" not in group.columns:
            group.loc[:, "confidence"] = 1.0

        arr = group[
            ["x_min", "y_min", "x_max", "y_max", "confidence", "class_id"]
        ].to_numpy(dtype=np.float32)
        if arr.size == 0:
            continue

        results = Boxes(arr, (480, 640))
        tracks = tracker.update(results)
        if tracks is not None and len(tracks) > 0:
            for track in tracks:
                for _, item in group.iterrows():
                    iou_score = numpy_iou_box(
                        np.array(track[:4]),
                        item[["x_min", "y_min", "x_max", "y_max"]].to_numpy(),
                    )
                    if iou_score >= 0.8:
                        tracking_results[frame_id].append((track, item["detection_id"]))
        tracked_frames += 1
        print(
            f"\rTracked {round(tracked_frames / procent)}% frames", flush=True, end=""
        )

    trajectories = []
    for frame, tracks in tracking_results.items():
        for track, detection_id in tracks:
            x1, y1, x2, y2, track_id, conf, *_ = track
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            trajectories.append(
                (
                    frame,
                    float(center_x),
                    float(center_y),
                    int(track_id),
                    conf,
                    detection_id,
                )
            )
    results_df = pd.DataFrame(
        trajectories,
        columns=["frame_offset", "x", "y", "track_id", "conf", "detection_id"],
    )
    merged_df = data.merge(
        results_df[["detection_id", "track_id"]], on="detection_id", how="left"
    )
    merged_df["track_id"] = merged_df["track_id"].fillna(-1).astype(int)
    merged_df.to_json("output/tracked_data.json", orient="split", force_ascii=False)
    print("\nFinished object tracking")
    return merged_df
