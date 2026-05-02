import json
import pandas as pd
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
from pathlib import Path


def area_homography(data: pd.DataFrame):
    pts_dst = np.array([[0, 0], [640, 0], [640, 480], [0, 480]]).astype(np.float32)

    keypoints_data = pd.read_json(
        "output/keypoints_predictions_normalized.json", orient="split"
    )
    video = cv2.VideoCapture("input/match.mp4")

    i = 0
    for _, row in keypoints_data.iterrows():
        video.set(cv2.CAP_PROP_POS_FRAMES, row["frame_offset"])
        ret, frame = video.read()
        if not ret or frame is None:
            continue

        pts_src = np.array([[81, 149], [567, 151], [640, 447], [0, 434]]).astype(
            np.float32
        )

        h, status = cv2.findHomography(pts_src, pts_dst)

        im_dst = cv2.warpPerspective(frame, h, (800, 600))

        cv2.imwrite(f"output/images/area_homography_{i}.png", im_dst)
        i += 1
        if i > 20:
            break
    return im_dst


def normalized_keypoints_data() -> pd.DataFrame:
    if Path("output/keypoints_predictions_normalized.json").exists():
        print("Skipping keypoints data normalization")
        print("Finished keypoints data normalization")
        return pd.read_json(
            "output/keypoints_predictions_normalized.json", orient="split"
        )
    keypoints_pointer_df = pd.read_json("output/keypoints_predictions.json")
    normalized_df = pd.json_normalize(
        keypoints_pointer_df["minecraft-hockey-rink-pointer"]
    )

    normalized_df["frame_offset"] = keypoints_pointer_df["frame_offset"]
    normalized_df["time_offset"] = keypoints_pointer_df["time_offset"]
    test = normalized_df.explode("predictions")
    test_non_null = test[test["predictions"].notna()].copy()

    test2 = pd.json_normalize(
        data=test_non_null["predictions"], meta=["time_offset", "frame_offset"]
    )
    test2["frame_offset"] = test_non_null["frame_offset"].values
    test2["time_offset"] = test_non_null["time_offset"].values
    test2["x_min"] = test2["x"] - test2["width"] / 2
    test2["y_min"] = test2["y"] - test2["height"] / 2
    test2["x_max"] = test2["x"] + test2["width"] / 2
    test2["y_max"] = test2["y"] + test2["height"] / 2
    test_final = test2
    test_final.to_json(
        "output/keypoints_predictions_normalized.json",
        orient="split",
        force_ascii=False,
    )
    print("Keypoints data normalized")
    return test_final


def normalize_results_data() -> pd.DataFrame:
    print("Normalizing data...")
    if Path("output/box_predictions_normalized.json").exists():
        print("Skipping normalizing data")
        print("Data normalized")
        return pd.read_json("output/box_predictions_normalized.json", orient="split")
    with open("output/box_predictions.json", "r") as file:
        data = json.load(file)
    normalized_df = pd.json_normalize(data["hockey-labeler-drbmv"])

    normalized_df["frame_offset"] = data["frame_offset"]
    normalized_df["time_offset"] = data["time_offset"]
    test = normalized_df.explode("predictions")
    test_non_null = test[test["predictions"].notna()].copy()

    test2 = pd.json_normalize(
        data=test_non_null["predictions"], meta=["time_offset", "frame_offset"]
    )
    test2["frame_offset"] = test_non_null["frame_offset"].values
    test2["time_offset"] = test_non_null["time_offset"].values
    test2["x_min"] = test2["x"] - test2["width"] / 2
    test2["y_min"] = test2["y"] - test2["height"] / 2
    test2["x_max"] = test2["x"] + test2["width"] / 2
    test2["y_max"] = test2["y"] + test2["height"] / 2
    test_final = test2
    test_final.to_json(
        "output/box_predictions_normalized.json", orient="split", force_ascii=False
    )
    print("Data normalized")
    return test_final


def find_player_team(data: pd.DataFrame):
    print("Starting guess player team...")
    if Path("output/data_with_players.json").exists():
        print("Skipping player team guessing")
        print("Finished guess player team")
        return pd.read_json("output/data_with_players.json", orient="split")
    players = data[data["class"] == "player"].reset_index(drop=True).copy()
    player_hsv = {}
    cap = cv2.VideoCapture("input/match.mp4")
    procent = len(players) / 100
    processed_rows = 0
    for track_id, group in players.groupby("track_id"):
        track_crop_hsv = []
        for i, row in group.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_offset"])
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            crop = frame[
                int(row["y_min"]) : int(row["y_max"]),
                int(row["x_min"]) : int(row["x_max"]),
            ]
            hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            median_hsv = np.median(hsv_crop.reshape(-1, 3), axis=0)
            track_crop_hsv.append(median_hsv)
            processed_rows += 1
            print(
                f"\rGuessed {round(processed_rows / procent)}% rows", flush=True, end=""
            )
        median_hsv = np.median(track_crop_hsv, axis=0)
        player_hsv[track_id] = median_hsv

    hsv_array = np.array(list(player_hsv.values()))
    if len(hsv_array) < 2:
        return pd.DataFrame()
    hsv_norm = np.column_stack(
        [hsv_array[:, 0] / 179, hsv_array[:, 1] / 255, hsv_array[:, 2] / 255]
    )
    centroids_norm, _ = kmeans(hsv_norm, 2)
    labels, _ = vq(hsv_norm, centroids_norm)
    counts = np.bincount(labels)
    if counts[0] > counts[1]:
        team_map = {0: 1, 1: 2}
    else:
        team_map = {0: 2, 1: 1}
    teams = [team_map.get(l, 0) for l in labels]

    cap.release()
    result_df = data.copy()
    result_df["player_team"] = np.nan
    track_ids_list = list(player_hsv.keys())
    track_to_team = {track_ids_list[i]: teams[i] for i in range(len(teams))}

    player_mask = (result_df["class"] == "player") & (
        result_df["track_id"].isin(track_to_team.keys())
    )
    result_df.loc[player_mask, "player_team"] = result_df.loc[
        player_mask, "track_id"
    ].map(track_to_team)
    result_df.to_json(
        "output/data_with_players.json", orient="split", force_ascii=False
    )
    print("\nFinished guess player team")
    return result_df
