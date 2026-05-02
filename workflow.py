import json
from normalized_video import normalize_video
from segment_boxes import segment_objects
from visualize import build_heatmap, build_trajectory, build_video_visualization
from player_team_guesser import (
    find_player_team,
    normalize_results_data,
    normalized_keypoints_data,
    area_homography,
)
from track_object_id import track_object_id
from models import box_video_predict, keypoints_video_predict
import pandas as pd


def df_to_trajectories(df: pd.DataFrame):
    trajectories = []

    for _, group in df.groupby("track_id"):
        points = list(
            group[["frame_offset", "x", "y", "width", "height"]]
            .sort_values("frame_offset")
            .itertuples(index=False, name=None)
        )
        trajectories.append(points)

    return trajectories


normalize_video()

print("Starting predictions")
box_video_predict()
keypoints_video_predict()
print("Finished predictions")

normalized_keypoints_data()
data = normalize_results_data()
keypoints_data = normalized_keypoints_data()
homography_image = area_homography(data)
trackedData = track_object_id(data)
data_with_players = find_player_team(trackedData)
segmented_data = segment_objects(data_with_players)
print("Started building heatmaps...")
build_heatmap(data_with_players[data_with_players["class"] == "player"], "players")
build_heatmap(
    data_with_players[data_with_players["player_team"] == 1], "first team players"
)
build_heatmap(
    data_with_players[data_with_players["player_team"] == 2], "second team players"
)

trajectories = df_to_trajectories(data_with_players)
build_trajectory(trajectories)
build_video_visualization(segmented_data)
print("Finished building heatmaps")
