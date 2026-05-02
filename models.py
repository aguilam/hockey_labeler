from roboflow import Roboflow
import json
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(os.getenv("ROBOFLOW_KEY"))


def box_video_predict():
    print("Started box predict")
    if Path("output/box_predictions.json").exists():
        print("Skipping box predict")
        print("Finished box predict")
        return
    project = rf.workspace().project("hockey-labeler-drbmv")
    model = project.version("1").model
    job_id, signed_url, expire_time = model.predict_video(
        "input/match.mp4", fps=20, prediction_type="batch-video"
    )
    print(signed_url)
    if not job_id:
        raise RuntimeError(f"predict_video failed for box: no job_id returned")
    results = model.poll_until_video_results(job_id)
    if isinstance(results, dict) and ("status" in results or "status_info" in results):
        status = results.get("status")
        status_info = results.get("status_info")
        if status not in (
            0,
            None,
        ):
            raise RuntimeError(
                f"job {job_id} finished with status={status}, status_info={status_info}"
            )
    with open("output/box_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Finished box predict")


def keypoints_video_predict():
    print("Started keypoints predict")
    if Path("output/keypoints_predictions.json").exists():
        print("Skipping keypoints predict")
        print("Finished keypoints predict")
        return
    project = rf.workspace().project("minecraft-hockey-rink-pointer")
    model = project.version("7").model
    job_id, signed_url, expire_time = model.predict_video(
        "input/match.mp4",
        fps=20,
        prediction_type="batch-video",
    )
    print(signed_url)
    if not job_id:
        raise RuntimeError(f"predict_video failed for keypoints: no job_id returned")
    results = model.poll_until_video_results(job_id)
    if isinstance(results, dict) and ("status" in results or "status_info" in results):
        status = results.get("status")
        status_info = results.get("status_info")
        if status not in (
            0,
            None,
        ):
            raise RuntimeError(
                f"job {job_id} finished with status={status}, status_info={status_info}"
            )
    with open("output/keypoints_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Finished keypoints predict")
