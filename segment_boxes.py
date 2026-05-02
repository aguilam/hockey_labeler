from pathlib import Path
import cv2
import pandas as pd
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import numpy as np
import json


def segment_objects(data: pd.DataFrame):
    print("Starting object segmentation...")
    if Path("output/segmented_data.json").exists():
        print("Skipping objects segmentation")
        with open("output/segmented_data.json", "r") as f:
            data = json.load(f)
        df_normalized = pd.json_normalize(data)
        columns = df_normalized.loc[0, "columns"]
        data = df_normalized.loc[0, "data"]

        return pd.DataFrame(data, columns=columns)
    filteredData = (
        data[(data["class"] != "scoreboard") & (data["class"] != "goal")]
        .reset_index()
        .copy()
    )
    video = cv2.VideoCapture("input/match.mp4")
    model = build_sam3_image_model(
        bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        device="cuda",
        eval_mode=True,
        checkpoint_path="sam3.pt",
        load_from_HF=False,
    )
    i = 0
    processor = Sam3Processor(model, device="cuda")
    procent = 1000 / 100
    # change to proceed rows
    processed_rows = 0
    objects_point = []
    for _, row in filteredData.iterrows():
        video.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_offset"]))
        ret, frame = video.read()
        if not ret or frame is None:
            continue

        x1 = int(row["x_min"]) - 5
        y1 = int(row["y_min"]) - 5
        x2 = int(row["x_max"]) + 5
        y2 = int(row["y_max"]) + 5

        h_full, w_full = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w_full, x2)
        y2 = min(h_full, y2)

        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            continue

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cx_abs = float(row["x"])
        cy_abs = float(row["y"])
        box_w_abs = float(row["width"])
        box_h_abs = float(row["height"])

        crop_h, crop_w = crop.shape[:2]
        cx_rel = (cx_abs - x1) / crop_w
        cy_rel = (cy_abs - y1) / crop_h
        nw = box_w_abs / crop_w
        nh = box_h_abs / crop_h
        box_norm = [cx_rel, cy_rel, nw, nh]
        pil_crop = Image.fromarray(crop_rgb)

        state = processor.set_image(pil_crop)
        state = processor.add_geometric_prompt(box=box_norm, label=True, state=state)
        state = processor.set_text_prompt(prompt="Central object", state=state)

        masks = (state["masks"].cpu().numpy() * 255).astype(np.uint8)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0, :, :]

        if masks.ndim == 2:
            masks = masks[None, ...]

        points = []
        for i in range(masks.shape[0]):
            m = masks[i]
            if m.ndim == 3:
                if m.shape[2] == 1:
                    m = m[..., 0]
                elif m.shape[2] in (3, 4):
                    m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
                else:
                    m = m[..., 0]
            _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in contours:
                if c.shape[0] < 3:
                    continue
                normalized_points = []
                for point in c:
                    x, y = point[0]
                    normalized_points.append([x + x1, y + y1])
                points.append(normalized_points)
        objects_point.append([row["detection_id"], points])
        processed_rows += 1
        i += 1
        if processed_rows > 1000:
            break
        print(
            f"\r Segmented {round(processed_rows / procent)}% objects",
            flush=True,
            end="",
        )
    video.release()
    pointed_pd = pd.DataFrame(objects_point, columns=["detection_id", "mask_points"])
    merged_pd = data.merge(pointed_pd, on="detection_id", how="left")
    merged_pd.to_json("output/segmented_data.json", orient="split", force_ascii=False)
    print("\nFinished segments objects")
    return merged_pd
