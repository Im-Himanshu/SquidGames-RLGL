import os.path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Configurations
nParts = 2  # Divide image into nParts x nParts grid
fps_target = 2
model_instances = [YOLO("yolo11n-seg.pt") for _ in range(nParts * nParts)]  # Load <nParts> instances of the model
overlap = 100  # Overlap in pixels between zones


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    else:
        inter_area = 0

    box1_area = w1 * h1
    box2_area = w2 * h2

    if inter_area/box1_area > 0.6:
        return 0.9 # if one of the box is inside another one
    if inter_area/box2_area > 0.6:
        return 0.9
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

video_path = "./videos/Processed_Videos/trimmed-172-3.9-sH4Y450PSVM_extended.mp4"
input_folder = os.path.abspath(video_path)
video_name = os.path.basename(video_path)
cap = cv2.VideoCapture(video_path)
w_orig, h_orig, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

output_path = os.path.join(input_folder, video_name.split(".")[0]+"_out.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w_orig, h_orig))

frame_interval = int(fps / fps_target)  # Process frames at the target fps
frame_count = 0

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    frame_count += 1
    if frame_count % frame_interval != 0:
        continue

    annotator = Annotator(im0, line_width=2)

    # Divide the image into nParts x nParts grid with overlap
    h_split, w_split = h_orig // nParts, w_orig // nParts
    results_aggregated = []

    for i in range(nParts):
        for j in range(nParts):
            # Extract sub-region with overlap
            x_start = max(j * w_split - overlap, 0)
            x_end = min((j + 1) * w_split + overlap, w_orig)
            y_start = max(i * h_split - overlap, 0)
            y_end = min((i + 1) * h_split + overlap, h_orig)
            sub_image = im0[y_start:y_end, x_start:x_end]

            # Run inference on sub-region using individual model instance
            model = model_instances[i * nParts + j]
            results = model.track(sub_image, persist=True)
            results_aggregated.append((results, (x_start, y_start)))

    # Aggregate results from all sub-regions and handle overlap
    global_boxes = []
    global_masks = []
    global_track_ids = []

    for results, (x_offset, y_offset) in results_aggregated:
        if results[0].boxes.id is not None and results[0].masks is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()  # Extract bounding boxes
            masks = results[0].masks.xy
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, mask, track_id in zip(boxes, masks, track_ids):
                x, y, w, h = box
                x += x_offset
                y += y_offset

                # Check if the box lies in the overlapping area
                in_overlap = (
                    (x_offset <= x <= x_offset + overlap or x_offset <= x + w <= x_offset + overlap) or
                    (y_offset <= y <= y_offset + overlap or y_offset <= y + h <= y_offset + overlap)
                )

                duplicate = False
                if in_overlap:
                    for existing_box in global_boxes:
                        if calculate_iou(existing_box, [x, y, w, h]) > 0.4:
                            duplicate = True
                            break

                if not duplicate:
                    global_boxes.append([x, y, w, h])
                    global_masks.append(mask + [x_offset, y_offset])
                    global_track_ids.append(track_id)

    # Annotate all global results
    for mask, box, track_id in zip(global_masks, global_boxes, global_track_ids):
        color = colors(track_id, True)
        txt_color = annotator.get_txt_color(color)
        annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)

    # Write the processed frame multiple times to match the original video duration
    for _ in range(frame_interval):
        out.write(im0)

    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()