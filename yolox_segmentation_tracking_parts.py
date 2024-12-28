import cv2
import os
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from utils import calculate_mask_iou, merge_masks
from KalmanTracker import KalmanTracker

# Configurations
nParts = 1  # Divide image into nParts x nParts grid
fps_target = 5
model_instances = [YOLO("yolo11n-seg.pt") for _ in range(nParts * nParts)]  # Load <nParts> instances of the model
model = YOLO("yolo11n-seg.pt")
overlap = 100  # Overlap in pixels between zones
tracker = KalmanTracker()

video_path = "./videos/Processed_Videos/trimmed-172-3.9-sH4Y450PSVM_extended.mp4"


# Merge masks with high overlap before passing to the tracker
def merge_high_overlap_masks(predictions, masks, iou_threshold=0.5):
    """
    Given two prediction and global nparts variable this is to merge detection which occurs inside overlapping region.

    Logic is if the two detection in two different image parts has high overlapping segmented portion
    in overlapping region, they are merged to be one single mask. Important point being
        - Overlapping is checked only in the common zone because that's what is feed in both the images
        - They become one mask, so there is a risk that some boundary condition may merge too many of them

    :param predictions:
    :param masks:
    :param iou_threshold:
    :return:
    """
    merged_predictions = []
    merged_masks = []

    for i, (pred_box, mask) in enumerate(zip(predictions, masks)):
        merged = False
        x1, y1 = pred_box[0] - pred_box[2] / 2, pred_box[1] - pred_box[3] / 2
        x2, y2 = pred_box[0] + pred_box[2] / 2, pred_box[1] + pred_box[3] / 2
        for j, (merged_box, merged_mask) in enumerate(zip(merged_predictions, merged_masks)):
            # Define common area pred box in x,y ,w,h style x,y is centre point on bbox
            common_area = (
                max(pred_box[0] - pred_box[2] / 2, merged_box[0] - merged_box[2] / 2),
                max(pred_box[1] - pred_box[3] / 2, merged_box[1] - merged_box[3] / 2),
                min(pred_box[0] + pred_box[2] / 2, merged_box[0] + merged_box[2] / 2),
                min(pred_box[1] + pred_box[3] / 2, merged_box[1] + merged_box[3] / 2),
            )

            if common_area[2] <= common_area[0] or common_area[3] <= common_area[1]:
                continue  # No overlap

            # Calculate IoU for masks in the common area
            iou = calculate_mask_iou(mask, merged_mask, common_area)
            if iou > iou_threshold:
                # Merge masks and boxes
                merged_masks[j] = merge_masks(mask, merged_mask)
                merged_predictions[j] = [
                    (pred_box[0] + merged_box[0]) / 2,
                    (pred_box[1] + merged_box[1]) / 2,
                    max(pred_box[2], merged_box[2]),
                    max(pred_box[3], merged_box[3]),
                ]
                merged = True
                break

        if not merged:
            merged_predictions.append(pred_box)
            merged_masks.append(mask)

    return merged_predictions, merged_masks

def run_detections(video_path):

    input_folder = os.path.abspath(video_path)
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    w_orig, h_orig, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    output_path = os.path.join(input_folder, video_name.split(".")[0]+"_out.avi")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w_orig, h_orig))

    frame_interval = int(fps / fps_target)  # Process frames at the target fps
    frame_count = 0

    # Update the main loop
    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        # Divide the image into nParts x nParts grid with overlap
        h_split, w_split = h_orig // nParts, w_orig // nParts
        all_predictions = []
        all_masks = []

        for i in range(nParts):
            for j in range(nParts):
                x_start = max(j * w_split - overlap, 0)
                x_end = min((j + 1) * w_split + overlap, w_orig)
                y_start = max(i * h_split - overlap, 0)
                y_end = min((i + 1) * h_split + overlap, h_orig)
                sub_image = im0[y_start:y_end, x_start:x_end]

                results = model.predict(sub_image)
                for box, mask in zip(results[0].boxes.xywh.cpu().numpy(), results[0].masks.data.cpu().numpy()):
                    x, y, w, h = box
                    x += x_start
                    y += y_start
                    all_predictions.append([x, y, w, h])
                    all_masks.append(mask)

        # Merge masks before tracking
        merged_predictions, merged_masks = merge_high_overlap_masks(all_predictions, all_masks)

        # Update tracker
        tracks = tracker.update(merged_predictions, merged_masks)

        # Annotate tracks
        annotator = Annotator(im0, line_width=2)
        for track_id, track in tracks.items():
            x, y, w, h = track["box"]
            label = f"ID: {track_id}"
            color = colors(track_id, True)
            annotator.box_label((x - w / 2, y - h / 2, x + w / 2, y + h / 2), label, color=color)

        # Write the processed frame
        for _ in range(frame_interval):
            out.write(im0)

        cv2.imshow("instance-segmentation-object-tracking", im0)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    out.release()
    cap.release()
    cv2.destroyAllWindows()
