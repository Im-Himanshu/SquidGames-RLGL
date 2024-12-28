import cv2
import os
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

model = YOLO("yolo11n-seg.pt")
# model_instances = [YOLO("yolo11n-seg.pt") for _ in range(nParts * nParts)]  # Load <nParts> instances of the model
video_path = "./videos/Processed_Videos/trimmed-172-3.9-sH4Y450PSVM_extended.mp4"
input_folder = os.path.abspath(video_path)
video_name = os.path.basename(video_path)
cap = cv2.VideoCapture(video_path)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

output_path = os.path.join(input_folder, video_name.split(".")[0]+"_out.avi")
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    annotator = Annotator(im0, line_width=2)

    results = model.track(im0, persist=True)

    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for mask, track_id in zip(masks, track_ids):
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=str(track_id), txt_color=txt_color)

    out.write(im0)
    cv2.imshow("instance-segmentation-object-tracking", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()