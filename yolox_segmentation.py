import cv2
import os

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors


model = YOLO("yolo11n-seg.pt")  # segmentation model
names = model.model.names

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

    results = model.predict(im0)
    annotator = Annotator(im0, line_width=2)

    if results[0].masks is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy
        for mask, cls in zip(masks, clss):
            color = colors(int(cls), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)

    out.write(im0)
    cv2.imshow("instance-segmentation", im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

out.release()
cap.release()
cv2.destroyAllWindows()