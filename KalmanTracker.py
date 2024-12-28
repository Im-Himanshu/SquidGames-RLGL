# Update the KalmanTracker class
import numpy as np
from filterpy.kalman import KalmanFilter
from utils import  calculate_mask_iou, merge_masks
class KalmanTracker:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_lost = 30  # Max frames to keep a lost track

    def _create_kf(self):
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.eye(7)  # State transition matrix
        kf.H = np.eye(4, 7)  # Measurement function
        kf.P *= 10.0  # Covariance matrix
        kf.R *= 1.0  # Measurement noise
        kf.Q *= 0.01  # Process noise
        return kf

    def update(self, detections, masks):
        updated_tracks = {}
        for track_id, track in self.tracks.items():
            track["kf"].predict()

        # Match detections to existing tracks using Mask IoU
        unmatched_detections = set(range(len(detections)))
        unmatched_tracks = set(self.tracks.keys())

        iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
        track_ids = list(self.tracks.keys())
        for i, track_id in enumerate(track_ids):
            for j, mask in enumerate(masks):
                iou_matrix[i, j] = calculate_mask_iou(mask, self.tracks[track_id]["mask"], (0, 0, mask.shape[1], mask.shape[0]))

        for i, j in zip(*np.where(iou_matrix > 0.3)):
            track_id = track_ids[i]
            self.tracks[track_id]["kf"].update(detections[j])
            self.tracks[track_id]["box"] = detections[j]
            self.tracks[track_id]["mask"] = merge_masks(masks[j], self.tracks[track_id]["mask"])
            updated_tracks[track_id] = self.tracks[track_id]
            unmatched_detections.discard(j)
            unmatched_tracks.discard(track_id)

        # Create new tracks for unmatched detections
        for detection_index in unmatched_detections:
            kf = self._create_kf()
            kf.x[:4] = detections[detection_index][:4]  # Initialize state with detection
            updated_tracks[self.next_id] = {
                "kf": kf,
                "box": detections[detection_index],
                "mask": masks[detection_index],
                "lost": 0,
            }
            self.next_id += 1

        # Increment lost counter for unmatched tracks
        for track_id in unmatched_tracks:
            self.tracks[track_id]["lost"] += 1
            if self.tracks[track_id]["lost"] > self.max_lost:
                continue
            updated_tracks[track_id] = self.tracks[track_id]

        self.tracks = updated_tracks
        return self.tracks