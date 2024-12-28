import numpy as np

def calculate_mask_iou(mask1, mask2, common_area):
    # Crop masks to the common area
    mask1_common = mask1[common_area[1]:common_area[3], common_area[0]:common_area[2]]
    mask2_common = mask2[common_area[1]:common_area[3], common_area[0]:common_area[2]]

    # Calculate intersection and union
    intersection = np.logical_and(mask1_common, mask2_common).sum()
    union = np.logical_or(mask1_common, mask2_common).sum()

    if union == 0:
        return 0
    return intersection / union

def merge_masks(mask1, mask2):
    return np.logical_or(mask1, mask2).astype(np.uint8)