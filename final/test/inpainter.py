import cv2
import numpy as np

def inpaint(image, mask, patch_size=9):
    """
    Criminisi exemplar-based inpainting.
    image: Input image (grayscale).
    mask: Binary mask indicating regions to inpaint (1 for target, 0 for source).
    patch_size: Size of the patch (must be odd).
    """
    image = image.astype(np.float32)
    mask = mask.astype(np.uint8)
    half_patch = patch_size // 2

    # Confidence values
    confidence = (1 - mask).astype(np.float32)

    # Initialize output image
    output = image.copy()

    # Get image dimensions
    h, w = image.shape

    # Define valid region
    valid = 1 - mask

    while np.any(mask):
        # Find the contour of the target region
        dilated_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        contour = dilated_mask - mask

        # Compute priorities
        priorities = np.zeros_like(image)
        for y in range(half_patch, h - half_patch):
            for x in range(half_patch, w - half_patch):
                if contour[y, x]:
                    patch_confidence = confidence[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
                    priorities[y, x] = patch_confidence.mean()

        # Find the patch with the highest priority
        max_priority_idx = np.unravel_index(np.argmax(priorities), priorities.shape)
        y, x = max_priority_idx

        # Define the target patch
        target_patch = output[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
        target_mask = mask[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]

        # Search for the best matching patch
        best_match = None
        min_error = float('inf')
        for i in range(half_patch, h - half_patch):
            for j in range(half_patch, w - half_patch):
                source_patch = output[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1]
                source_mask = mask[i - half_patch:i + half_patch + 1, j - half_patch:j + half_patch + 1]
                if np.any(source_mask):
                    continue
                error = ((source_patch - target_patch) * (1 - target_mask)).sum()
                if error < min_error:
                    min_error = error
                    best_match = source_patch.copy()

        # Copy the best match into the target region
        if best_match is not None:
            output_patch = output[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
            mask_patch = mask[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1]
            output_patch[mask_patch == 1] = best_match[mask_patch == 1]
            output[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1] = output_patch
            mask[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1] = 0
            confidence[y - half_patch:y + half_patch + 1, x - half_patch:x + half_patch + 1] = \
                confidence[y, x]

    return output.astype(np.uint8)
