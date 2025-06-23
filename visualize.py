import cv2
import matplotlib.pyplot as plt

# Load original and predicted
image = cv2.imread("data/images/lake1.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
binary_mask = cv2.imread("outputs/predicted_mask.png", cv2.IMREAD_GRAYSCALE)

# Show side-by-side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(binary_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")
plt.tight_layout()
plt.show()

# Optional overlay
mask_rgb = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2RGB)
overlay = cv2.addWeighted(image, 0.7, mask_rgb, 0.3, 0)
cv2.imwrite("outputs/overlay.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
