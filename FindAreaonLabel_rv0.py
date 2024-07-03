import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/Users/basmech/Desktop/test.jpeg"
image = cv2.imread(image_path)
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Blurring Image
# cv2.medianBlur(gray,1)
# # Balancing Brightness
# cv2.equalizeHist(gray)

_, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

# Apply morphological operations to highlight horizontal lines
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))  # Kernel size can be adjusted
morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(morph, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)

# Draw the detected lines on the original image
output_image = image.copy()
max_length = 0
longest_line = None
x1 = 0
y1 = 0
x2 = 0
y2 = 0

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > max_length:
            max_length = length
            longest_line = line[0]
        # cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Draw the longest line in red
if longest_line is not None:
    x1, y1, x2, y2 = longest_line
    cv2.line(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

height, width = image.shape[:2]

lenc = int(x1 / 2)
lenc0 = 0

if width > height:
    lenc0 = int(height/10)
elif width <= height:
    lenc0 = int(width/10)

if lenc > lenc0:
    lenc = lenc0

if x1 - lenc > 0:
    x1 = x1 - lenc
elif x1 - lenc <= 0:
    x1 = 1

if x2 + lenc > width:
    x2 = width
elif x2 + lenc <= width:
    x2 = x2 + lenc


# Y1 >> Y2 and X1 >> X2  => [1:100, 300:600]
Croped_img = image[1:height, x1:x2]

# Display the original and output images
plt.figure(figsize=(10, 8))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(Croped_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()
