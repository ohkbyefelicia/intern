import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread('C:/Users/akhil/Downloads/i.webp')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)
plt.imshow(edges, cmap='gray')
plt.title('Preprocessed Image')
plt.show()


def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

cropped_edges = region_of_interest(edges)
plt.imshow(cropped_edges, cmap='gray')
plt.title('Edges after ROI Application')
plt.show()


lines = cv2.HoughLinesP(cropped_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)


line_image = np.copy(image)
if lines is not None:
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
plt.imshow(line_image_rgb)
plt.title('Detected Lines')
plt.show()


def filter_lines(lines, slope_threshold, length_threshold):
    filtered_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x2 - x1 == 0:
                slope = np.inf
            else:
                slope = (y2 - y1) / (x2 - x1)
            line_length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if abs(slope) > slope_threshold and line_length > length_threshold:
                filtered_lines.append(line)
    return np.array(filtered_lines)

lines = cv2.HoughLinesP(cropped_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
filtered_lines = filter_lines(lines, slope_threshold=0.5, length_threshold=50)


line_image = np.copy(image)
if filtered_lines is not None:
    for line in filtered_lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

line_image_rgb = cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB)
plt.imshow(line_image_rgb)
plt.title('Filtered Lines')
plt.show()


