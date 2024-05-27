import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_of_interest(img):
    height = img.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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

def fit_lane_lines(lines, image_shape):
    left_lines = []
    right_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append([x1, y1])
                left_lines.append([x2, y2])
            else:
                right_lines.append([x1, y1])
                right_lines.append([x2, y2])
    
    left_points = np.array(left_lines)
    right_points = np.array(right_lines)

    if len(left_points) == 0 or len(right_points) == 0:
        print("Not enough points to fit lane lines")
        return []

    left_coeffs = np.polyfit(left_points[:, 1], left_points[:, 0], deg=1)
    right_coeffs = np.polyfit(right_points[:, 1], right_points[:, 0], deg=1)

    y_min = image_shape[0]
    y_max = int(image_shape[0] * 0.6)
    left_x_min = int((y_min - left_coeffs[1]) / left_coeffs[0])
    left_x_max = int((y_max - left_coeffs[1]) / left_coeffs[0])
    right_x_min = int((y_min - right_coeffs[1]) / right_coeffs[0])
    right_x_max = int((y_max - right_coeffs[1]) / right_coeffs[0])

    return [(left_x_min, y_min, left_x_max, y_max), (right_x_min, y_min, right_x_max, y_max)]

def combine_images(original_image, lane_image, alpha=0.8, beta=1., gamma=0.):
    return cv2.addWeighted(original_image, alpha, lane_image, beta, gamma)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(cropped_edges, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    filtered_lines = filter_lines(lines, slope_threshold=0.5, length_threshold=50)
    lane_lines = fit_lane_lines(filtered_lines, cropped_edges.shape)
    lane_image = np.copy(frame)
    for line in lane_lines:
        x1, y1, x2, y2 = line
        cv2.line(lane_image, (x1, y1), (x2, y2), (0, 255, 0), 10)
    combined_image = combine_images(frame, lane_image)
    return combined_image, gray, edges, cropped_edges, filtered_lines

def main():
    cap = cv2.VideoCapture('C:/Users/akhil/Downloads/video.mp4')
    plt.ion()
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result, gray, edges, cropped_edges, filtered_lines = process_frame(frame)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        axs[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axs[0, 0].set_title('Original Frame')
        axs[0, 1].imshow(gray, cmap='gray')
        axs[0, 1].set_title('Grayscale')
        axs[0, 2].imshow(edges, cmap='gray')
        axs[0, 2].set_title('Edges')
        axs[1, 0].imshow(cropped_edges, cmap='gray')
        axs[1, 0].set_title('Cropped Edges')
        axs[1, 1].imshow(result_rgb)
        axs[1, 1].set_title('Lane Detection Overlay')

        if filtered_lines is not None:
            line_img = np.zeros_like(frame)
            for line in filtered_lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
            line_img_rgb = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
            axs[1, 2].imshow(line_img_rgb)
            axs[1, 2].set_title('Filtered Lines')

        plt.pause(0.001)
        for ax in axs.flatten():
            ax.clear()

    cap.release()
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

