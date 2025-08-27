import cv2 as cv
import skimage.io
import numpy as np
import math
import os
import glob
import argparse



def get_hough_angle(img):

    """
    Estimate the skew angle of the image using Hough Line Transform.

    Steps:
        1. Convert the image to grayscale and apply bilateral filter to reduce noise.
        2. Use Canny edge detection to find edges in the image.
        3. Apply Hough Line Transform to detect lines in the edge-detected image.
        4. Calculate the angle of the detected lines.
        5. Reject outliers and compute the mean angle.
        6. Rotate the original image to correct the skew.
        7. Resize or pad the image to a target size.
    """
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgray = cv.bilateralFilter(imgray, 10, 10, 10)

    v = np.median(imgray)
    sigma = 0.4
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    edges = cv.Canny(imgray, lower_thresh, upper_thresh)

    lines = cv.HoughLinesP(edges, 1, math.pi / 128, 40, None, 60, 10)
    if lines is None or len(lines) < 5:
        lines = cv.HoughLinesP(edges, 1, math.pi / 128, 40, None, 80, 20)
    if lines is None or len(lines) < 5:
        lines = cv.HoughLinesP(edges, 1, math.pi / 128, 20, None, 80, 20)
    if lines is None:
        return 0

    lines_theta = np.array([
        np.arctan((line[0][3] - line[0][1]) / (line[0][2] - line[0][0] + 1e-5))
        for line in lines[:100]
    ])

    def reject_outliers(data, m=2):
        return data[np.abs(data - np.mean(data)) < m * np.std(data)]

    filter_lines_theta = reject_outliers(lines_theta, 0.5)
    if len(filter_lines_theta) == 0:
        filter_lines_theta = reject_outliers(lines_theta, 1.0)

    angle = -np.mean(filter_lines_theta)
    return 0 if np.isnan(angle) else angle


def cut_black_area(img):

    """
    Crop out black or empty background after rotation (if present).

    Steps:
        1. Find non-zero pixels and extract bounding box coordinates.
        2. Return the coordinates of the bounding box to crop the image.
    """
    mask = img > 0
    mask = mask.all(2)
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return x0, x1, y0, y1


def rotate_image(img, theta):
    """
    Rotate the image by a given angle (in radians), expanding canvas as needed.

    """
    angle_deg = theta * 180 / math.pi
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv.warpAffine(img, M, (new_w, new_h), flags=cv.INTER_LINEAR, borderValue=(0, 0, 0))
    return rotated


def resize_or_pad(img, target_size=(768, 1024)):
    """
    Resize the image to target_size, preserving aspect ratio and padding with black.

    """
    th, tw = target_size
    h, w = img.shape[:2]

    scale = min(tw / w, th / h)
    nh, nw = int(h * scale), int(w * scale)
    img_resized = cv.resize(img, (nw, nh), interpolation=cv.INTER_AREA)

    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left

    img_padded = cv.copyMakeBorder(
        img_resized, top, bottom, left, right,
        borderType=cv.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return img_padded


def save_hough_transform(image_path, dst_folder):
    """
    remove black border (if any) from padded images, and save the processed image to the destination folder.

    """
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    x0, x1, y0, y1 = cut_black_area(img)
    cropped = img[x0:x1, y0:y1]
    theta = get_hough_angle(cropped)
    img_color = skimage.io.imread(image_path)
    rotated = rotate_image(img_color, theta)

    resized_padded = resize_or_pad(rotated, target_size=(768, 1024))

    output_path = os.path.join(dst_folder, os.path.splitext(os.path.basename(image_path))[0] + '.png')
    skimage.io.imsave(output_path, resized_padded)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("hough_transform.py")
    parser.add_argument("--dest", help="Destination folder path for processed images", type=str, required=True)
    parser.add_argument("--src", help="Source folder path for input images (PNG only)", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    file_paths = glob.glob(os.path.join(args.src, '*.png'))

    for file_path in file_paths:
        save_hough_transform(file_path, args.dest)
