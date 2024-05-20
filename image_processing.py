import base64
import os
import tempfile
import rawpy
import cv2
import numpy as np
from nicegui import ui
from config import *
import gphoto2 as gp


def capture_photo():
    global camera, context
    file_path = camera.capture(gp.GP_CAPTURE_IMAGE)
    target = os.path.join(tempfile.gettempdir(), file_path.name)
    camera_file = camera.file_get(file_path.folder, file_path.name, gp.GP_FILE_TYPE_NORMAL)
    camera_file.save(target)
    return target


def process_raw_image(file_path):
    with rawpy.imread(file_path) as raw:
        rgb = raw.postprocess()
    bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr_image


def capture_and_process_image():
    set_camera_connection_lock(True)
    try:
        file_path = capture_photo()
        bgr_image = process_raw_image(file_path)
    finally:
        set_camera_connection_lock(False)
    return bgr_image


def display_image(image):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    output_width = 600
    output_height = int(output_width / aspect_ratio)

    resized_image = cv2.resize(image, (output_width, output_height))
    compress_quality = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, compressed_image = cv2.imencode('.jpg', resized_image, compress_quality)
    img_data = base64.b64encode(compressed_image).decode('utf-8')
    img_str = f"data:image/jpeg;base64,{img_data}"

    def update_image():
        image_display.set_source(img_str)

    update_ui_safely(main_container, update_image)


def capture_photo_and_display():
    image = capture_and_process_image()
    display_image(image)


def locate_elements(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img_with_keypoints, keypoints, descriptors


def preprocess_image(image):
    processed_image = cv2.GaussianBlur(image, (5, 5), 0)
    return processed_image


def analyze_sharpness(image, keypoints):
    sharpness_scores = [kp.response for kp in keypoints]
    average_sharpness = np.mean(sharpness_scores)
    return average_sharpness


def analyze_vignetting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    corners = np.concatenate(
        [gray[:h // 4, :w // 4].flatten(), gray[:h // 4, -w // 4:].flatten(), gray[-h // 4:, :w // 4].flatten(),
         gray[-h // 4:, -w // 4:].flatten()])
    center_mean = np.mean(center)
    corners_mean = np.mean(corners)
    vignetting = corners_mean - center_mean
    return vignetting


def analyze_psf(image, keypoints):
    psf_values = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        psf_values.append(np.sum(image[y - 2:y + 3, x - 2:x + 3]))
    average_psf = np.mean(psf_values)
    return average_psf


def analyze_bokeh(image, keypoints):
    bokeh_values = []
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        bokeh_values.append(np.std(image[y - 2:y + 3, x - 2:x + 3]))
    average_bokeh = np.mean(bokeh_values)
    return average_bokeh


def analyze_image():
    image = capture_and_process_image()
    preprocessed_image = preprocess_image(image)
    img_with_keypoints, keypoints, descriptors = locate_elements(preprocessed_image)

    sharpness = analyze_sharpness(img_with_keypoints, keypoints)
    vignetting = analyze_vignetting(img_with_keypoints)
    psf = analyze_psf(img_with_keypoints, keypoints)
    bokeh = analyze_bokeh(img_with_keypoints, keypoints)

    result = f"Sharpness: {sharpness}, Vignetting: {vignetting}, PSF: {psf}, Bokeh: {bokeh}"
    update_ui_safely(main_container, lambda: ui.notify(result, duration=10))
