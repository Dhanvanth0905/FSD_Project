from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import cv2
import numpy as np
from matplotlib import pyplot as plt
import base64
import os
from django.conf import settings


def index(request):
    if request.method == 'POST' and 'image' in request.FILES:
        # Handle uploaded image
        image_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        uploaded_image_url = fs.url(filename)

        # Get segmentation method choice
        segmentation_method = request.POST.get('segmentation_method')

        # Get visualization choice
        visualization_choice = request.POST.get('visualization_choice')

        # Perform image segmentation and visualization
        original_image, segmented_image, visualization = perform_segmentation(
            uploaded_image_url, segmentation_method, visualization_choice)

        return render(request, 'index.html', {
            'uploaded_image_url': uploaded_image_url,
            'original_image': original_image,
            'segmented_image': segmented_image,
            'visualization': visualization
        })

    return render(request, 'index.html')



def perform_segmentation(image_url, segmentation_method, visualization_choice):
    # Load the image
    image_path = os.path.join(settings.MEDIA_ROOT, image_url[1:])
    print("Image path:", image_path)  # Debug output
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Failed to load the image.")
    image = np.array(image)

    # Perform the selected segmentation method
    if segmentation_method == 'edge':
        segmented_image = perform_edge_based_segmentation(image)
    elif segmentation_method == 'threshold':
        segmented_image = perform_threshold_based_segmentation(image)
    elif segmentation_method == 'region':
        segmented_image = perform_region_based_segmentation(image)
    elif segmentation_method == 'cluster':
        segmented_image = perform_cluster_based_segmentation(image)
    elif segmentation_method == 'watershed':
        segmented_image = perform_watershed_segmentation(image)
    else:
        # Default to edge-based segmentation if no method is selected
        segmented_image = perform_edge_based_segmentation(image)

    # Perform the selected visualization
    visualization = None
    if visualization_choice == 'overlay':
        visualization = get_overlay(segmented_image, image)  # Pass the original image as well
    elif visualization_choice == 'color_map':
        visualization = get_color_map(segmented_image)
    elif visualization_choice == 'contour':
        visualization = get_contour_visualization(segmented_image)
    elif visualization_choice == 'heatmap':
        visualization = get_heatmap_visualization(segmented_image)

    # Convert images to base64 format for displaying in HTML
    _, original_image = cv2.imencode('.png', image)
    original_image_base64 = base64.b64encode(original_image).decode('utf-8')

    _, segmented_image = cv2.imencode('.png', segmented_image)
    segmented_image_base64 = base64.b64encode(segmented_image).decode('utf-8')

    _, visualization = cv2.imencode('.png', visualization)
    visualization_base64 = base64.b64encode(visualization).decode('utf-8')

    return original_image_base64, segmented_image_base64, visualization_base64

def perform_edge_based_segmentation(image):
    # Perform edge-based segmentation using Canny edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def perform_threshold_based_segmentation(image):
    # Perform threshold-based segmentation using global thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return thresholded


def perform_region_based_segmentation(image):
    # Perform region-based segmentation using GrabCut algorithm
    mask = np.zeros(image.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented = image * mask[:, :, np.newaxis]
    return segmented


def perform_cluster_based_segmentation(image):
    # Perform cluster-based segmentation using K-means clustering
    reshaped_image = image.reshape((-1, 3))
    reshaped_image = np.float32(reshaped_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(reshaped_image, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented = centers[labels.flatten()]
    segmented = segmented.reshape(image.shape)
    return segmented


def perform_watershed_segmentation(image):
    # Perform watershed segmentation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist_transform = cv2.distanceTransform(thresholded, cv2.DIST_L2, 3)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[thresholded == 255] = 0
    markers = cv2.watershed(image, markers)
    segmented = np.zeros(image.shape, dtype=np.uint8)
    segmented[markers > 1] = [0, 0, 255]  # Set segmented regions to red color
    return segmented



def get_overlay(segmented_image, original_image):
    if segmented_image.shape != original_image.shape:
        segmented_image = cv2.resize(segmented_image, (original_image.shape[1], original_image.shape[0]))

    if len(segmented_image.shape) == 2:
        segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)

    # Perform overlay visualization
    overlay = cv2.addWeighted(segmented_image, 0.7, original_image, 0.3, 0)
    return overlay




def get_color_map(segmented_image):
    # Perform color mapping visualization
    color_map = cv2.applyColorMap(segmented_image, cv2.COLORMAP_JET)
    return color_map


def get_contour_visualization(segmented_image):
    # Convert the segmented image to BGR color format
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)

    # Perform contour visualization
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_visualization = cv2.drawContours(segmented_image.copy(), contours, -1, (0, 255, 0), 2)
    return contour_visualization



def get_heatmap_visualization(segmented_image):
    # Convert the segmented image to BGR color format
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_GRAY2BGR)

    # Perform heatmap visualization
    heatmap_visualization = cv2.applyColorMap(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)
    return heatmap_visualization





def reset_segmentation():
    global original_image, segmented_image, histogram_plots
    original_image = None
    segmented_image = None
    histogram_plots = None