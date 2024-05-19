import cv2

def combine_images(images):
    combined_image = cv2.vconcat(images)
    return combined_image


