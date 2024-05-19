import cv2
from matplotlib import pyplot as plt

def save_and_display_image(image, output_path):
    cv2.imwrite(output_path, image)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    

