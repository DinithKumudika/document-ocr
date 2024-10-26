import tempfile
import cv2
import os
import numpy as np
from PIL import Image

temp_path = "C:/Users/dinit/Documents/ocr_rnd/temp"

def list_images_in_folder(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.svg']
    files = os.listdir(folder_path)
    images = [file for file in files if os.path.splitext(file)[1].lower() in image_extensions]
    return images

def print_images_with_numbers(images):
    for idx, image in enumerate(images, 1):
        print(f"{idx}: {image}")
        
def get_image_by_number(images):
    while True:
        try:
            selected_number = int(input("Enter the number of the image you want to select: "))
            if 1 <= selected_number <= len(images):
                selected_image = images[selected_number - 1]
                return selected_image
            else:
                print(f"Please enter a number between 1 and {len(images)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            

# TODO: update noise removal 
def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=3)
    
    kernel = np.ones((1,1), np.uint8)
    image = cv2.erode(image, kernel, iterations=3)
    
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    
    return image



def erode_image(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image

def dilate_image(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def get_sections(image, image_name:str)->list:
    new_image = image.copy()
    im_h, im_w, im_d = image.shape
    section_height = im_h // 6
    sections = []
    for i in range(4):
        start_row = i * section_height
        end_row = start_row + section_height
    
        # Slice the image
        section = new_image[start_row:end_row, :]
        sections.append(section)
    
    last_section = new_image[4* section_height:, :]
    sections.append(last_section)
    
    for i, section in enumerate(sections):
        image_name = image_name.split('.')[0]
        cv2.imwrite(f'{temp_path}/sections/{image_name}-section_{i+1}.jpg', section)
        
    return sections

def draw_bounding_box(image, image_name:str):
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = image.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    # cv2.imwrite(f"{temp_path}/sections/{image_name}-blur.jpg", blur)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite(f"{temp_path}/sections/{image_name}-thresh.jpg", thresh)

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = lambda x: cv2.boundingRect(x)[1])
    # print(f"writing output to {temp_path}/contours.txt...")
    # text_file = open(rf"{temp_path}/contours.txt", "w+", encoding="utf-8")
    
    selected_contours = []
    
    for i, c in enumerate(contours):
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        # text_file.write(f"contour {i+1} => height = {h}, width = {w}\n")
        if 130> h > 70 and w >  150:
            print(f"contour {i+1} => height = {h}, width = {w}")
            cv2.rectangle(newImage,(x,y),(x+w,y+h),(0,0,255), 3)
            selected_contours.append((x,y,w,h))
            label_position = (x, y - 10)  # Adjust the position as needed
            cv2.putText(newImage, f"{h} x {w}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 1)
            
            

    # Find largest contour and surround in min area box
    # largestContour = contours[0]
    # print (len(contours))
    # minAreaRect = cv2.minAreaRect(largestContour)
    
    return newImage, selected_contours

def scale_image(pil_image):
    length_x, width_y = pil_image.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = pil_image.resize(size, Image.LANCZOS)

    return im_resized

    

def get_skew_angle(image):
    _, min_area_rect = draw_bounding_box(image)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = min_area_rect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle

# Rotate the image around its center
def rotate_image(image, angle: float):
    newImage = image.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage

def draw_border(image):
    # Prep image, copy, convert to gray scale, blur, and threshold
    new_image = image.copy()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt_sorted = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour
    border_contour = cnt_sorted[1]
    rect = cv2.boundingRect(border_contour)
    x,y,w,h = rect
    cv2.rectangle(new_image,(x,y),(x+w,y+h),(0,0,255), 2)
    
    return new_image, rect


# remove borders from the image
def remove_borders(image):
    # Prep image, copy, convert to gray scale, blur, and threshold
    new_image = image.copy()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt_sorted = sorted(contours, key = cv2.contourArea, reverse = True)

    # Find largest contour
    border_contour = cnt_sorted[1]
    rect = cv2.boundingRect(border_contour)
    x,y,w,h = rect
    crop = new_image[y:y+h, x:x+w]
    return crop