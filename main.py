from PIL import Image
import pytesseract
import os
import time
import cv2
import numpy as np
from utils import list_images_in_folder, print_images_with_numbers, get_image_by_number, scale_image, noise_removal, erode_image, dilate_image, draw_bounding_box, remove_borders, get_sections

output_path = "C:/Users/dinit/Documents/ocr_rnd/output"
input_path = "C:/Users/dinit/Documents/ocr_rnd/input"
temp_path = "C:/Users/dinit/Documents/ocr_rnd/temp"

def preprocess_roi(roi, roi_id, image_name = None):
    if roi is None:
        print("Error: Unable to read image")
        return None
        
    try:
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{roi_id}-grayscale.jpg", roi_gray)

        # Normalize the image
        roi_norm = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{roi_id}-normalize.jpg", roi_norm)

        # Median blur for noise reduction
        roi_median_blur = cv2.medianBlur(roi_norm, 3)
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{roi_id}-mdedian-blur.jpg", roi_median_blur)

        # Gaussian blur for further noise reduction
        roi_blur = cv2.GaussianBlur(roi_median_blur, (5, 5), 0.5)
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{roi_id}-gaussian-blur.jpg", roi_blur)

        # Sharpen the image
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        roi_sharp = cv2.filter2D(roi_blur, -1, sharpen_kernel)
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{roi_id}-sharp.jpg", roi_sharp)

        # Apply gamma correction
        gamma = 2.0
        roi_gamma_correct = np.array(255 * (roi_sharp / 255) ** gamma, dtype='uint8')
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{roi_id}-gamma.jpg", roi_gamma_correct)
        

        # Apply Otsu's thresholding
        _, roi_threshold = cv2.threshold(roi_gamma_correct, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{roi_id}-threshold.jpg", roi_threshold)
        
        kernel = np.ones((3, 3), np.uint8)
        roi_morph = cv2.morphologyEx(roi_threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{roi_id}-morph.jpg", roi_morph)

        print('ROI preprocessing completed successfully.')

        return roi_morph

    except Exception as e:
        print('Error occurred during image preprocessing:', e)
        return None
            
def preprocess_image(pil_image, image_name = None):
    # rotate the image if needed
    if pil_image.width < pil_image.height:
            pil_image = pil_image.rotate(-90, expand=True)
            
    im = scale_image(pil_image)
    im.save(f"{temp_path}/{image_name.split('.')[0]}-scaled.jpg", dpi=(300,300))
    cv_image = np.array(im)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(f"{temp_path}/{image_name.split('.')[0]}-scaled.jpg", cv_image)
    
    # border_image, _ = draw_border(cv_image)
    # cv2.imwrite(f"{temp_path}/{image_name.split('.')[0]}-border.jpg", border_image)
    
    # remove border from the image
    no_border_image = remove_borders(cv_image)
    cv2.imwrite(f"{temp_path}/{image_name.split('.')[0]}-no-border.jpg", no_border_image)
    
    sections = get_sections(no_border_image, image_name)
    
    roi_sections = []
    
    for i, section  in enumerate(sections):
        bounding_box_image, contours = draw_bounding_box(section, f"{image_name.split('.')[0]}-section_{i+1}")
        cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-contours.jpg", bounding_box_image)
        
        # crop region of interests
        for j, contour in enumerate(contours):
            x,y,w,h = contour
            roi = section[y:y+h, x:x+w]
            roi_id = f"{i+1}-{j+1}"
            
            print(f"roi shape for section {roi_id} : {roi.shape[0]} x {roi.shape[1]}")
            
            # Normalize the image
            norm_img = np.zeros((roi.shape[0], roi.shape[1]))
            roi_norm = cv2.normalize(roi, norm_img, 0, 255, cv2.NORM_MINMAX)
            
            # cv2.imshow(f"roi normalized {i+1}-{j+1}", roi_norm)
            # cv2.waitKey(0)
            
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-normalized.jpg", roi_norm)
            
            
            # Binarization
            roi_gray = cv2.cvtColor(roi_norm, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-grayscale.jpg", roi_gray)
            
            # Black and White
            roi_thresh_binary = cv2.threshold(roi_gray, 130, 255, cv2.THRESH_BINARY)[1]
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-binary.jpg", roi_thresh_binary)
            
            roi_thresh_binary_inv = cv2.threshold(roi_gray, 125, 255, cv2.THRESH_BINARY_INV)[1]
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-binary-inv.jpg", roi_thresh_binary_inv)
            
            roi_median_blur = cv2.medianBlur(roi_gray,5)
            
            # Adaptive Mean Thresholding
            roi_thresh_mean = cv2.adaptiveThreshold(roi_median_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-adaptive-mean.jpg", roi_thresh_mean)
            
            # Adaptive Gaussian Thresholding
            roi_thresh_gaussian = cv2.adaptiveThreshold(roi_median_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-adaptive-gaussian.jpg", roi_thresh_gaussian)
            
            # Noise Removal
            roi_no_noise = noise_removal(roi_thresh_binary)
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-no-noise.jpg", roi_no_noise)
            
            # erosion
            roi_erode = erode_image(roi_no_noise)
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-eroded.jpg", roi_erode)
            
            # dilation
            roi_dilate = dilate_image(roi_no_noise)
            cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-section_{i+1}-{j+1}-dilated.jpg", roi_dilate)
            
            # preprocessed_roi = preprocess_roi(roi, roi_id, image_name=image_name)
            roi_sections.append(roi_thresh_gaussian)
            
        # Normalize the image
        # norm_image = np.zeros((section.shape[0], section.shape[1]))
        # normalized_image = cv2.normalize(section, norm_image, 0, 255, cv2.NORM_MINMAX)
        # cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-part_{i+1}-normalized.jpg", normalized_image)
        
    
        # inverte the image
        # inverted_image = cv2.bitwise_not(normalized_image)
        # cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-part_{i+1}-inverted.jpg", inverted_image)
    
        # Binarization
        # grayscale_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-part_{i+1}-grayscale.jpg", grayscale_image)
    
        # thrsh, bw_image = cv2.threshold(grayscale_image, 130, 255, cv2.THRESH_BINARY_INV)
        # cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-part_{i+1}-bw.jpg", bw_image)
    
        # Noise Removal
        # no_noise = noise_removal(bw_image)
        # cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-part_{i+1}-no-noise.jpg", no_noise)

        # eroded_image = erode_image(no_noise)
        # cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-part_{i+1}-eroded.jpg", eroded_image)
    
        # dilated_image = dilate_image(no_noise)
        # cv2.imwrite(f"{temp_path}/sections/{image_name.split('.')[0]}-part_{i+1}-dilated.jpg", dilated_image)
        # cv2.destroyAllWindows()
    
    return roi_sections

def extract_text(image_name):
    translated_text = ""
    
    im = Image.open(f"{temp_path}/sections/{image_name}")
    
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    translated_text += pytesseract.image_to_string(im, lang='sin')
    print(translated_text)
        
    output_text_file_path = f"{output_path}/sections/{image_name.split('.')[0]}.txt"
    print(f"writing output to {output_text_file_path}...")
    text_file = open(rf"{output_text_file_path}", "w+", encoding="utf-8")
                
    print("Cleaning up the translation.......")
    translated_text = translated_text.replace("\n\n", "new_para")
    translated_text = translated_text.rstrip("\n")
    translated_text = translated_text.replace("\n", " ")
    translated_text = translated_text.replace("\u200c", " ")
    translated_text = translated_text.replace("\u200d", "")
    translated_text = translated_text.replace("new_para", "\n\n")
    text_file.write(translated_text)
    print(f"Text is completely translated and is saved in {output_path}......")
    
def main2():
    while True:
        if not os.path.exists(f"{output_path}/sections"):
            os.makedirs(output_path)
            print("Directory was not found.")
            print(f"Creating output directory at {output_path}")
            time.sleep(5)
            print("Restart Application and Put your stuff in the relevant folders.......")
            time.sleep(5)
            exit()

        items_exist = False
        
        images = list_images_in_folder(input_path)
        print_images_with_numbers(images)
        input_image = get_image_by_number(images)
        
        print(f"selected input image-: {input_image}")
        items_exist = True
        
        pil_image = Image.open(f"{input_path}/{input_image}")
        
        print("pre processing the image....")
        
        roi_images = preprocess_image(pil_image, image_name=input_image)
        
        print("---------------Translating------------------")
                
        if items_exist == True:
            for idx, image in enumerate(roi_images):
                translated_text = ""
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
                custom_config = r'--oem 3 --psm 6'
                translated_text += pytesseract.image_to_string(image, config=custom_config, lang='sin')
                print(translated_text)
                output_text_file_path = f"{output_path}/{input_image.split('.')[0]}-{idx+1}.txt"
                print(f"writing output to {output_text_file_path}...")
                text_file = open(rf"{output_text_file_path}", "w+", encoding="utf-8")
                
                print("Cleaning up the translation.......")
                translated_text = translated_text.replace("\n\n", "new_para")
                translated_text = translated_text.rstrip("\n")
                translated_text = translated_text.replace("\n", " ")
                translated_text = translated_text.replace("\u200c", " ")
                translated_text = translated_text.replace("\u200d", "")
                translated_text = translated_text.replace("new_para", "\n\n")
                text_file.write(translated_text)
                print(f"Text is completely translated and is saved in {output_path}......")
            time.sleep(5)
            exit()
        else:
            print("Items not found inside the Input Folder.......")
            time.sleep(5)
            exit()

def main():
    while True:
        if not os.path.exists(f"{output_path}/sections"):
            os.makedirs(output_path)
            print("Directory was not found.")
            print(f"Creating output directory at {output_path}")
            time.sleep(5)
            print("Restart Application and Put your stuff in the relevant folders.......")
            time.sleep(5)
            exit()

        items_exist = False
        
        images = list_images_in_folder(input_path)
        print_images_with_numbers(images)
        input_image = get_image_by_number(images)
        
        print(f"selected input image-: {input_image}")
        items_exist = True
        
                
        pil_image = Image.open(f"{input_path}/{input_image}")
        # new_size = tuple(1 * x for x in im.size)  # Increasing this 1*x part to different integer values may increase the accuracy sometimes
        # im_resize = im.resize(new_size, Image.LANCZOS)
        
        print("pre processing the image....")
        
        roi_images = preprocess_image(pil_image, image_name=input_image)
        
        print("Select the temp image that need to OCR from below....")
        
        images = list_images_in_folder(f"{temp_path}/sections")
        print_images_with_numbers(images)
        input_image = get_image_by_number(images)
        
        print("---------------Translating------------------")
        
        # im = Image.open(f"{temp_path}/sections/{input_image}")
            
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # translated_text += pytesseract.image_to_string(im, lang='sin')
        # print(translated_text)
        
        # output_text_file_path = f"{output_path}/{input_image.split('.')[0]}.txt"
        # print(f"writing output to {output_text_file_path}...")
        # text_file = open(rf"{output_text_file_path}", "w+", encoding="utf-8")
                
        if items_exist == True:
            # print("Cleaning up the translation.......")
            # translated_text = translated_text.replace("\n\n", "new_para")
            # translated_text = translated_text.rstrip("\n")
            # translated_text = translated_text.replace("\n", " ")
            # translated_text = translated_text.replace("\u200c", " ")
            # translated_text = translated_text.replace("\u200d", "")
            # translated_text = translated_text.replace("new_para", "\n\n")
            # text_file.write(translated_text)
            # print(f"Text is completely translated and is saved in {output_path}......")
            extract_text(input_image)
            time.sleep(5)
            exit()
        else:
            print("Items not found inside the Input Folder.......")
            time.sleep(5)
            exit()
            
if __name__ == "__main__":
    main2()