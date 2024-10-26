from enum import Enum
import fillpdf
from fillpdf import fillpdfs
import json

from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from os import path
import cv2
import numpy as np
import easyocr
import json

input_path = "C:/Users/dinit/Documents/ocr_rnd/input"
poppler_path = "C:/Users/dinit/Documents/ocr_rnd/poppler-24.07.0/Library/bin"
output_path = "C:/Users/dinit/Documents/ocr_rnd/output"
temp_path = "C:/Users/dinit/Documents/ocr_rnd/temp"

class TermsOfDelivery(Enum):
    FOB = "FOB"
    CFR = "CFR"
    CIF = "CIF"
    FCA = "FCA"
    CPT = "CPT"
    CIP = "CIP"
     

def flatten_pdf(pdf_path):
    file_name = path.basename(pdf_path)
    output_file = f"{input_path}/{path.splitext(file_name)[0]}-flatten.pdf"
    fillpdfs.flatten_pdf(pdf_path, output_file, as_images=False)
    
    return output_file


def get_form_field_values(pdf_path):
    form_field_values = {}
    form_fields_page_1 = fillpdfs.get_form_fields(pdf_path, sort=False, page_number=1)
    
    print(f"form fields from page 1: {form_fields_page_1}")
    
    form_field_values['transferable_credit'] = True if form_fields_page_1['untitled1'] == 'Yes' else False
    form_field_values['standby_credit'] = True if form_fields_page_1['untitled2'] == 'Yes' else False
    
    form_field_values['date_of_application'] = {}
    form_field_values['date_of_application']['date'] = int(form_fields_page_1['untitled4'])
    form_field_values['date_of_application']['month'] = form_fields_page_1['untitled5']
    form_field_values['date_of_application']['year'] = int(form_fields_page_1['untitled6'])
    
    form_field_values['date_of_expiry'] = {}
    form_field_values['date_of_expiry']['date'] = int(form_fields_page_1['untitled7'])
    form_field_values['date_of_expiry']['month'] = form_fields_page_1['untitled8']
    form_field_values['date_of_expiry']['year'] = int(form_fields_page_1['untitled9'])
    
    form_field_values['place_of_expiry'] = form_fields_page_1['untitled11']
    
    form_field_values['applicant_details'] = {}
    form_field_values['applicant_details']['name'] = form_fields_page_1['untitled12'].split('\r')[0]
    form_field_values['applicant_details']['address'] = form_fields_page_1['untitled12'].split('\r')[1]
    
    form_field_values['benificiary_details'] = {}
    form_field_values['benificiary_details']['name'] = form_fields_page_1['untitled13'].split('\r')[0]
    form_field_values['benificiary_details']['address'] = form_fields_page_1['untitled13'].split('\r')[1]
    form_field_values['benificiary_details']['contact_details'] = form_fields_page_1['untitled14']
    
    form_field_values['curency_n_amount'] = form_fields_page_1['untitled15']
    
    form_field_values['tollerance'] = {}
    form_field_values['tollerance']['percent_1'] = int(form_fields_page_1['untitled16'])
    form_field_values['tollerance']['percent_2'] = int(form_fields_page_1['untitled17'])
    form_field_values['tollerance']['type'] = form_fields_page_1['untitled18']
    
    form_field_values['credit_available'] = {}
    
    if(form_fields_page_1['untitled19'] == 'Yes'):
        form_field_values['credit_available']['from_any_bank'] = True
        form_field_values['credit_available']['bank_location'] = form_fields_page_1['untitled20']
    if(form_fields_page_1['untitled21'] == 'Yes'):
        form_field_values['credit_available']['from_any_bank'] = False
        form_field_values['credit_available']['bank_name'] = form_fields_page_1['untitled22']
        form_field_values['credit_available']['bank_location'] = form_fields_page_1['untitled23']
    
    if(form_fields_page_1['untitled24'] == 'Yes'):
        form_field_values['drafts_at'] = 'sight'
    if(form_fields_page_1['untitled25'] == 'Yes'):
        form_field_values['drafts_at'] = f"{form_fields_page_1['untitled26']} after {form_fields_page_1['untitled27']}"
    
    if(form_fields_page_1['untitled28'] == 'Yes'):
        form_field_values['partial_shipment'] = True
    if(form_fields_page_1['untitled29'] == 'Yes'):
         form_field_values['partial_shipment'] = False
         
    if(form_fields_page_1['untitled30'] == 'Yes'):
        form_field_values['trans_shipment'] = True
    if(form_fields_page_1['untitled31'] == 'Yes'):
         form_field_values['trans_shipment'] = False
         
    form_field_values['shipment'] = {}
    form_field_values['shipment']['44E'] = form_fields_page_1['untitled32']
    form_field_values['shipment']['44F'] = form_fields_page_1['untitled33']
    
    form_field_values['latest_shipment_date'] = {}
    form_field_values['latest_shipment_date']['date'] = int(form_fields_page_1['untitled34'])
    form_field_values['latest_shipment_date']['month'] = form_fields_page_1['untitled35']
    form_field_values['latest_shipment_date']['year'] = int(form_fields_page_1['untitled36'])
    
    form_field_values['description_of_goods'] = form_fields_page_1['untitled37']
    
    form_field_values['terms_of_delivery'] = {}
    
    form_field_values['terms_of_delivery']['sea_shipment'] = []
    if(form_fields_page_1['untitled38'] == 'Yes'):
        form_field_values['terms_of_delivery']['sea_shipment'].append('FOB')
    if(form_fields_page_1['untitled39'] == 'Yes'):
        form_field_values['terms_of_delivery']['sea_shipment'].append('CFR')
    if(form_fields_page_1['untitled40'] == 'Yes'):
        form_field_values['terms_of_delivery']['sea_shipment'].append('CIF')
    
    form_field_values['terms_of_delivery']['all'] = []
    if(form_fields_page_1['untitled41'] == 'Yes'):    
        form_field_values['terms_of_delivery']['all'].append('FCA')
    if(form_fields_page_1['untitled42'] == 'Yes'):    
        form_field_values['terms_of_delivery']['all'].append('CPT')
    if(form_fields_page_1['untitled43'] == 'Yes'):    
        form_field_values['terms_of_delivery']['all'].append('CIP')
    if(form_fields_page_1['untitled44'] == 'Yes'):
        form_field_values['terms_of_delivery']['all'].append(form_fields_page_1['untitled45'])
        
    form_field_values['documents_required'] = {}
    
    if(form_fields_page_1['untitled46'] == 'Yes'):
        form_field_values['documents_required']['option_1'] = {}
        form_field_values['documents_required']['option_1']['checked'] = True
        form_field_values['documents_required']['option_1']['invoice_copies'] = int(form_fields_page_1['untitled47'])
        form_field_values['documents_required']['option_1']['goods_origin'] = form_fields_page_1['untitled48']
        form_field_values['documents_required']['option_1']['hs_code'] = form_fields_page_1['untitled49']
     
    if(form_fields_page_1['untitled50'] == 'Yes'):
        form_field_values['documents_required']['option_2'] = {}
        form_field_values['documents_required']['option_2']['checked'] = True
        form_field_values['documents_required']['option_2']['freight_status'] = form_fields_page_1['untitled51']
    
    if(form_fields_page_1['untitled52'] == 'Yes'):
        form_field_values['documents_required']['option_3'] = {}
        form_field_values['documents_required']['option_3']['checked'] = True
        form_field_values['documents_required']['option_3']['air_waybill_freight_status'] =  form_fields_page_1['untitled53']
        
    if(form_fields_page_1['untitled54'] == 'Yes'):    
        form_field_values['documents_required']['option_4'] = {}
        form_field_values['documents_required']['option_4']['checked'] = True
        form_field_values['documents_required']['option_4']['institute_war_clauses'] = form_fields_page_1['untitled55']
        form_field_values['documents_required']['option_4']['institute_strike_clauses'] = form_fields_page_1['untitled56']
        form_field_values['documents_required']['option_4']['insurance_term'] = form_fields_page_1['untitled57']
        form_field_values['documents_required']['option_4']['insurance_percentage'] = form_fields_page_1['untitled58']
    if(form_fields_page_1['untitled59'] == 'Yes'):
        form_field_values['documents_required']['option_5'] = {}
        form_field_values['documents_required']['option_5']['checked'] = True
        form_field_values['documents_required']['option_5']['packing_list_copies'] = form_fields_page_1['untitled60']    
    if(form_fields_page_1['untitled61'] == 'Yes'):
        form_field_values['documents_required']['option_6'] = {}
        form_field_values['documents_required']['option_6']['checked'] = True
        form_field_values['documents_required']['option_6']['weight_list_copies'] = form_fields_page_1['untitled62'] 
    
    form_field_values['additional_instructions'] = {}
    
    form_field_values['additional_instructions']['option_1'] = {}
    if(form_fields_page_1['untitled63'] == 'Yes'):
        form_field_values['additional_instructions']['option_1']['checked'] = True
        form_field_values['additional_instructions']['option_1']['send_copy_by'] = form_fields_page_1['untitled66']
        form_field_values['additional_instructions']['option_1']['days_after_shipment'] = int(form_fields_page_1['untitled67'])
        form_field_values['additional_instructions']['option_1']['accompanying_document'] = form_fields_page_1['untitled68']
    else:
        form_field_values['additional_instructions']['option_1']['checked'] = False
        
    form_field_values['additional_instructions']['option_2'] = {}    
    if(form_fields_page_1['untitled64'] == 'Yes'):
        form_field_values['additional_instructions']['option_2']['checked'] = True
        form_field_values['additional_instructions']['option_2']['open_cover_no'] = form_fields_page_1['untitled69']
    else:
        form_field_values['additional_instructions']['option_2']['checked'] = False
    
    form_field_values['additional_instructions']['option_3'] = {}
    if(form_fields_page_1['untitled65'] == 'Yes'):
        form_field_values['additional_instructions']['option_3']['checked'] = True
    else:
        form_field_values['additional_instructions']['option_3']['checked'] = False
    
    form_field_values['period_for_presentation'] = f"{form_fields_page_1['untitled70']} days after "  
    
    if(form_fields_page_1['untitled3'] == 'Yes'):
        form_field_values['period_for_presentation'] += "shipment date"
    
    if(form_fields_page_1['untitled71'] == 'Yes'):
        form_field_values['period_for_presentation'] += "date of GRN but within the validity of the credit"
    
    form_field_values['confirmation'] = {}
    if(form_fields_page_1['untitled72'] == 'Yes' and form_fields_page_1['untitled73'] == 'Off'):
        form_field_values['confirmation']['required'] = True
    if(form_fields_page_1['untitled72'] == 'Off' and form_fields_page_1['untitled73'] == 'Yes'):
        form_field_values['confirmation']['required'] = False
    
    if(form_fields_page_1['untitled74'] == 'Yes'):
        form_field_values['confirmation']['charges_on'] = 'Applicant'
        
    if(form_fields_page_1['untitled75'] == 'Yes'):
        form_field_values['confirmation']['charges_on'] = 'Beneficiary'
    
    if(form_fields_page_1['untitled76'] == 'Yes'):
        form_field_values['charges_outside_issuing_bank'] = 'Applicant'
    if(form_fields_page_1['untitled77'] == 'Yes'):    
        form_field_values['charges_outside_issuing_bank'] = 'Beneficiary'
    
    form_field_values['advising_bank'] = form_fields_page_1['untitled78']
    
    if(form_fields_page_1['untitled79'] == 'Yes'):
        form_field_values['separate_sheet_is_attached'] = True
    
    if(form_fields_page_1['untitled80'] == 'Yes'):
        form_field_values['book_forward_exchange_months'] = int(form_fields_page_1['untitled81'])
    
    form_field_values['applicant_contact_details'] = form_fields_page_1['untitled82']
    
    form_fields_page_2 = fillpdfs.get_form_fields(pdf_path, sort=False, page_number=2)
    
    print(f"form fields from page 2: {form_fields_page_2}")
    
    form_field_values['account_no'] = form_fields_page_2['untitled83']
    form_field_values['bank_branch'] = form_fields_page_2['untitled84']
    form_field_values['applicant_name'] = form_fields_page_2['untitled85']
    form_field_values['date'] = form_fields_page_2['untitled86']
    
    return form_field_values

def pdf_to_img(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=poppler_path, dpi=300, fmt='png')
    print(f"pdf converted to {len(images)}")
    return images

def save_images(images) -> list:
    images_list = []
    for idx, image in enumerate(images):
        image_path = f"{output_path}/page_{idx+1}.jpg"
        image.save(image_path)
        print(f"page {idx+1} saved to {image_path}")
        print(image_path)
        images_list.append(image_path)
    return images_list

def ocr_image(image, method: str, from_path = False, paragraph = False, gpu = False, numbers_only = False, special_characters = True):
    if method == "easy-ocr":
        reader = easyocr.Reader(['en'], gpu=gpu) # this needs to run only once to load the model into memory
        if numbers_only == True:
            result = reader.readtext(image, paragraph=paragraph, allowlist='0123456789/-')
        elif special_characters == False:
            result = reader.readtext(image, paragraph=paragraph, blocklist='+/}=!@$%^&*{()~`')
        else:
            result = reader.readtext(image, paragraph=paragraph)
        return result
    
    if method == "tesseract":
        if from_path == True:
            image = Image.open(image)
            
        translated_text = ""
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        translated_text += pytesseract.image_to_string(image, lang='eng', config='--psm 6')
        print(translated_text)
                
        print("Cleaning up the translation.......")
        translated_text = translated_text.replace("\n\n", "new_para")
        translated_text = translated_text.rstrip("\n")
        translated_text = translated_text.replace("\n", " ")
        translated_text = translated_text.replace("\u200c", " ")
        translated_text = translated_text.replace("\u200d", "")
        translated_text = translated_text.replace("new_para", "\n\n")
        return translated_text

def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

def postprocess_ocr(ocr_result, distance_threshold=5):
    merged_text = []
    previous_word = ""
    previous_box = None
    
    ocr_result_sorted = sorted(ocr_result, key=lambda x: x[0][0][0])

    for res in ocr_result_sorted:
        text, box = res[1], res[0]

        if previous_box is not None:
            # Calculate horizontal distance between current box and previous box
            distance = box[0][0] - previous_box[1][0]
            if distance < distance_threshold:  # If boxes are close, merge text
                previous_word += text
            else:
                merged_text.append(previous_word)
                previous_word = text
        else:
            previous_word = text
        previous_box = box

    if previous_word:
        merged_text.append(previous_word)

    return " ".join(merged_text)

def text_bounding_box(ocr_res, image):
    for (bbox, text, prob) in ocr_res:
        top_left = tuple([int(val) for val in bbox[0]])
        bottom_right = tuple([int(val) for val in bbox[2]])
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        # Display the detected text next to the bounding box
        cv2.putText(image, text, (top_left[0], top_left[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return image
    
def print_ocr_res(ocr_res):
    for (_, text, score) in ocr_res:
        print(f"Text: {text}")
        print(f"Score: {score}")
        print(" ")

def preprocess_page(img_path):
    img = cv2.imread(img_path)
    file_name = path.basename(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    img_median_blur = cv2.medianBlur(img_norm, 3)
    img_blur = cv2.GaussianBlur(img_median_blur, (5, 5), 0.5)
    
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img_sharp = cv2.filter2D(img_blur, -1, sharpen_kernel)
    
    gamma = 2.0
    img_gamma_correct = np.array(255 * (img_sharp / 255) ** gamma, dtype='uint8')
    
    _, img_threshold = cv2.threshold(img_gamma_correct, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(f"{temp_path}/document/{path.splitext(file_name)[0]}-thresh.jpg", img_threshold)
    
    # kernel = np.ones((3, 3), np.uint8)
    # img_morph = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    # cv2.imwrite(f"{temp_path}/document/{path.splitext(file_name)[0]}-morph-1.jpg", img_morph)
    
    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_morph = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(f"{temp_path}/document/{path.splitext(file_name)[0]}-morph.jpg", img_morph)
    # invert = 255 - img_morph
    
    return img_threshold
        
    # cv2.imshow("threshold image", img_threshold)
    # cv2.waitKey(0)
    
def draw_bounding_boxes(image, detections, threshold=0.25):

    for idx, (bbox, text, score) in enumerate(detections):

        if score > threshold:

            cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)

            cv2.putText(image, f"{idx+1}", tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (255, 0, 0), 2)
    
    return image

def get_contours(processed_image, image):
    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 3))
    dilate = cv2.dilate(processed_image, kernel, iterations=1)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    contours_list = []
    
    for i, c in enumerate(contours):
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        # text_file.write(f"contour {i+1} => height = {h}, width = {w}\n")
        # print(f"contour {i+1} => height = {h}, width = {w}")
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255), 3)
        label_position = (x, y - 10)  # Adjust the position as needed
        cv2.putText(image, f"{i}: {h} x {w}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 1)
    
        contours_list.append({
            "id": i,
            "height": h,
            "width": w,
            "x_coord": x,
            "y_coord": y
        })
    
    return contours_list, image


def get_contours2(processed_image, image):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    dilate = cv2.dilate(processed_image, kernel, iterations=1)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    contours_list = []
    
    for i, c in enumerate(contours):
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        # text_file.write(f"contour {i+1} => height = {h}, width = {w}\n")
        # print(f"contour {i+1} => height = {h}, width = {w}")
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255), 3)
        label_position = (x, y - 10)  # Adjust the position as needed
        cv2.putText(image, f"{i}: {h} x {w}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 1)
    
        contours_list.append({
            "id": i,
            "height": h,
            "width": w,
            "x_coord": x,
            "y_coord": y
        })
    
    return contours_list, image

def get_form_contours(image):
    # Adaptive thresholding for better table detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                            cv2.THRESH_BINARY_INV, 15, 8)

    # Define kernels for horizontal and vertical line detection
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))

    # Apply morphological operations to detect horizontal and vertical lines
    horizontal_lines = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, vertical_kernel)

    # Combine the horizontal and vertical lines
    table_structure = cv2.bitwise_or(horizontal_lines, vertical_lines)

    # Find contours from the combined table structure
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by position, not area
    contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    contours_list = []

    for i, c in enumerate(contours):
        rect = cv2.boundingRect(c)
        x, y, w, h = rect

        # Draw rectangles around detected contours
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        label_position = (x, y - 10)  # Adjust the position as needed
        cv2.putText(image, f"{i}: {h} x {w}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 1)

        # Add contour information to the list
        contours_list.append({
            "id": i,
            "height": h,
            "width": w,
            "x_coord": x,
            "y_coord": y
        })

    return contours_list, image

def get_form_rows(image):
    row_ratios = [
        {
            "id": "40A",
            "ratio": 0.025
        },
        {
            "id": "20",
            "ratio": 0.025
        },
        {
            "id": "31D",
            "ratio": 0.027
        },
        {
            "id": "50",
            "ratio": 0.071
        },
        {
            "id": "59",
            "ratio": 0.083
        },
        {
            "id": "32B",
            "ratio": 0.028
        },
        {
            "id": "41A",
            "ratio": 0.053
        },
        {
            "id": "42C",
            "ratio": 0.025
        },
        {
            "id": "43P",
            "ratio": 0.025
        },
        {
            "id": "43T",
            "ratio": 0.025
        },
        {
            "id": "44",
            "ratio": 0.0563
        },
        {
            "id": "44C",
            "ratio": 0.028
        },
        {
            "id": "45A",
            "ratio": 0.139
        },
        {
            "id": "46A",
            "ratio": 0.230
        },
        {
            "id": "47A",
            "ratio": 0.086
        },
        {
            "id": "48",
            "ratio": 0.033
        },
        {
            "id": "49",
            "ratio": 0.016
        },
        {
            "id": "71B",
            "ratio": 0.026
        }
    ]
    
    rows = []
    
    im_h, im_w = image.shape
    temp = 0
    
    for idx, row_ratio in enumerate(row_ratios):
        row_height = int(im_h * row_ratio["ratio"])
        
        start_height = temp
        end_height = temp + row_height
        row = image[start_height : end_height, :]
        
        rows.append(
            {
                "id": row_ratio['id'],
                "image": row,
                "start_height": start_height,
                "end_height": end_height,
                "end_width": im_w
            }
        )
        
        cv2.imwrite(f"{temp_path}/document/table_rows/row-{row_ratio['id']}.jpg", row)
        
        temp = temp + row_height
        
    return rows

def crop_contours(image, contours_list):
    cropped = []
    for contour in contours_list:
        x, y = contour['x_coord'], contour['y_coord']
        w, h = contour['width'], contour['height']
        cropped_image = image[y:y+h, x:x+w]
        
        cropped.append({
            "id": contour["id"],
            "image": cropped_image,
            "x": contour['x_coord'],
            "y": contour['y_coord'],
            "w": contour['width'],
            "h": contour['height']
        })
        
        cv2.imshow("cropped", cropped_image)        
        cv2.waitKey(0)
        
    return cropped

def get_from_printed(pdf_path):
    form_field_values = get_form_field_values(pdf_path)
    with open("./form_fields.json", "w") as outfile: 
        json.dump(form_field_values, outfile)
    print("wrote output to form_fields.json")
    
def scale_image(image, scale_percent):
    original_width = image.shape[1]
    original_height = image.shape[0]
    
    new_width = int(original_width * scale_percent / 100)
    new_height = int(original_height * scale_percent / 100)
    
    scaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return scaled_image

def get_white_pixel(image):
    return cv2.countNonZero(image)

def is_checked(image):
    white_pixel_count = cv2.countNonZero(image)
    
    if white_pixel_count > 300:
        return True, white_pixel_count
    return False, white_pixel_count

def multiline_to_single_line(ocr_res):
    output_string = ""
    
    for idx, res in enumerate(ocr_res):
        if idx == len(ocr_res) - 1:  # Last element
            output_string += res[1]
        else:
            output_string += str(res[1]) + ", "
    
    return output_string

def dict_to_json(data):
    json_path = "./doc_ocr.json"
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    print(f"dictionary has been converted to JSON and saved to {json_path}")

def extract_bottom_part(original_image_path, threshold_image, output_json):
    file_name = path.basename(original_image_path)
    im_h, im_w = threshold_image.shape
    original_image = cv2.imread(original_image_path)
    
    split_height = int(im_h * 0.81)
    bottom_20 = threshold_image[split_height:, :]
    
    contours, _ = get_contours(bottom_20, original_image.copy()[split_height:, :])
    
    filter_ids = [0, 4, 5, 3]
    filtered_contours = [c for c in contours if c["id"] in filter_ids]
    cropped = crop_contours(bottom_20, filtered_contours)
    
    cropped_0 = [image for image in cropped if image["id"] == 0 ][0]
    cropped_3 = [image for image in cropped if image["id"] == 3 ][0]
    cropped_4 = [image for image in cropped if image["id"] == 4 ][0]
    cropped_5 = [image for image in cropped if image["id"] == 5 ][0]
    
    im_h, im_w = cropped_0['image'].shape
    applicant_contact_detail_area = cropped_0['image'][:, im_w - int(im_w * (1/3)) : im_w]
    # cv2.imwrite(f"{temp_path}/document/{path.splitext(file_name)[0]}-applicant-contact.jpg", applicant_contact_detail_area)
                
    applicant_contact_contours, _ = get_contours(applicant_contact_detail_area, original_image.copy()[split_height:, :]
        [cropped_0["y"]:cropped_0["y"]+cropped_0["h"], cropped_0["x"]:cropped_0["x"]+cropped_0["w"]]
        [:, im_w - int(im_w * (1/3)) : im_w]
    )
    
    filtered_contours = [c for c in applicant_contact_contours if c["id"] in [0]]
    cropped_contact_details = crop_contours(applicant_contact_detail_area, filtered_contours)[0]
    im_h, im_w = cropped_contact_details['image'].shape
    contact_details = cropped_contact_details['image'][:, im_w - int(im_w * 0.65) : im_w]
    ocr_res = ocr_image(contact_details, method="easy-ocr")
    output_json['page_1']['contact_details'] = ocr_res[0][1]
    
    im_h, im_w = cropped_3['image'].shape
    advising_bank_area = cropped_3['image'][:, im_w - int(im_w * 0.59) : im_w]
    # cv2.imwrite(f"{temp_path}/document/{path.splitext(file_name)[0]}-advising-bank.jpg", advising_bank_area)
    ocr_res = ocr_image(cv2.blur(advising_bank_area, (3,3)), method="easy-ocr", paragraph=True)
    print("advising bank: ", ocr_res[0][1])
    output_json['page_1']['advising_bank'] = ocr_res[0][1]
    
    im_h, im_w = cropped_5['image'].shape     
    forward_exchange_check = cropped_5['image'][:, : int(im_w * 0.11)]
    cv2.imshow("forward_exchange_check", forward_exchange_check)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    if(is_checked(forward_exchange_check)[0]):
        output_json['page_1']['forward_exchange_check'] = {}
        output_json['page_1']['forward_exchange_check']['checked'] = True
        forward_exchange_check_months = cropped_5['image'][:, im_w - int(im_w * 0.27) : im_w - int(im_w * 0.18)]
        # cv2.imshow("forward_exchange_check_months", forward_exchange_check_months)
        ocr_res = ocr_image(forward_exchange_check_months, method="easy-ocr")
        output_json['page_1']['forward_exchange_check']['months'] = ocr_res[0][1]
    
    im_h, im_w = cropped_4['image'].shape     
    other_sheets_check = cropped_4['image'][:, : int(im_w * 0.07)]
    if(is_checked(other_sheets_check)[0]):
        output_json['page_1']['seperate_sheets'] = True  
    
    return output_json
    
def extract_top_part(original_image_path, threshold_image, output_json):
    file_name = path.basename(original_image_path)
    im_h, im_w = threshold_image.shape
    original_image = cv2.imread(original_image_path)
    
    split_height = int(im_h * 0.81)
    top_80 = threshold_image[:split_height, :]
    
    contours, contour_image = get_contours(top_80, original_image.copy()[:split_height, :])
    
    filtered_contours = [c for c in contours if c["id"] in [0]]
    cropped = crop_contours(top_80, filtered_contours)
    
    cropped_0 = [image for image in cropped if image["id"] == 0 ][0]
    
    cv2.imwrite(f"{temp_path}/document/{path.splitext(file_name)[0]}-tabular-form.jpg", cropped_0["image"])
    
    contours, contour_image = get_form_contours(original_image.copy()[:split_height, :]
        [cropped_0["y"]:cropped_0["y"]+cropped_0["h"], cropped_0["x"]:cropped_0["x"]+cropped_0["w"]])
    
    filtered_contours = [c for c in contours if c["id"] in [0]]
    cropped = crop_contours(cropped_0["image"], filtered_contours)
    
    form_table = [image for image in cropped if image["id"] == 0 ][0]
    
    cv2.imwrite(f"{temp_path}/document/{path.splitext(file_name)[0]}-form.jpg", form_table["image"])
    
    form_table_original = original_image.copy()[:split_height, :][cropped_0["y"]:cropped_0["y"]+cropped_0["h"], cropped_0["x"]:cropped_0["x"]+cropped_0["w"]][form_table["y"]:form_table["y"]+form_table["h"], form_table["x"]:form_table["x"]+form_table["w"]]
    
    rows = get_form_rows(form_table["image"])
    
    for row in rows:
        row_image = row["image"]
        im_h, im_w  = row_image.shape
        # im_w = 1691
        form_input = row_image[:, int(im_w * 0.25) : im_w]
        form_row_original = form_table_original[row['start_height']:row['end_height'], :]
        form_input_original = form_row_original[:, int(im_w * 0.25) : im_w]
        
        if(row['id'] == '40A'):
            standby_credit = form_input[ : , int(form_input.shape[1] * 0.64):]
            standby_credit_check = standby_credit[:, : int(standby_credit.shape[1] * 0.1)]
            transferable_credit = form_input[:, int(form_input.shape[1] * 0.28) : int(form_input.shape[1] - form_input.shape[1] * 0.38)]
            transferable_credit_check = transferable_credit[:, : int(transferable_credit.shape[1] * 0.2)]
            
            # cv2.imshow('transferable credit', transferable_credit_check)     
            # cv2.waitKey(0)
            
            output_json['page_1']['form_of_credit'] = {}
            
            if get_white_pixel(standby_credit_check) > get_white_pixel(transferable_credit_check):
                output_json['page_1']['form_of_credit']['standby_credit'] = True
                output_json['page_1']['form_of_credit']['transferable_credit'] = False
            else:
                output_json['page_1']['form_of_credit']['standby_credit'] = False
                output_json['page_1']['form_of_credit']['transferable_credit'] = True
                
        if(row['id'] == '20'):
            date_of_application = form_input[:, int(form_input.shape[1] * 0.66):]
            ocr_res = ocr_image(date_of_application, method="easy-ocr", numbers_only=True)
            output_json['page_1']['date_of_application'] = ocr_res[0][1]
            
            credit_number = form_input[:, : int(form_input.shape[1] * 0.4)]
            ocr_res = ocr_image(credit_number, method="easy-ocr")
            output_json['page_1']['credit_number'] = ocr_res[0][1]
        
        if(row['id'] == '31D'):
            output_json['page_1']['date_n_place_of_expiry'] = {}
            place = form_input[:, int(form_input.shape[1] * 0.45):]
            ocr_res = ocr_image(place, method="easy-ocr")
            output_json['page_1']['date_n_place_of_expiry']['place'] = ocr_res[0][1]
            
            date = form_input[:, int(form_input.shape[1] * 0.06) : int(form_input.shape[1] * 0.36)]
            ocr_res = ocr_image(date, method="easy-ocr", numbers_only=True)
            output_json['page_1']['date_n_place_of_expiry']['date'] = ocr_res[0][1]
            
        if(row['id'] == '50'):
            ocr_res = ocr_image(form_input, method="easy-ocr")
            output_json['page_1']['applicant_details'] = multiline_to_single_line(ocr_res)
        
        if(row['id'] == '59'):
            output_json['page_1']['benificiary details'] = {}
            benificiary_details = form_input[:, : int(form_input.shape[1] * 0.64)]
            ocr_res = ocr_image(benificiary_details, method="easy-ocr")
            output_json['page_1']['benificiary details']['name_n_address'] = multiline_to_single_line(ocr_res)
            
            contact_details = form_input[:, int(form_input.shape[1] * 0.65):][int(form_input.shape[0] * 0.32):, :]
            ocr_res = ocr_image(contact_details, method="easy-ocr")
            output_json['page_1']['benificiary details']['contact_details'] = multiline_to_single_line(ocr_res)
        
        if(row['id'] == '32B'):
            currency_n_amount = form_input[:, : int(form_input.shape[1] * 0.37)]
            ocr_res = ocr_image(currency_n_amount, method="easy-ocr")
            output_json['page_1']['currency_n_amount'] = ocr_res[0][1]
            
            output_json['page_1']['tolerance'] = {}
            
            tolerance_1 = currency_n_amount = form_input[:, int(form_input.shape[1] * 0.675) : int(form_input.shape[1] * 0.725)]       
            ocr_res = ocr_image(tolerance_1, method="easy-ocr", numbers_only=True)
            if len(ocr_res) != 0:
                output_json['page_1']['tolerance']['from'] = ocr_res[0][1] 
            else:
                output_json['page_1']['tolerance']['from'] = 0
            
            tolerance_2 = currency_n_amount = form_input[:, int(form_input.shape[1] * 0.77) : int(form_input.shape[1] * 0.81)]        
            ocr_res = ocr_image(tolerance_2, method="easy-ocr", numbers_only=True)
            if len(ocr_res) != 0:
                output_json['page_1']['tolerance']['to'] = ocr_res[0][1] 
            else:
                output_json['page_1']['tolerance']['to'] = 0
        
                # TODO: complete this
        if(row['id'] == '41A'):
            any_bank = form_input[:, : int(form_input.shape[1] * 0.5)]
            specific_bank = form_input[:, int(form_input.shape[1] * 0.5) : ]
            # cv2.imshow('tolerance', specific_bank)   
            # cv2.waitKey(0)
        
        if(row['id'] == '42C'):
            output_json['page_1']['drafts_at'] = {}
            drafts_at_sight = form_input[:, : int(form_input.shape[1] * 0.045)]
            drafts_at_days_after = form_input[:, int(form_input.shape[1] * 0.2) : int(form_input.shape[1] * 0.25)]
            if get_white_pixel(drafts_at_sight) > get_white_pixel(drafts_at_days_after):
                output_json['page_1']['drafts_at']['sight'] = True
                output_json['page_1']['drafts_at']['days_after'] = False
            else:
                output_json['page_1']['drafts_at']['sight'] = False
                output_json['page_1']['drafts_at']['days_after'] = True
            
        if(row['id'] == '43P'):
            output_json['page_1']['partial_shipment'] = {}
            partial_shipment_allowed = form_input[:, : int(form_input.shape[1] * 0.045)]
            partial_shipment_not_allowed = form_input[:, int(form_input.shape[1] * 0.32) : int(form_input.shape[1] * 0.385)]
            
            print(f"partial shipment allowed: {get_white_pixel(partial_shipment_allowed)}")
            print(f"partial shipment not allowed: {get_white_pixel(partial_shipment_not_allowed)}")

            if get_white_pixel(partial_shipment_allowed) > get_white_pixel(partial_shipment_not_allowed):
                output_json['page_1']['partial_shipment']['allowed'] = True
            else:
                output_json['page_1']['partial_shipment']['not_allowed'] = True
                
        if(row['id'] == '43T'):
            output_json['page_1']['transshipment'] = {}
            transshipment_allowed = form_input[:, : int(form_input.shape[1] * 0.045)]
            transshipment_not_allowed = form_input[:, int(form_input.shape[1] * 0.32) : int(form_input.shape[1] * 0.385)]
            
            print(f"transshipment allowed: {get_white_pixel(transshipment_allowed)}")
            print(f"transshipment not allowed: {get_white_pixel(transshipment_not_allowed)}")
            
            if get_white_pixel(transshipment_allowed) > get_white_pixel(transshipment_not_allowed):
                output_json['page_1']['transshipment']['allowed'] = True
            else:
                output_json['page_1']['transshipment']['not_allowed'] = True
            
        if(row['id'] == '44'):
            output_json['page_1']['shipment'] = {}
            
            shipment_44e = form_input[: int(form_input.shape[0] * 0.5), int(form_input.shape[1] * 0.32) : ]
            ocr_res = ocr_image(cv2.blur(shipment_44e, (3,3)), method="easy-ocr")
            output_json['page_1']['shipment']['44E'] = postprocess_ocr(ocr_res)
            
            shipment_44f = form_input[int(form_input.shape[0] * 0.5) : , int(form_input.shape[1] * 0.32) : ]
            ocr_res = ocr_image(cv2.blur(shipment_44f, (3,3)) , method="easy-ocr")
            output_json['page_1']['shipment']['44F'] = postprocess_ocr(ocr_res) 
            
            
        if(row['id'] == '44C'):
            
            latest_shipment_date = form_input[:, : int(form_input.shape[1] * 0.37)]
            ocr_res = ocr_image(cv2.blur(latest_shipment_date, (3,3)) , method="easy-ocr")
            output_json['page_1']['latest_shipment_date'] = postprocess_ocr(ocr_res)

        if(row['id'] == '45A'):
            description_of_goods = form_input[:, : int(form_input.shape[1] * 0.7)]
            ocr_res = ocr_image(cv2.blur(description_of_goods, (3,3)), method="easy-ocr", paragraph=True)
            output_json['page_1']['description_of_goods'] = ocr_res[0][1].lower()
            
            terms_of_delivery = form_input[int(form_input.shape[0] * 0.18) : , int(form_input.shape[1] * 0.71) : ]
            terms_of_delivery_1 = terms_of_delivery[int(form_input.shape[0] * 0.28) :, : int(terms_of_delivery.shape[1] * 0.48)]
            terms_of_delivery_1_opts = terms_of_delivery_1[: int(terms_of_delivery_1.shape[0] * 0.6), : int(terms_of_delivery_1.shape[1] * 0.24)]
            
            fob = terms_of_delivery_1_opts[: int(terms_of_delivery_1_opts.shape[0] //3), :]
            cfr = terms_of_delivery_1_opts[ int(terms_of_delivery_1_opts.shape[0] //3) : 2 * int(terms_of_delivery_1_opts.shape[0] //3), :]
            cif = terms_of_delivery_1_opts[ 2 * int(terms_of_delivery_1_opts.shape[0] //3) : , :]
            
            terms_of_delivery_2 = terms_of_delivery[:, int(terms_of_delivery.shape[1] * 0.49):]
            terms_of_delivery_2_opts = terms_of_delivery_2[: int(terms_of_delivery_2.shape[0] * 0.57), : int(terms_of_delivery_2.shape[1] * 0.24)]
            
            fca = terms_of_delivery_2_opts[: int(terms_of_delivery_2_opts.shape[0] //4), :]
            cpt = terms_of_delivery_2_opts[ int(terms_of_delivery_2_opts.shape[0] //4) : 2 * int(terms_of_delivery_2_opts.shape[0] //4), :]
            cip = terms_of_delivery_2_opts[ 2 * int(terms_of_delivery_2_opts.shape[0] //4) : 3 * int(terms_of_delivery_2_opts.shape[0] //4), :]
            other = terms_of_delivery_2_opts[ 3 * int(terms_of_delivery_2_opts.shape[0] //4) : , :]
            
            terms_of_delivery_checks = {
                TermsOfDelivery.FOB.value: get_white_pixel(fob),
                TermsOfDelivery.CFR.value: get_white_pixel(cfr),
                TermsOfDelivery.CIF.value: get_white_pixel(cif),
                TermsOfDelivery.FCA.value: get_white_pixel(fca),
                TermsOfDelivery.CPT.value: get_white_pixel(cpt),
                TermsOfDelivery.CIP.value: get_white_pixel(cip),
                'OTHER': get_white_pixel(other)
            }
            
            # marked average-: 330
            # non marked average-: 150
            
            MARKED_THRESHOLD = 300
            NON_MARKED_THRESHOLD = 150
            
            output_json['page_1']['terms_of_delivery'] = {}
            output_json['page_1']['terms_of_delivery']['sea_shipment'] = []
            output_json['page_1']['terms_of_delivery']['all_shipment'] = []
            
            for key, value in terms_of_delivery_checks.items():
                if key in [TermsOfDelivery.FOB.name, TermsOfDelivery.CFR.name, TermsOfDelivery.CIF.name]:
                    if value > MARKED_THRESHOLD:
                        output_json['page_1']['terms_of_delivery']['sea_shipment'].append(key)
                else:
                    if value > MARKED_THRESHOLD:
                        output_json['page_1']['terms_of_delivery']['all_shipment'].append(key)
                        
            
            
        if(row['id'] == '46A'):
            output_json['page_1']['documents_required'] = {}
            opt_1_check = form_input[: int(form_input.shape[0] * 0.1) , : int(form_input.shape[1] * 0.046)]
            opt_2_check = form_input[int(form_input.shape[0] * 0.18) : int(form_input.shape[0] * 0.28) , : int(form_input.shape[1] * 0.046)]
            opt_3_check = form_input[int(form_input.shape[0] * 0.39) : int(form_input.shape[0] * 0.5) , : int(form_input.shape[1] * 0.046)]
            opt_4_check = form_input[int(form_input.shape[0] * 0.57) : int(form_input.shape[0] * 0.67) , : int(form_input.shape[1] * 0.046)]
            opt_5_check = form_input[int(form_input.shape[0] * 0.91) : int(form_input.shape[0] * 0.995) , : int(form_input.shape[1] * 0.046)]
            opt_6_check = form_input[int(form_input.shape[0] * 0.91) : int(form_input.shape[0] * 0.995), int(form_input.shape[1] * 0.365) : int(form_input.shape[1] * 0.4)]
            
            print('46A')
            print('option 1: ', get_white_pixel(opt_1_check))
            print('option 2: ', get_white_pixel(opt_2_check))    
            print('option 3: ', get_white_pixel(opt_3_check))    
            print('option 4: ', get_white_pixel(opt_4_check))    
            print('option 5: ', get_white_pixel(opt_5_check))
            print('option 6: ', get_white_pixel(opt_6_check))           
            
            
            # cv2.imshow('option 6', opt_6_check)
            # cv2.waitKey(0)
            
            print(f"weight list copies: {ocr_res[0][1]}")
            
            output_json['page_1']['documents_required']['option_1'] = {}
            
            no_of_copies = form_input[: int(form_input.shape[0] * 0.11) , int(form_input.shape[1] * 0.451) : int(form_input.shape[1] * 0.5)]
            ocr_res = ocr_image(cv2.blur(no_of_copies, (3,3)), numbers_only=True, method="easy-ocr")
            output_json['page_1']['documents_required']['option_1']['no_of_copies'] = ocr_res[0][1]
            
            hs_code_no = form_input[int(form_input.shape[0] * 0.1) :  int(form_input.shape[0] * 0.2), int(form_input.shape[1] * 0.32) : int(form_input.shape[1] * 0.9)]
            ocr_res = ocr_image(cv2.blur(hs_code_no, (3,3)), method="easy-ocr", paragraph=True)
            output_json['page_1']['documents_required']['option_1']['hs_code_no'] = ocr_res[0][1]
            
            output_json['page_1']['documents_required']['option_2'] = {}
            lading_marked_freight = form_input[int(form_input.shape[0] * 0.17) :  int(form_input.shape[0] * 0.28), int(form_input.shape[1] * 0.62) : int(form_input.shape[1] * 0.95)]
            ocr_res = ocr_image(cv2.blur(lading_marked_freight, (3,3)), method="easy-ocr", paragraph=True)
            output_json['page_1']['documents_required']['option_2']['lading_marked_freight'] = ocr_res[0][1]
            
            output_json['page_1']['documents_required']['option_3'] = {}
            plc_marked_freight = form_input[int(form_input.shape[0] * 0.48) :  int(form_input.shape[0] * 0.58), : int(form_input.shape[1] * 0.28)]
            ocr_res = ocr_image(cv2.blur(plc_marked_freight, (3,3)), method="easy-ocr", paragraph=True)
            output_json['page_1']['documents_required']['option_3']['plc_marked_freight'] = ocr_res[0][1]
            
            output_json['page_1']['documents_required']['option_4'] = {}
            institute_war_clause = form_input[int(form_input.shape[0] * 0.645) :  int(form_input.shape[0] * 0.73), int(form_input.shape[1] * 0.49) : int(form_input.shape[1] * 0.71)]
            ocr_res = ocr_image(institute_war_clause, method="easy-ocr", paragraph=True)
            output_json['page_1']['documents_required']['option_4']['institute_war_clause'] = ocr_res[0][1]
            
            institute_strike_clause = form_input[int(form_input.shape[0] * 0.71) :  int(form_input.shape[0] * 0.8), : int(form_input.shape[1] * 0.20)]
            ocr_res = ocr_image(cv2.blur(institute_strike_clause, (3,3)), method="easy-ocr", paragraph=True)
            output_json['page_1']['documents_required']['option_4']['institute_strike_clause'] = ocr_res[0][1]
            
            insured_value = form_input[int(form_input.shape[0] * 0.71) :  int(form_input.shape[0] * 0.8), int(form_input.shape[1] * 0.23) : int(form_input.shape[1] * 0.315)]
            ocr_res = ocr_image(cv2.blur(insured_value, (3,3)), method="easy-ocr", paragraph=True)
            output_json['page_1']['documents_required']['option_4']['insured_value'] = ocr_res[0][1]
            
            percentage_addon_for_coverage = form_input[int(form_input.shape[0] * 0.71) :  int(form_input.shape[0] * 0.8), int(form_input.shape[1] * 0.42) : int(form_input.shape[1] * 0.475)]
            ocr_res = ocr_image(cv2.blur(percentage_addon_for_coverage, (3,3)), method="easy-ocr", numbers_only=True)
            output_json['page_1']['documents_required']['option_4']['percentage_addon_for_coverage'] = ocr_res[0][1]
            
            output_json['page_1']['documents_required']['option_5'] = {}
            packing_list_copies = form_input[int(form_input.shape[0] * 0.89) : , int(form_input.shape[1] * 0.2) : int(form_input.shape[1] * 0.27)]
            ocr_res = ocr_image(cv2.blur(packing_list_copies, (3,3)), method="easy-ocr", numbers_only=True)
            output_json['page_1']['documents_required']['option_5']['packing_list_copies'] = ocr_res[0][1]
            
            output_json['page_1']['documents_required']['option_6'] = {}
            weight_list_copies = form_input[int(form_input.shape[0] * 0.89) : , int(form_input.shape[1] * 0.55) : int(form_input.shape[1] * 0.605)]
            ocr_res = ocr_image(cv2.blur(weight_list_copies, (3,3)), method="easy-ocr", numbers_only=True)
            output_json['page_1']['documents_required']['option_6']['weight_list_copies'] = ocr_res[0][1]
            
        if(row['id'] == '47A'):
            output_json['page_1']['special_instructions'] = {}
            opt_1_check = form_input[: int(form_input.shape[0] * 0.33) , : int(form_input.shape[1] * 0.045)]
            opt_2_check = form_input[int(form_input.shape[0] * 0.45) : int(form_input.shape[0] * 0.78) , : int(form_input.shape[1] * 0.045)]
            opt_3_check = form_input[int(form_input.shape[0] * 0.75) : int(form_input.shape[0] * 0.99) , : int(form_input.shape[1] * 0.045)]
            
            # 685
            
            if get_white_pixel(opt_1_check) > 500:
                output_json['page_1']['special_instructions']['option_1'] = {}
                
                send_copies_by = form_input[int(form_input.shape[0] * 0.02) : int(form_input.shape[0] * 0.28) , int(form_input.shape[1] * 0.638) : int(form_input.shape[1] * 0.81)]
                ocr_res = ocr_image(cv2.blur(send_copies_by, (3,3)), method="easy-ocr")
                output_json['page_1']['special_instructions']['option_1']['send_copies_by'] = ocr_res[0][1]
                
                within_days = form_input[int(form_input.shape[0] * 0.02) : int(form_input.shape[0] * 0.28) , int(form_input.shape[1] * 0.873) : int(form_input.shape[1] * 0.906)]
                ocr_res = ocr_image(cv2.blur(within_days, (3,3)), method="easy-ocr", numbers_only=True)
                output_json['page_1']['special_instructions']['option_1']['within_days'] = ocr_res[0][1]
                
                accompany_original_document = form_input[int(form_input.shape[0] * 0.25) : int(form_input.shape[0] * 0.5) , int(form_input.shape[1] * 0.12) : int(form_input.shape[1] * 0.53)]
                ocr_res = ocr_image(cv2.blur(accompany_original_document, (3,3)), method="easy-ocr", paragraph=True, special_characters=False)
                output_json['page_1']['special_instructions']['option_1']['accompany_original_document'] = ocr_res[0][1]
            
            if get_white_pixel(opt_2_check) > 500:
                output_json['page_1']['special_instructions']['option_2'] = {}
                cover_no = form_input[int(form_input.shape[0] * 0.48) : int(form_input.shape[0] * 0.75) , int(form_input.shape[1] * 0.6) : int(form_input.shape[1] * 0.99)]
                ocr_res = ocr_image(cv2.blur(cover_no, (3,3)), paragraph=True, method="easy-ocr")
                
                output_json['page_1']['special_instructions']['option_2']['open_cover_no'] = ocr_res[0][1]
                
            if get_white_pixel(opt_3_check) > 500:
                output_json['page_1']['special_instructions']['option_3'] =  True
        
        if(row['id'] == '48'):
            output_json['page_1']['period_for_presentation'] = {}
            period_for_presentation = form_input[: int(form_input.shape[0] * 0.65), : ]
            
            no_of_days = period_for_presentation[:, int(period_for_presentation.shape[1] * 0.372) : int(period_for_presentation.shape[1] * 0.428)]
            shipment_date_check = period_for_presentation[:, int(period_for_presentation.shape[1] * 0.525) : int(period_for_presentation.shape[1] * 0.555)]
            grn_date_check = period_for_presentation[:, int(period_for_presentation.shape[1] * 0.702) : int(period_for_presentation.shape[1] * 0.725)]
            
            ocr_res = ocr_image(cv2.blur(no_of_days, (3,3)), method="easy-ocr", numbers_only=True)          
            output_json['page_1']['period_for_presentation']['days'] = ocr_res[0][1]
            
            if get_white_pixel(shipment_date_check) > get_white_pixel(grn_date_check):
                output_json['page_1']['period_for_presentation']['shipment_date'] = True
            else:
                output_json['page_1']['period_for_presentation']['date_of_grn'] = True
            
        if(row['id'] == '49'):
            output_json['page_1']['confirmation'] = {}
            required_check = form_input[:, : int(form_input.shape[1] * 0.035)]
            not_required_check = form_input[:, int(form_input.shape[1] * 0.13) : int(form_input.shape[1] * 0.16)]
            
            if get_white_pixel(required_check) > get_white_pixel(not_required_check):
                output_json['page_1']['confirmation']['required'] = True
            else:
                output_json['page_1']['confirmation']['required'] = False
            
            applicant_check = form_input[:, int(form_input.shape[1] * 0.65) : int(form_input.shape[1] * 0.67)]
            benificiary_check = form_input[:, int(form_input.shape[1] * 0.81) : int(form_input.shape[1] * 0.845)]
            
            if get_white_pixel(applicant_check) > get_white_pixel(benificiary_check):
                output_json['page_1']['confirmation']['charges_on'] = 'Applicant'
            else:
                output_json['page_1']['confirmation']['charges_on'] = 'Beneficiary'
        
        if(row['id'] == '71B'):
            applicant_check = form_input[:, int(form_input.shape[1] * 0.51) : int(form_input.shape[1] * 0.55)]
            benificiary_check = form_input[:, int(form_input.shape[1] * 0.69) : int(form_input.shape[1] * 0.72)]
            
            if get_white_pixel(applicant_check) > get_white_pixel(benificiary_check):
                output_json['page_1']['charges'] = 'Applicant'
            else:
                output_json['page_1']['charges'] = 'Beneficiary'
            
    cv2.destroyAllWindows()
    
    return output_json

def main():
    pdf_path = f"{input_path}/letter-of-credit-application-new-handwritten.pdf"
    # get_from_printed(pdf_path)
    
    # flat_pdf = flatten_pdf(pdf_path)
    images = pdf_to_img(pdf_path)
    images = save_images(images)
    output_json = {}
    
    for idx, image in enumerate(images):
        if(path.isfile(image)):
            file_name = path.basename(image)
            thresholded_img = preprocess_page(image)
            
            if(idx == 1):
                img = thresholded_img.copy()
                im_h, im_w = thresholded_img.shape
                output_json['page_1'] = {}
                
                output_json = extract_top_part(image, img, output_json)
                output_json = extract_bottom_part(image, img, output_json)
                
                print(output_json)
                # ocr_res = ocr_image(cropped_5['image'], method="easy-ocr")
                # print_ocr_res(ocr_res)
                
                
                # cv2.imshow('contours', cv2.resize(img, (1800, 666)))
                
                # ocr_res = ocr_image(bottom_20, method="easy-ocr")
                
                # for (_, text, score) in ocr_res:
                #     print(f"Text: {text}")
                #     print(f"Score: {score}")
                #     print(" ")
                    
                # img = cv2.imread(image).copy()
                # threshold = 0.25
                # img = draw_bounding_boxes(img[split_height:, :], ocr_res, threshold)
                
                # cv2.imshow('ocr text', cv2.resize(img, (1800, 666)))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                img = thresholded_img.copy()
                output_json['page_2'] = {}
                
                # get_name_n_date(image, img, output_json)
            
            # file_name = path.basename(image)
            # ocr_res = ocr_image(f"{temp_path}/document/{path.splitext(file_name)[0]}-thresh.jpg", method="easy-ocr")
            
            # img = cv2.imread(image).copy()
            # threshold = 0.25
            # img = draw_bounding_boxes(img, ocr_res, threshold)
            
            # cv2.imwrite(f"{temp_path}/document/{path.splitext(file_name)[0]}-ocr.jpg", img)
            
            # with open(f"{temp_path}/document/ocr_result.txt", 'w') as f:
            #     for element in ocr_res:
            #         f.write(json.dumps(str(element)) + '\n')
    dict_to_json(output_json)        

if __name__ == "__main__":
    main()