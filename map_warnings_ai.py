# The code identifies the warning present in WWU and WPD maps +
# All required imports
import cv2
import numpy as np
import pdfbox
import threading
import sys
import argparse
import json
import os
from logging_config import logging
# Logger
logger = logging.getLogger(__name__)
logger.info("inside map warnings logger")

### 1.Setting the parameters for identifying the colour ranges for identifying the warning signs in different utility types
'''
Values are calibrated manually for each type of warning. The HSV value for lower and upper ranges are mentioned.
warning: string to be displayed if a specific warning is detected.
z_thresh: The value of threshold to increase the robustness of the model.
'''
Parameters ={
            "Gas":
                {"MP":{"low":np.array([80, 100, 20]),
                        "high":np.array([110, 255, 255]),
                        "warning":'There is a Medium Pressure Gas Line in this Area| ',
                        "z_thresh": 0}, 
                "LHP":{"low":np.array([10, 150, 20]),
                        "high":np.array([20, 255, 255]),
                        "warning":'There is an LHP Pressure Gas Line in this Area | ',
                        "z_thresh": 20},
                "IP":{"low":np.array([50, 100, 20]),
                        "high":np.array([65, 255, 255]),
                        "warning":'There is an Intermediate Pressure Gas Line in this Area | ',
                        "z_thresh": 20}
                },
            "Electricity":
                {"11KV":{"low":np.array([0, 50, 50]),
                        "high":np.array([10, 255, 255]),
                        "warning": 'There is a 11kV High Voltage Electricity Line in this Area | ',
                        "z_thresh": 0}, 
                "33KV":{"low":np.array([50, 100, 20]),
                        "high":np.array([65, 255, 255]),
                        "warning":'There is a 33kV High Voltage Electricity Line in this Area | ',
                        "z_thresh": 0},
                "66KV":{"low":np.array([10, 100, 20]),
                        "high":np.array([20, 200, 255]),
                        "warning":'There is a 66kV High Voltage Electricity Line in this Area| ',
                        "z_thresh": 20},
                "132KV":{"low":np.array([140, 150, 20]),
                        "high":np.array([160, 255, 255]),
                        "warning":'There is a 132kV High Voltage Electricity Line in this Area| ',
                        "z_thresh": 20},
                },
            "Water":{
                "Mains":{
                    "low":np.array([100, 100, 20]),
                    "high":np.array([130, 255, 255]),
                    "warning": 'Warning: There is a Water Mains Line in this Area|',
                    "z_thresh":0
                    },
                "Combined":{
                    "low":np.array([0, 50, 50]),
                    "high":np.array([10, 255, 255]),
                    "warning":'Warning: There is a Combined Line in this Area|',
                    "z_thresh":0
                    },
                "Surface":{
                    "low":np.array([50, 100, 20]),
                    "high":np.array([65, 255, 255]),
                    "warning":'Warning: There is a Surface Line/Gravity Sewer in this Area|',
                    "z_thresh":0
                    }
                },
            "Electricity_SPE":{
                "11KV":{
                    "low":np.array([0, 50, 50]),
                    "high":np.array([10, 255, 255]),
                    "warning": 'Warning: There is HV 22/11KV Cable in this area|',
                    "z_thresh":0
                    },
                 "EHV":{
                    "low":np.array([50, 100, 20]),
                    "high":np.array([65, 255, 255]),
                    "warning": 'Warning: There is 33KV Cable in this area|',
                    "z_thresh":0
                    },
                 "Trans":{
                    "low":np.array([100, 100, 20]),
                    "high":np.array([130, 255, 255]),
                    "warning": 'Warning: There is 132KV Cable in this area|',
                    "z_thresh":20
                   }
                },
            "Gas_Cadent":{
                "MP":{
                    "low":np.array([80, 100, 20]),
                    "high":np.array([110, 255, 255]),
                    "warning":'There is a Medium Pressure Gas Line in this area| ',
                    "z_thresh": 0
                    }, 
                "LHP":{
                    "low":np.array([10, 150, 20]),
                    "high":np.array([20, 255, 255]),
                    "warning":'There is an LHP Pressure Gas Line in this area | ',
                    "z_thresh": 20
                    },
                "IP":{
                    "low":np.array([50, 100, 20]),
                    "high":np.array([65, 255, 255]),
                    "warning":'There is an Intermediate Pressure Gas Line in this area | ',
                    "z_thresh": 20
                    }
                },
            "Electricity_NGE":{
                "Undergroud":{
                    "low":np.array([40, 100, 20]),
                    "high":np.array([75, 255, 255]),
                    "warning": 'Warning: There is an Underground Cable in this area|',
                    "z_thresh":0
                    },
                "Overhead":{
                    "low":np.array([0, 50, 50]),
                     "high":np.array([10, 255, 255]),
                     "warning": 'Warning: There is an Overhead Cable in this area|',
                     "z_thresh":40
                     },
                "Fiber":{
                    "low":np.array([10, 100, 20]),
                    "high":np.array([30, 255, 255]),
                    'warning': 'Warning: There is a Fiber Cable in this area|',
                    'z_thresh':0
                      }
                },
            "NGG":{
                "NHP_Mains":{
                    "low":np.array([140, 150, 20]),
                    "high":np.array([160, 155, 20]),
                    "warning": 'Warning: There is NHP Mains in this area|',
                    'z_thresh':0
                     }
                },
            'Gigaclear':{
                "Route":{
                    "low":np.array([0, 50, 50]),
                    "high":np.array([10, 255, 255]),
                    "warning": 'Warning: There is Gigaclear Route in this area|',
                    "z_thresh":0
                    }
                },
            'Zayo':{
                "Duct":{
                    "low":np.array([0, 50, 50]),
                    "high":np.array([10, 255, 255]),
                    "warning": 'Warning: There is Zayo Duct in this area|',
                    "z_thresh":20
                    }
                },
            'Neos':{
                "Underground_Route":{
                    "low":np.array([130, 100, 20]),
                    "high":np.array([140, 255, 255]),
                    "warning": 'Warning: There is Neos Underground Route in this area|',
                    "z_thresh":10
                    }
                }
            }
'''
Parameters for finding the region of interest(Interior of the circle in the case of spot pack) in different utility maps
'''
Parameters_ROI = {'Electricity':{
                            'min_r':30,
                            'max_r':75,
                            'param_1':100,
                            'param_2': 0.9,
                            'r_boundary':35#5
                                    },
                    'Water':{
                            'min_r':2,
                            'max_r':15,
                            'param_1':100,
                            'param_2': 30,
                            'r_boundary':15
                            },
                    'Gas':{
                            'min_r':60,
                            'max_r':80,
                            'param_1':300,
                            'param_2': 0.9,
                            'r_boundary':15#5
                            },
                    'Electricity_NGE':{
                            'min_r':20,
                            'max_r':40,
                            'param_1':100,
                            'param_2':30,
                            'r_boundary':15
                            },
                    'Electricity_SPE':{
                            'min_r':140,
                            'max_r':190,
                            'param_1':100,
                            'param_2': 30,
                            'r_boundary':5
                            },
                    'Gas_Cadent':{
                            'min_r':140,
                            'max_r':190,
                            'param_1':100,
                            'param_2': 30,
                            'r_boundary':5
                            },
                    'NGG':{
                            'min_r':15,
                            'max_r':40,
                            'param_1':100,
                            'param_2':30,
                            'r_boundary':15
                            },
                    'Gigaclear':{
                            'min_r':60,
                            'max_r':80,
                            'param_1':300,
                            'param_2': 0.9,
                            'r_boundary':15#5
                            },
                    "Zayo":{ #Needs to be updated for 1:1000 scale
                            'min_r':70,
                            'max_r':100,
                            'param_1':200,
                            'param_2': 10,
                            'r_boundary':15#5
                            },
                    'Neos':{
                            'min_r':60,
                            'max_r':80,
                            'param_1':300,
                            'param_2': 0.9,
                            'r_boundary':15#5
                            }
                }


def threshold_filter(img, z_thresh, low_mask, high_mask, warning_p, warning_n=""):
    '''
    Function to do the thresholding operation for detection of specific warning in each utility map.
    The value of different parameters come from Parameters dictionary.

    img: the 3 channel RGB image of the utility map
    z_thresh: Threshold value for the specific utility type as mentioned in Parameters dictionary
    low_mask: The lower value of hsv from the range to be identified for identification of colour for a specific warning
    high_mask: The higher value of hsv from the range to be identified for identification of colour for a specific warning
    warning_p: The string to return for the positive warning identified
    warning_n="", The string to return when no warning identified in the code
    '''
    # Convert the image to HSV space
    try:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except Exception as e:
        print("Error:", e)
        logger.info(e)

    # Check the HSV values at each pixel value if it is within the color ranges to be detected
    color_mask = cv2.inRange(hsv_img, low_mask, high_mask)
    color_det = cv2.bitwise_and(img, img, mask=color_mask)

    # # Read the image in greyscale, count detected pixels
    img_grey = color_det[:, :, -1]
    z = cv2.countNonZero(img_grey)
    if (z > z_thresh):
        warning_string = warning_p
    else:
        warning_string = warning_n
    return warning_string

# Function to find the circular Zone of Interest on each spot pack
def crop_circle(img, utility_type):
    # The parameters are set for each type of utility
    # Detect circle using Hough Circles
    '''
    Input parameters:
        img: image of the map, 3 channel image(RGB)
        utility_type: the utility type of the map file
    Output parameters:
        result: image of same size as 'img', containing only the region of interest(As identified by a circle on a spot pack)
    '''
    # result=None
    imgcopy = img.copy()
    gray = cv2.cvtColor(imgcopy, cv2.COLOR_BGR2GRAY)
    rows = gray.shape[0]
    # Pass Circle Parameters
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                                param1 = Parameters_ROI[utility_type]['param_1'], param2=Parameters_ROI[utility_type]['param_2'],
                                minRadius = Parameters_ROI[utility_type]['min_r'], maxRadius = Parameters_ROI[utility_type]['max_r'])
    # print("This is circles", circles)
    # Find the center coordinates and radius of the detected circle

    if circles is not None:
        circles = np.uint16(np.around(circles))
        center = (circles[0][0][0], circles[0][0][1])
        radius = circles[0][0][2]
        # Crop the detected circle
        (x,y) = center
        r = radius+Parameters_ROI[utility_type]['r_boundary']
        image = img.copy()
        mask = np.zeros(image.shape, dtype=np.uint8)
        mask = cv2.circle(mask,(x,y), r, (255,255,255), -1)
        result = cv2.bitwise_and(image, mask)

        # Uncomment this if you want to visualize the detected circle
    else:
        logger.info('Error: No Circles detected')
        result='None'

    return result
# Water map protected site detection using a cropped template
def water_map_warning(image,template, method, templ_type):
    '''
    Input parameters:
        input_map: image of the map, 3 channel image(RGB)
        utility_type: the utility type of the map file
        templ_type: protected or contaminated
    Output parameters:
        result: image of same size as 'img', containing only the region of interest(As identified by a circle on a spot pack)
    '''
    # image = image.astype(np.uint8)
    temp0=cv2.imread(template,0)
    # temp0 = temp0.astype(np.uint8)
    r=cv2.matchTemplate(image,temp0,method=eval(method))
    print("Match score: ", np.max(r))
    if (np.max(r))>=0.5 and templ_type == 'Protected':
        return "Protected site"
    elif (np.max(r))>=0.4 and templ_type == 'Contaminated':
        return "Contaminated site"
    else:
        return " "

##************************************************************MAIN PROGRAM ************************************************************************


def inputfunction(filename_pdf=r'G1.pdf', utility_type="Gas", packFolderPath='.', number_of_files=1):
    # 1) Read a pdf file and take input its utility type
    # 2) Convert pdf into jpg and select the map area
    '''
    The function takes in a filename, associated utility type and the path and returns the associated warnings if any.
    Input parameters:
        filename_pdf: The name of the file to be analysed
        utility_type: the utility type of the filename_pdf
        packFolderPath: The path to the filename_pdf
    Output parameters:
        Dictionary of utility type to indentified warning

    '''
    filename = os.path.join(packFolderPath, filename_pdf)
    print("this is filename",filename)
    logger.info("----------------filename-----------")
    logger.info(filename)

    # Empty warning string and the type of utilities considered in this function    
    warning_str_total = ''
    warning_dict = {'Gas':"", 'Electricity':"", 'Water':"", 'Electricity_SPE':"", "Electricity_NGE":"", "Gas_Cadent":"",
                    'Zayo':"", 'NGG':"", 'Neos':"", 'Gigaclear':""}
    #Checking if the utility type can be processed by the function or not
    if utility_type in warning_dict.keys():
        assert filename.endswith('.pdf'),  'Filename is expected to be of pdf type for this utility type'
        print("assertion done")
        try:
            # Convert the pdf to image
            p = pdfbox.PDFBox()
            print("Hi",filename)
            p.pdf_to_images(filename)
            print("converted",filename)
            fileName,fileExtension = os.path.splitext(filename)
            print("pdf has been conerted")
            file_name = fileName
            print("filename  ",file_name)
        except Exception as e:
            print("e",e)
            logger.info(e)

        # Count the number of pages for a multi page pdf file
        

        print("This is file_name split: ",file_name.split('\\')[-1])
        count = 0
        for name in os.listdir(packFolderPath):
            if name.endswith('.jpg') and file_name.split('\\')[-1] in name:
                print(name)
                count += 1
        print('This file: ', file_name.split('\\')[-1], 'has a count: ', count)
        no_of_pages = count


        for i in range(number_of_files):

            # If pdf is multi-page, then 'second last' page contains the map
            # THIS HAS TO BE MANUALLY CHECKED ANYTIME A NEW UTILITY COMES UP
            if (no_of_pages > 1):
                utility_map_image_name = file_name + str(no_of_pages-1)+'.jpg'

            else:
                # If pdf is single page, then first page is the map
                utility_map_image_name = file_name + str(i+1)+'.jpg'
            
            # Read Image file and crop region of interest
            ori_img = cv2.imread(utility_map_image_name)
            # assert ori_img.empty() , "File not found"
            assert ori_img.shape[2]==3 , "Image not a 3 channel image."
            img = crop_circle(ori_img, utility_type)

            # print(Parameters[utility_type])
            # Running the warning identification for all different type of warning to be identified
            # threads=[]
            if img == 'None':
                warning_str_total = "Please check the map manually"
                print(warning_str_total)
            else:
                for type_warning in Parameters[utility_type].keys():
                    if utility_type != 'Water':

                        # t = threading.Thread(target=threshold_filter, args=[Parameters[utility_type][type_warning]["z_thresh"], 
                        #                                 Parameters[utility_type][type_warning]["low"], 
                        #                                 Parameters[utility_type][type_warning]["high"],
                        #                                 Parameters[utility_type][type_warning]["warning"]])
                        # t.start()
                        # threads.append(t)
                        # for thread in threads:
                        #     thread.join()
                        print('This is utility type', utility_type, type_warning)
                        warning_str = threshold_filter(img,
                                                        Parameters[utility_type][type_warning]["z_thresh"], 
                                                        Parameters[utility_type][type_warning]["low"], 
                                                        Parameters[utility_type][type_warning]["high"],
                                                        Parameters[utility_type][type_warning]["warning"]
                                                        )
                        warning_str_total = warning_str_total + warning_str
            if utility_type=='Water':
                warning_str_total+= water_map_warning(image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY),template='protected_region.jpg' , method = 'cv2.TM_CCOEFF_NORMED', templ_type = 'Protected')
                warning_str_total+= water_map_warning(image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY),template='protected_green.jpeg' , method = 'cv2.TM_CCOEFF_NORMED', templ_type = 'Protected')
                warning_str_total+= water_map_warning(image = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY),template='contaminated.jpg' , method = 'cv2.TM_CCOEFF_NORMED', templ_type = 'Contaminated')


            if (warning_str_total == ''):
                warning_str_total = " No Warning for "+str(utility_type)

            warning_dict[utility_type]= warning_str_total
            # Deleting the image file created
            if (no_of_pages>1):
                for i in range(no_of_pages):
                    os.remove(file_name + str(i+1)+'.jpg')
            else:
                os.remove(utility_map_image_name)
            # for file in os.listdir(packFolderPath):
            #     if name.endswith('.jpg') and file_name.split('\\')[-1] in name:
            #         os.remove(file)
    else:
        logger.info("Map type not recognized")
    logger.info(warning_dict[utility_type])
    ##Enable this return statement if it is AI code testing only
    # print("warning dict",warning_dict)
    return warning_dict
    ##Enable this return statement if it is end to end pipeline integration
    # return warning_dict[utility_type]
# output_trail=inputfunction(filename_pdf=r'26001259_CadentGas.pdf', utility_type="Gas_Cadent", packFolderPath='.', number_of_files=1)
# print(output_trail)
