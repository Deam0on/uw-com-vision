from PIL import Image
import cv2
import numpy as np
from numpy import sqrt
from google.colab.patches import cv2_imshow
import easyocr
import re

test_img_path = "/home/deamoon_uw_nn/DATASET/INFERENCE/"
# Load image, convert to grayscale, Otsu's threshold
for test_img in os.listdir(test_img_path):
    # Read image
    image = cv2.imread('/content/70.tif')
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    # Use canny edge detection
    edges = cv2.Canny(gray,50,150,apertureSize=3)
    
    reader = easyocr.Reader(['en'])
    result = reader.readtext(gray, detail = 0)
    pxum_r = result[0]
    psum = re.sub("[^0-9]", "", pxum_r)
    print(psum)
    
    # Apply HoughLinesP method to
    # to directly obtain line end points
    lines_list =[]
    lines = cv2.HoughLinesP(
    			edges, # Input edge image
    			1, # Distance resolution in pixels
    			np.pi/180, # Angle resolution in radians
    			threshold=100, # Min number of votes for valid line
    			minLineLength=100, # Min allowed length of line
    			maxLineGap=1 # Max allowed gap between line for joining them
    			)
    
    # Iterate over points
    for points in lines:
        # Extracted points nested in the list
        x1,y1,x2,y2=points[0]
        # Draw the lines joing the points
        # On the original image
        cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1,y1),(x2,y2)])
        scale_len = sqrt((x2-x1)**2+(y2-y1)**2)
        um_pix = float(psum)/scale_len
    
    print (um_pix)
    
    # Save the result image
    cv2.imwrite('detectedLines.png',image)
    cv2_imshow(image)
