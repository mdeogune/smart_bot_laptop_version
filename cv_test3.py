import cv2
import cv
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
os.chdir('/home/mukesh-deo/Documents')
global x
global y
global z
global c_img
y = z = float(500)
x=75



def draw_lanes(img, lines, color=[0, 255, 255], thickness=3):

    # if this fails, go with some default line
    try:

        # finds the maximum y value for a lane marker 
        # (since we cannot assume the horizon will always be at the same point.)

        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                # These four lines:
                # modified from http://stackoverflow.com/questions/21565994/method-to-return-the-equation-of-a-straight-line-given-two-points
                # Used to calculate the definition of a line, given two sets of coords.
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                # Calculating our new, and improved, xs
                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.8):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2]
    except Exception as e:
        pass
       # print(str(e))
   

#cap=cv2.VideoCapture('Dash Cam - Car pool lane.mp4')
def cv_size(img):
    return tuple(img.shape[1::-1])

def wrap_image(x,y,z):
                                              
##    x=(x-90)*(np.pi/180)
##    y=(y-90)*(np.pi/180)
##    z=(z-90)*(np.pi/180)
##    R_array = np.array([[x1], [y1], [z1]])
##    R_Vec = cv.fromarray(R_array)
##    R = cv.CreateMat(3, 3, cv2.cv.CV_64FC1)
##    h,w=cv_size(img)
##
##    
##    cv.Rodrigues2(R_Vec, R)
##    R_r = np.array(R)
##    R_r[0][2] = 0
##    R_r[1][2] = 0
##    R_r[2][2] = 1
##    RR = cv.fromarray(R_r)
##
##    Trans_Mat = array([[[1], [0], [-w/2]],
##                        [[0], [1], [-h/2]],
##                        [[0], [0], [1]]])

    #Trans_Mat2 = cv.fromarray(Trans_Mat)
##    R_T_Mat = dot(RR, Trans_Mat2)
##    R_T_Mat[2][2] += h
##
##
##    Intrinsic_Mat = np2.array([[[h], [0], [w/2]],
##                           [[0], [h], [h/2]],
##                           [[0], [0], [1]]])
##
##    Int_Mat = cv.fromarray(Intrinsic_Mat)
##    H = dot(Int_Mat, R_T_Mat)
    rX = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    rY = np.array([[np.cos(y), 0, -np.sin(y)], [0, 1, 0], [np.sin(y), 0, np.cos(y)]])
    rZ = np.array([[np.cos(z), np.sin(z), 0], [-np.sin(z), np.cos(z), 0], [0, 0, 1]])


    X = cv2.cv.fromarray(rX)
    Y = cv2.cv.fromarray(rY)
    Z = cv2.cv.fromarray(rZ)
    dst = np.array([
	[0, 0],
	[x,200],
	[y,200],
	[600, 0]], dtype = "float32")
    return dst

def warp_displayImage():
    dst = wrap_image(x,y,z)
    #dst = cv2.cv.CreateImage(cv_size(c_img), cv2.cv.IPL_DEPTH_8U, 3)
    h,w=cv_size(c_img)
    #rect = np.zeros((4, 2), dtype = "float32")
    rect=np.array([[0,0],[0,w],[h,w],[h,0]],dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(c_img, M, (600, 200))
    #cv2.cv.WarpPerspective(cv.fromarray(c_img), dst, H2)
##    warp=Image.open(warp)
##    warp=warp.rotate(-90)
    cv2.imshow('crop',warp)
    return warp



def resetX(pos):
    global x
    x = float(pos)
    warp_displayImage()

def resetY(pos):
    global y
    y = float(pos)
    warp_displayImage()

def resetZ(pos):
    global z
    z = float(pos)
    warp_displayImage()


def roi(img, vertices):
    mask=np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    #cv2.imshow('mask',mask)
    
   # cv2.fillPoly(mask,np.int32[vertices],255)
    masked=cv2.bitwise_and(img,mask)
    return masked
def draw_line(img,lines):
    try:
        i=1
        for line in lines:
            for i in range(0,len(line)):
                       
                coord=line[i]
##                if coord[0]!=0 and coord[1]!=0:
                print coord
                cv2.line(img,(coord[0],coord[1]),(coord[2],coord[3]),[255,255,255],3)
    except:
        pass
def process_img(p_img):
    
    
    p_img=cv2.cvtColor(p_img,cv2.COLOR_BGR2GRAY)
    p_img=cv2.Canny(p_img,threshold1=100,threshold2=150)
    p_img=cv2.GaussianBlur(p_img,(9,9),1)
    #vertices = np.array([[[10,360],[10,260],[200,220],[400,220],[600,260],[600,360],
                         #]], np.int32)
    #vertices=np.array([[[0,360],[0,360],[30,20],[30,30]]],np.int32)
    #p_img=roi(p_img,vertices)
    lines=cv2.HoughLinesP(p_img,1,np.pi/280,180,np.array([]),1,0)
    draw_line(p_img,lines)
##    try:
##        l1, l2 = draw_lanes(p_img,lines)
##        cv2.line(p_img, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
##        cv2.line(p_img, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
##    except Exception as e:
##      #  print(str(e))
##        pass
    return p_img    


frame1=cv2.imread('IMG_20170624_155626.jpg')
frame1=cv2.resize(frame1,(640,360))
c_img=frame1[200:310,:]
cv2.namedWindow('undistorted')
cv2.cv.CreateTrackbar("X axis", "undistorted", int(x),600, resetX)
cv2.cv.CreateTrackbar("Y axis", "undistorted", int(y),600, resetY)
cv2.cv.CreateTrackbar("Z axis", "undistorted", int(z),600, resetZ)
frame=warp_displayImage()
p_img=process_img(frame)
cv2.imshow('frame',frame1)
cv2.imshow('edge',p_img)
##if cv2.waitKey(1)& 0xFF == ord('q'):
##    break
cv.WaitKey(0)
cap.release()
cv2.waitKey(0)& 0xFF == ord('q')
cv2.destroyAllWindows()

