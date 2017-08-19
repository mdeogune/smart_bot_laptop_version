import cv2
import os
import numpy as np
import time
os.chdir('/home/mukesh-deo/Downloads')


def draw_line(img,lines):
    try:
        i=1
        for line in lines:
            for i in range(0,len(line)):
                       
                coord=line[i]
                
                cv2.line(img,(coord[0],coord[1]),(coord[2],coord[3]),[255,255,255],3)
    except:
        pass

def process_img(p_img):
    
    
    p_img=cv2.cvtColor(p_img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',p_img)
    p_img=cv2.Canny(p_img,threshold1=100,threshold2=300)
    #cv2.imshow('canny',p_img)
    p_img=cv2.GaussianBlur(p_img,(3,3),0)
    #cv2.imshow('blur',p_img)
    lines=cv2.HoughLinesP(p_img,1,np.pi/180,100,np.array([]),20,15)
    draw_line(p_img,lines)
    return p_img


last_time = time.time()

img=cv2.imread('1.jpg')
#cv2.imshow('img',img)
p_img=process_img(img)
    #print('Loop took {} seconds'.format(time.time()-last_time))

WINDOW_NAME = 'Imag'

        #cv2.startWindowThread()

        # Display an image
cv2.imshow(WINDOW_NAME,p_img)
if cv2.waitKey(25) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
    

    
