import cv2 as cv
import numpy as np

def findDepth(left_X,right_X):
    return (100 / abs(left_X - right_X + 0.0001))

blue_arr = []
green_arr = []
red_arr =[]
disp_list = []

img_L = cv.imread('cam_photo/left_1_resize.jpg')
img_R = cv.imread('cam_photo/right_1_resize.jpg')
blank = np.zeros(img_L.shape[:2], dtype='uint8')

sift = cv.SIFT_create()
kp_L, desp_L = sift.detectAndCompute(img_L, None)
kp_R, desp_R = sift.detectAndCompute(img_R, None)

index_params= dict(algorithm=0,
                    table_number=6,  
                    key_size=10,    
                    multi_probe_level=1) 

search_params = {}
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desp_L, desp_R, k=2)

good_matches = []
for m,n in matches:
    if m.distance < 0.5 * n.distance:
        good_matches.append(m)

for i in good_matches:
    D = findDepth(kp_L[i.queryIdx].pt[0], kp_R[i.trainIdx].pt[0])
    disp_list.append([kp_L[i.queryIdx],D])   

for i,j in disp_list:
    if j < 4:
        red_arr.append(i)
    elif j >= 4 and j < 8:
        green_arr.append(i)
    else:
        blue_arr.append(i)

img_fin = img_L.copy()
img_temp_L = img_L.copy()
img_temp_R = img_R.copy()
img_temp_L = cv.drawKeypoints(img_temp_L,kp_L,0,(100,100,0),
                              flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
img_temp_R = cv.drawKeypoints(img_temp_R,kp_R,0,(100,100,0),
                              flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
img_fin = cv.drawKeypoints(img_fin,red_arr,0,(0,0,255),
                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
img_fin = cv.drawKeypoints(img_fin,green_arr,0,(0,255,0),
                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
img_fin = cv.drawKeypoints(img_fin,blue_arr,0,(255,0,0),
                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
blank = cv.drawKeypoints(blank,red_arr,0,(0,0,255),
                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
blank = cv.drawKeypoints(blank,green_arr,0,(0,255,0),
                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
blank = cv.drawKeypoints(blank,blue_arr,0,(255,0,0),
                 flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


cv.imshow('depth_map',blank)
cv.imshow('depth_pic',img_fin)
cv.imshow('left_keys',img_temp_L)
cv.imshow('right_keys',img_temp_R)
cv.waitKey(0)
cv.destroyAllWindows()