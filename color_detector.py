# Notes
''' 
check hsv color @ https://alloyui.com/examples/color-picker/hsv  &&  https://www.rapidtables.com/web/color/RGB_Color.html
calculation tutorial @ https://www.youtube.com/watch?v=x4qPhYamRDI

h = hue (color in degrees) [0-360]
S = saturation [0%-100%]
v = value [0%-100%]

converted
0 > H > 360 -> OpenCV range = H/2 (0 > H > 180)
0 > S > 1 -> OpenCV range = 255*S (0 > S > 255)
0 > V > 1 -> OpenCV range = 255*V (0 > V > 255)

h value divide by 4

'''

import cv2
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="Path to video or left blank to capture from camera")
args = ap.parse_args()

cap = cv2.VideoCapture(args.video if args.video else 0)

boundaries = { # hsv color boundaries
'red' : np.array([[0,120,5], [5,255,255], [161, 125, 5], [179, 255, 255]]),  
'blue' : np.array([[98, 109, 2], [116, 255, 255]]),
'green' : np.array([[38, 100, 100], [75, 255, 255]]),
'yellow' : np.array([[20,90,20],[30,255,255]]), 
'orange' : np.array([[6,100,150],[14,255,255]])
}

BGR_colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}

while True:
	ret, frame = cap.read()

	if not ret:
		break

	blur = cv2.GaussianBlur(frame, (11, 11), 0) # reduce noises, smoothing image
	hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) # convert BGR image to HSV image

	for color, code in boundaries.items():
		if color == 'red':  # red has 2 ranges in HSV
			low1, high1, low2, high2 = code
			mask1 = cv2.inRange(hsv, low1, high1)
			mask2 = cv2.inRange(hsv, low2, high2)
			mask = mask1 + mask2
		else:
			low, high = code
			mask = cv2.inRange(hsv, low, high)

		kernel = np.ones((10,10),np.uint8) # Unsigned(no negative value) int 8bits(0-255)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # remove false positives. remove pixels(noises) from image (outside detected shape)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # remove false negatives. inside detected shape

		_,cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


		for index, con in enumerate(cnts):
			(x,y),radius = cv2.minEnclosingCircle(con)
			if radius < 0.05: # ignore noises 
				pass
			else:
				center = (int(x),int(y))  
				cv2.drawContours(frame, cnts, -1, BGR_colors[color], 5)
				cv2.putText(frame,color, center, cv2.FONT_HERSHEY_SIMPLEX, 0.6,BGR_colors[color],2)


	cv2.imshow('Color Detector',frame)

	key = cv2.waitKey(10)

	if key == ord('q'):   # press q to quit
		break

cap.release()
cv2.destroyAllWindows()



