# HOW TO USE?
# python transform_images --i input_folder
# (the script generates new .jpg images based on the ones contained in input
# folder and all its sub-directories. Then remove the originals images.)

import numpy as np
import cv2
import argparse
import os
from scipy import ndimage

NB_TRANS = 100	  #Number of total transformations per image
MAX_HOR_TRANS = 1 #maximum horizontal translation in pixels 
MAX_VER_TRANS = 1 #maximum vertical translation in pixels
MAX_ROT_ANGLE = 1 #maximum rotation angle in degrees
MAX_BLUR = 4 	  #maximum blur kernel size

def add_noise(img):
	# trans = np.zeros((img.shape))
	# (h, w, d) = trans.shape
	# for x in range(h):
	# 	for y in range(w):
	# 		trans[x,y:] = np.random.rand(d)*255
	rows,cols, d = img.shape[0],img.shape[1], img.shape[2]
	trans = img
	for y in range(rows/2):
		for x in range(cols/2):
			if np.mean(trans[y,x,:]) < 1:
				trans[y,x,:]=np.random.rand(d)*255
			if np.mean(trans[y+rows/2,x,:]) < 1:
				trans[y+rows/2,x,:]=np.random.rand(d)*255
			if np.mean(trans[y+rows/2,x+cols/2,:]) < 2:
				trans[y+rows/2,x+cols/2,:]=np.random.rand(d)*255
			if np.mean(trans[y,x+cols/2,:]) < 1:
				trans[y,x+cols/2,:]=np.random.rand(d)*255
	return trans


def rotate_image(img, angle):
	if angle !=0:
		rows,cols = img.shape[0],img.shape[1]
		# mat = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
		# trans = cv2.warpAffine(img,mat,(cols,rows))
		trans = ndimage.rotate(img, angle, reshape=False)
		trans = add_noise(trans)
		return trans
	else:
		return img

def complete_image(img, h, w, d):
	patch = np.zeros((h, w, d))
	for x in range(h):
		for y in range(w):
			patch[x,y:] = np.random.rand(d)*255
	rows,cols = img.shape[0],img.shape[1]
	dif = h - rows
	if dif%2!=0:
		x1 = dif/2-0.5+1
		x1 = int(x1)
		x2 = dif/2+0.5+1
		x2 = int(x2)
	else:
		x1 = dif/2
		x1 = int(x1)
		x2 = dif/2
		x2 = int(x2)

	dif2 = w - cols
	if dif2%2!=0:
		y1 = dif2/2-0.5+1
		y1 = int(y1)
		y2 = dif2/2+0.5+1
		y2 = int(y2)
	else:
		y1 = dif2/2
		y1 = int(y1)
		y2 = dif2/2
		y2 = int(y2)
	patch[x1:-x2,y1:-y2,:]=img
	return patch

def translate_image_bis(img, coord):
	if coord !=0:
		patch = np.zeros((img.shape))
		(h, w, d) = patch.shape
		for x in range(h):
			for y in range(w):
				patch[x,y:] = np.random.rand(d)*255
		if coord > 0:
			patch[coord:,:,:] = img[0:-coord,:,:]
		else:
			patch[:coord,:] = img[-coord:,:,:] 
		return patch
	else:
		return img

def translate_image(img, coord):
	if coord !=0:
		patch = np.zeros((img.shape))
		(h, w, d) = patch.shape
		for x in range(h):
			for y in range(w):
				patch[x,y:] = np.random.rand(d)*255
		if coord > 0:
			patch[:,coord:,:] = img[:,0:-coord,:]
		else:
			patch[:,:coord,:] = img[:,-coord:,:]
		return patch
	else:
		return img

def change_brightness(img, factor):
	if factor != 1:
		hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		print img
		h, s ,v = cv2.split(hsv_img)
		v = (v*factor).astype(np.uint8)
		new_img = cv2.merge((h,s,v))
		new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
		return new_img
	else:
		return umg

def zoom(image, factor, h, w ,d):
	image = cv2.resize(image, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
	completed = complete_image(image, h ,w ,d)
	return completed

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Path to the images input folder")
# ap.add_argument("-o", "--output", required=True, help="Path to the images output folder")
args = vars(ap.parse_args())

tot = 0
subdirs = [x[0] for x in os.walk(args["input"])]
print subdirs
# output_dir = args["output"]
for subdir in subdirs:
	nb_files = len([name for name in os.listdir(subdir) if os.path.isfile(name)])
	print nb_files
	print os.listdir(subdir)
	if nb_files < 1:
		for filename in os.listdir(subdir):
			if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
				for i in range(NB_TRANS):
					filepath = os.path.join(subdir, filename)
					print filepath
					img = cv2.imread(filepath)
					print img
					# Blur
					new_blur = np.random.randint(MAX_BLUR)
					img = cv2.blur(img,(new_blur,new_blur))
					x,y,z = img.shape

					new_brightness = 0.7 + 0.05*np.random.randint(7)
					v_trans = np.random.randint(MAX_VER_TRANS*2) - MAX_VER_TRANS
					h_trans = np.random.randint(MAX_HOR_TRANS*2) - MAX_HOR_TRANS
					rot_angle = np.random.randint(MAX_ROT_ANGLE*2) - MAX_ROT_ANGLE

					img = change_brightness(img, new_brightness)
					img = translate_image(img, h_trans)
					img = translate_image_bis(img, v_trans)
					img = rotate_image(img, rot_angle)

					# Saving
					basename = os.path.basename(subdir)
					filepath = os.path.join(args["input"], basename, str(i+1) + "_" + os.path.splitext(filename)[0] + ".jpg")
					cv2.imwrite(filepath,img)
					print(filepath)
				#os.remove(os.path.join(subdir, filename))
