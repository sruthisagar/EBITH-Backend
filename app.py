from flask import Flask, render_template, request,jsonify
import numpy as np
import argparse
import time
import cv2
import os

app = Flask(__name__)
# model = load_model('model.h5')
# model.make_predict_function()

def predict_label(img_path):
	# i = image.load_img(img_path, target_size=(100,100))
	# i = image.img_to_array(i)/255.0
	# i = i.reshape(1, 100,100,3)
	# p = model.predict(i)
	# if(p[0][0] < p[0][1]):
	# 	return "Mask on"
	# else:
	# 	return "No Mask detected"
	labelsPath = os.path.sep.join(["yolo3", "coco.names"])
	LABELS = open(labelsPath).read().strip().split("\n")
	np.random.seed(42)
	COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
	weightsPath = os.path.sep.join(['yolo3', "yolov3.weights"])
	configPath = os.path.sep.join(["yolo3", "yolov3.cfg"])

	print("[INFO] loading YOLO from disk...")
	net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


	image = cv2.imread(img_path)
	(H, W) = image.shape[:2]


	ln = net.getLayerNames()
	ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]


	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()


	print("[INFO] YOLO took {:.6f} seconds".format(end - start))


	boxes = []
	confidences = []
	classIDs = []
	ID = 0


	for output in layerOutputs:

		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]


			if confidence > 0.5:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))


				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)


	idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)


	if len(idxs) > 0:
		list1 = []
		for i in idxs.flatten():

			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			centerx = round((2*x + w)/2)
			centery = round((2*y + h)/2)
			if centerX <= W/3:
				W_pos = "left "
			elif centerX <= (W/3 * 2):
				W_pos = "center "
			else:
				W_pos = "right "

			if centerY <= H/3:
				H_pos = "top "
			elif centerY <= (H/3 * 2):
				H_pos = "mid "
			else:
				H_pos = "bottom "
			list1.append(LABELS[classIDs[i]] + ' at ' + H_pos + W_pos)

		description = ', '.join(list1)
		print(description)
# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Flask app....."

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + img.filename	
		img.save(img_path)
		p = predict_label(img_path)

	return jsonify({'prediction':p,'img_path':img_path})


if __name__ =='__main__':
	app.run(debug=True)