from flask import Flask,render_template,Response,flash,request,redirect,url_for
import cv2
import numpy as np
import face_recognition as face
import PIL
import os
from datetime import datetime as time
import winsound
import smtplib,email,ssl

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("home.html")

@app.route("/land_page")
def land_page():
	return render_template("landing.html")

# The send_email function send email to you once identified person caught on camera
fromaddr = "place_your_email_here"
toaddr = "place_recipient_email_here"
pwd = "place_your_email_password_here"

def send_email(attach_src,name):
	msg = MIMEMultipart()
	msg['From'] = fromaddr
	msg['To'] = toaddr
	msg['Subject'] = "IMAGE"
	body = "Identified Image"

	msg.attach(MIMEText(body, 'plain'))
	attach_src = open(attach_src, "rb")
	p = MIMEBase('application', 'octet-stream')
	p.set_payload((attach_src).read())
	encoders.encode_base64(p)
	p.add_header('Content-Disposition', "attachment; filename= %s" % f"{name}.jpg")
	msg.attach(p)
	text = msg.as_bytes()
	con = smtplib.SMTP('smtp.gmail.com', 587)
	con.starttls()
	con.login(fromaddr,pwd)
	con.sendmail(fromaddr, toaddr, text)
	con.quit()

@app.route("/login_through_camera")
def login_through_camera():
	cam = cv2.VideoCapture(0)
	img_path = "static/train_images"
	saved_img_path = "static/saved_images"
	img_encs,img_names = list(),list()
	login_name = list()
	for image in os.listdir(img_path):
		img_name = image.split(".")[0]
		img_names.append(img_name)
		img = os.path.join(img_path,image)
		img = PIL.Image.open(img)
		img_array = np.array(img)
		img_enc = face.face_encodings(img_array)[0]
		img_encs.append(img_enc)
	# named_person = img_names
	while True:
		success,frame = cam.read()
		frame_locs = face.face_locations(np.array(frame))
		frame_encs = face.face_encodings(np.array(frame))
		new_names = list()
		for enc in frame_encs:
			results = face.compare_faces(img_encs,enc)
			distance = face.face_distance(img_encs,enc)
			best_index = np.argmin(distance)
			if results[best_index]:
				name = img_names[best_index]
				new_names.append(name)
			else:
				name = "NoMatch"
				new_names.append(name)
		for index,(person_name,(top,right,bottom,left)) in enumerate(zip(new_names,frame_locs)):
			frame = cv2.rectangle(
					img = frame,
					pt1 = (left,top),
					pt2 = (right,bottom),
					color = (0,255,0),
					thickness = 2
					)
			frame = cv2.putText(
		            img = frame,
		            text = person_name,
		            org = (left+6,top-6),
		            fontScale = 1,
		            fontFace = cv2.FONT_HERSHEY_COMPLEX,
		            thickness = 2,
		            color = (255,0,255),
		            )
			frame = cv2.putText(
		            img = frame,
		            text = f'Arrival At:{time.now().strftime("%H:%M:%S")}',
		            org = (70,70),
		            fontScale = 1,
		            fontFace = cv2.FONT_HERSHEY_COMPLEX,
		            thickness = 2,
		            color = (255,0,255),
		            )
			if person_name in img_names: #can be modified to include all person by set "NoMatch"
				cv2.imwrite(os.path.join(saved_img_path,f"{person_name}.{index}.jpg"),frame)
				winsound.Beep(frequency=2000,duration=1000)
				saved_img = os.path.join(saved_img_path,f"{person_name}.{index}.jpg")
				try: # Handling exception in case of no internet to send email hence stored internally
					send_email(saved_img,person_name)
				except:
					pass
					# flash("No internet connection",success)
				# saved_img = cv2.imread(saved_img)
				# cv2.imshow("IMG",saved_img) 
				# cv2.waitKey(0)
				return redirect(url_for("login_through_camera"))
			else:
				continue
		if cv2.waitKey(1) == ord("q"):
			break
	cam.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	app.run(port=8081,debug=True)  