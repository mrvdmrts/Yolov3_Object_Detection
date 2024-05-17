import streamlit as st
import cv2
import tempfile
import numpy as np
import urllib.request
import wget
import requests
import pafy
import yt_dlp as youtube_dl
from pytube import YouTube 

@st.cache_data
@st.cache
def download_weights():
    weights_url="https://pjreddie.com/media/files/yolov3.weights"
    urllib.request.urlretrieve(weights_url, "yolov3.weights")
    urllib.request.urlopen(weights_url)
    requests.get(weights_url)
    wget.download(weights_url, "yolov3.weights")

download_weights()

# Load YOLO
@st.cache_resource
def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print(output_layers)
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, output_layers, classes

net, output_layers, classes = load_yolo()

# Streamlit app
st.title("Object Counting with YOLO")
#st.sidebar.image("datasets/images/logo.png", use_column_width=True)
st.sidebar.title("Object Counting with YOLO")
radio=st.sidebar.radio("Video", ["From File", "From URL"])
uploaded_file=None
if radio == "From File":
    st.sidebar.write("Upload a video file to count cars using YOLO.")
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])   
    
else:
    #get video link from user
    uploaded_file=st.sidebar.text_input("Enter the URL of the video file")

#select an object to detect
selected_object=st.sidebar.selectbox("Select an object to detect",["car", "person", "bike", "cat", "dog",'flower'])
#detect button
st.sidebar.write("Click the button below to detect the selected object")
if st.sidebar.button("Detect"):
    st.write("Detecting...")
    st.write("Please wait for a while")
    st.write("The detected objects will be shown below")
    st.write("The number of detected objects will be shown on the top left corner of the video")
    st.write("The detected objects will be highlighted in green")
    st.write("The selected object is: ",selected_object)
    cap=None
if uploaded_file is not None:
    if radio == "From File":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
       #create a cap object from video url
       st.video(uploaded_file)
    
    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Information to show on screen
        class_ids = []
        confidences = []
        boxes = []

        # For each detection from each output layer, get the confidence, class id, bounding box parameters
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and classes[class_id] == selected_object:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-max suppression to eliminate redundant overlapping boxes with lower confidences
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        object_count = 0
        for i in range(len(boxes)):
            if i in indexes:
                object_count += 1
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the resulting frame with car count
        cv2.putText(frame, selected_object+f" : {object_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        stframe.image(frame, channels="BGR")

    #cap.release()
    #cv2.destroyAllWindows()
