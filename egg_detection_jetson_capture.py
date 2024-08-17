# egg_detection_jetson_capture.py, this code uses jetson_utils library to capture video and a custom yolo model to detect eggs from a camera source.
# Copyright (C) 2024 DEBUG NOMAD SLU, www.debugnomad.com

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
from ultralytics import YOLO
import torch
import jetson_utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}") # cuda
model = YOLO('./models/yolo_egg.pt').to(device) # load a custom model

boundary_line = 250 + 250
boundary_line2 = 300 + 250
color_in = (64,224,208)
color_out =(0,255,255)
color_done = (255,182,193)
thickness = 1
alpha = 0.7
id_set = set()

frame_count = 0

max_per_frame = 0

# Initialize video capture using jetson.utils
source = 'v4l2:///dev/video1'  # Adjust according to your USB camera device

cap = jetson_utils.videoSource(source)
#output = jetson_utils.videoOutput()
output = jetson_utils.videoOutput()
img = cap.Capture()

# Loop through the video frames
while cap.IsStreaming():
    # Read a frame from the video
    if img:
        # Convert the image from jetson.utils to a numpy array
        np_img = jetson_utils.cudaToNumpy(img)

        # Convert RGBA to RGB
        np_img_rgb = np_img[:, :, :3]

        # Run YOLOv8 inference on the frame
        results = model.track(np_img_rgb, persist=True, conf=0.5, device=device)

        # Annotate the frame 
        annotated_frame = np_img.copy()
        frame_overlay = np_img.copy()
        cv2.line(frame_overlay, (0, boundary_line), (annotated_frame.shape[1], boundary_line), (0, 255, 0), thickness)
        cv2.line(frame_overlay, (0, boundary_line2), (annotated_frame.shape[1], boundary_line2), (0, 255, 0), thickness)
        
        bboxs, ids = results[0].boxes.data, results[0].boxes.id
        coordiantes = []
        if ids is not None:
            for bbox, id in zip(bboxs, ids):
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                coordiantes.append([id.item(), x1, y1, x2, y2])
                if y1 < boundary_line:
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color_out, -1)
                elif y1 > boundary_line and y1 < boundary_line2:
                    id_set.add(id.item())
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color_in, -1)
                elif y1 > boundary_line2:
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color_done, -1)
        cv2.putText(annotated_frame, f'Count: {len(id_set)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        frame_overlay=cv2.addWeighted(frame_overlay, alpha, annotated_frame,1-alpha, gamma=0)
        # cv2.imshow("YOLOv8 Inference", frame_overlay)
        output.Render(jetson_utils.cudaFromNumpy(frame_overlay))

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.Close()
            break
    img = cap.Capture()

# Release the video capture object and close the display window
cap.Close()