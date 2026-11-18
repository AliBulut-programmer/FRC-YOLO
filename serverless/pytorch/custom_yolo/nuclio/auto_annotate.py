from ultralytics import YOLO
import cv2
import os
import xml.etree.ElementTree as ET

model = YOLO("runs/detect/frc_scouting/custom_annotated_augmented_v0/weights/best.pt")
video_path = "./unannotated_vids/final_output.mp4"
output_dir = "./cvat_xml_annotations/"
starting_frame= 5100

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)
frame_count = starting_frame
saved_count = int(starting_frame//30)

# Root XML structure for CVAT
annotations = ET.Element("annotations")
meta = ET.SubElement(annotations, "meta")
project = ET.SubElement(meta, "task")
ET.SubElement(project, "name").text = "FRC Auto Annotations"

print("Generating CVAT XML annotations...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_count >50000:
        break
    if frame_count<starting_frame:
        continue
        
    # Check every 30 frames
    if frame_count % 30 == 0:
        results = model.predict(frame, conf=0.3, imgsz=1024, verbose=False)
        
        # Get actual frame dimensions dynamically
        h, w, _ = frame.shape
        
        image_el = ET.SubElement(annotations, "image", id=str(frame_count), name=f"frame_{frame_count:06d}.jpg", width=str(w), height=str(h))
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label_name = model.names[cls_id]
                xyxy = box.xyxy[0].tolist() # [xmin, ymin, xmax, ymax]
                
                ET.SubElement(image_el, "box", 
                              label=label_name, 
                              xtl=f"{xyxy[0]:.2f}", 
                              ytl=f"{xyxy[1]:.2f}", 
                              xbr=f"{xyxy[2]:.2f}", 
                              ybr=f"{xyxy[3]:.2f}", 
                              occluded="0")
        
        cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:06d}.jpg"), frame)
        saved_count += 1

    frame_count += 1

cap.release()

# Save the final XML file cleanly without trailing semicolons
tree = ET.ElementTree(annotations)
xml_path = os.path.join(output_dir, "annotations.xml")
tree.write(xml_path, encoding="utf-8", xml_declaration=True)

print(f"Done! Processed up to frame {frame_count}. Saved annotations.xml and {saved_count} frames to {output_dir}")