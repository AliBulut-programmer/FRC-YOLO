import cv2
import math
from src.config import COLOR, FONT, FONT_SCALE,THICKNESS
def annotate(frame, frame_count, detections, move, center_positions, past_center_positions):
    #write frame number on top left
    #if move == None:
        #return frame
    cv2.putText(frame, str(frame_count), (500,20), FONT,FONT_SCALE,COLOR, THICKNESS, )
    for  xyxy, _, confidence, class_id, tracker_id, _ in detections:
        p1 = (int(xyxy[0]), int(xyxy[1]))
        p2 = (int(xyxy[2]), int(xyxy[3]))
        p3 = (int(xyxy[0]), int(xyxy[1])-20)
        p4 = (int(xyxy[0]), int(xyxy[1])-40)
        p5 = (int(xyxy[0]), int(xyxy[1])-60)
        w = -int(xyxy[0]) + int(xyxy[2])
        h = -int(xyxy[1]) + int(xyxy[3])
        color = (int((tracker_id * 37) % 255), int((tracker_id * 83) % 255), int((tracker_id * 131) % 255))
        cv2.rectangle(frame, p1, p2, color, THICKNESS )
        cv2.putText(frame, f" {confidence:.2f} | {tracker_id}", p1, FONT, FONT_SCALE, color, THICKNESS)
        if move == None:
            print("no move")
            continue

        cv2.putText(frame, f" v:{move.get(tracker_id, 0):.2f} ", p3, FONT, FONT_SCALE, color, THICKNESS)

        dist = math.dist(center_positions.get(tracker_id,(0,0)),past_center_positions.get(tracker_id,(0,0))) 
        cv2.putText(frame, f" d:{dist:.2f} ", p4, FONT, FONT_SCALE, color, THICKNESS)
        cv2.putText(frame, f" s:{w:.2f},{h:.2f}  ", p5, FONT, FONT_SCALE, color, THICKNESS)
        print(f" s:{w:.2f},{h:.2f}  ")
        cv2.circle(frame, center_positions.get(tracker_id), 5, color, 5)
        cv2.circle(frame, past_center_positions.get(tracker_id), 3, (0,255,0), 3)
        cv2.line(frame_count, center_positions.get(tracker_id),past_center_positions.get(tracker_id), (0,0,255), 2)

    
    return frame
