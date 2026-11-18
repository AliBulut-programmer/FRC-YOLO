from ultralytics import YOLO
import numpy as np

class RobotDetection:
    def __init__ (self, model_path: str, conf_thresh: float =0.3, imgsz: int = 1024, IOU: float = 0.45):
        #use self so these parameters can be used later on in other functions as
        self.model = YOLO(model_path) 
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz
        self.iou= IOU
    
    def predict(self, frame: np.ndarray) -> list:
        #make prediction
        results = self.model(
            frame,
            conf= self.conf_thresh,
            imgsz= self.imgsz,
            iou= self.iou,
            verbose= False
        )[0]
        return results
        
        