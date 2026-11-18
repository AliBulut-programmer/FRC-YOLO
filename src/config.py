import cv2

MODEL_PATH = "runs/detect/frc_scouting/custom_annotated_augmented_v1/weights/best.pt"
CONF_THRESH = 0.1
TRACK_THRESH = 0.2
LOST_TRACK_BUFFER = 10
MIN_CONSECUTIVE_FRAME = 5
DETECTION_IOU = 0.2
INERTIA = 0.35

IMG_SIZE = 1024
DEVICE = 0
LETTERBOX_COLOR = (144, 144, 144)
VIDEO_PATH = "C:/Users/alibu/Desktop/FRC-YOLO/raw_videos/Einstein Final Tiebreaker - FIRST Championship - FIRST Robotics Competition_j8wz5vw5XfE.mp4"
FIELD_SIZE = (1657, 800)  # width height 1cm->1pixel
HOMOGRAPHY_CSV_PATH = "src/homography/initiation/homography.csv"
BBOX_THRESH_CALC_FRAME = 2

FOURCC = cv2.VideoWriter_fourcc(*"mp4v")


STARTING_FRAME_HOMOGRAPHY_MAP = 500
# test parameters
COLOR = (0, 255, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
START_FRAME_COUNT = 300
END_FRAME_COUNT = 1000
