import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import sys
import numpy as np

root_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(root_dir))

from src.config import VIDEO_PATH

def run_trapezoid_calibration():
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    ok, frame = cap.read()
    
    if not ok:
        print("Error reading video frame.")
        return

    height, width = frame.shape[:2]
    
    fig, ax = plt.subplots(figsize=(16, 9))
    
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.grid(True, which='both', linestyle='--', linewidth=.7, color='green')
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=.3, color='black')

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb_frame, origin='upper', extent=[0, width, height, 0])
    
    padding = 1000
    ax.set_xlim(-padding, width + padding)
    ax.set_ylim(height + padding, -padding)

    # Initialize a default trapezoid near the center of the frame
    points = [
        (width * 0.35, height * 0.4),  # Top-left
        (width * 0.65, height * 0.4),  # Top-right
        (width * 0.80, height * 0.8),  # Bottom-right
        (width * 0.20, height * 0.8)   # Bottom-left
    ]
    
    dragging_idx = None
    
    line_plot, = ax.plot([], [], color='red', linewidth=3, marker='+', markersize=5, markerfacecolor='red', alpha=0.4 )

    def update_plot():
        x_coords = [p[0] for p in points] + [points[0][0]]
        y_coords = [p[1] for p in points] + [points[0][1]]
        line_plot.set_data(x_coords, y_coords)
        fig.canvas.draw_idle()

    def on_press(event):
        nonlocal dragging_idx
        if event.xdata is None or event.ydata is None:
            return
        
        # Check if clicking near any of the 4 vertex points to drag it
        for idx, pt in enumerate(points):
            dist = np.hypot(pt[0] - event.xdata, pt[1] - event.ydata)
            if dist < 30:  # Hit-box tolerance for grabbing vertices
                dragging_idx = idx
                return

    def on_motion(event):
        nonlocal dragging_idx
        if event.xdata is None or event.ydata is None:
            return
        
        if dragging_idx is not None:
            points[dragging_idx] = (event.xdata, event.ydata)
            update_plot()

    def on_release(event):
        nonlocal dragging_idx
        dragging_idx = None

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    
    # Draw initial trapezoid state
    update_plot()
    
    print("Trapezoid Calibration Tool Active:")
    print(" - Click and drag any of the 4 red corner points to match the field perspective.")
    print(" - Close the plot window when finished to output the final coordinates.")
    
    plt.show()

    print("\nFinal Captured Trapezoid Calibration Points (X, Y):")
    for pt in points:
        print(f"({pt[0]:.1f}, {pt[1]:.1f})")
    csv_save_data = np.array(points,dtype=float)
    np.savetxt('scripts/homography/homography.csv', csv_save_data, delimiter=',', header='X,Y', comments='', fmt= '%.1f')
        

if __name__ == "__main__":
    run_trapezoid_calibration()