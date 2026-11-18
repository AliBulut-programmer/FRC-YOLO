from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.CompositeVideoClip import concatenate_videoclips
import os

file_path= "unannotated_vids"
final_sequence = []
for i in os.listdir(file_path):
    vid = VideoFileClip(os.path.join("unannotated_vids",i)).subclipped(5,170)
    final_sequence.append(vid)
merged = concatenate_videoclips(final_sequence)
merged.write_videofile("unannotated_vids/final_output.mp4", codec="libx264")
