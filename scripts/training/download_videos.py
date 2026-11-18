import os
import sys
import yt_dlp
from links import links

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.config import VIDEO_PATH
def download_video(url, output_dir="./unannotated_vids"):
    #create output directory if it doesnt exit alr
    os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        
        "format": 'bestvideo[ext=mp4]', 
        'outtmpl': os.path.join(output_dir, '%(title)s_%(id)s.%(ext)s')
    }
    
    print(f"downloading {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
if __name__ == "__main__":
    """
    for target_url in links:
        download_video(target_url)
        """
    download_video("https://www.youtube.com/watch?v=j8wz5vw5XfE&t=16s")