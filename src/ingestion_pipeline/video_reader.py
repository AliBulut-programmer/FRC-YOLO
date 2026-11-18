import os
import yt_dlp


def download_video(url, output_dir="./raw_videos"):
    # create output directory if it doesnt exit alr
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]",
        "outtmpl": os.path.join(output_dir, "%(title)s_%(id)s.%(ext)s"),
    }

    print(f"downloading {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print("download complete")


if __name__ == "__main__":
    target_url = "https://www.youtube.com/watch?v=j8wz5vw5XfE&t=47s"
    download_video(target_url)
