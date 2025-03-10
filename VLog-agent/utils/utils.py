import os
import pdb
import logging
import subprocess

def format_time(seconds):
    hours = seconds // 3600
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def download_video(url, save_dir='./examples', size=768):
    save_path = f'{save_dir}/{url}.mp4'
    cmd = f'yt-dlp -S ext:mp4:m4a --throttled-rate 5M -f "best[width<={size}][height<={size}]" --output {save_path} --merge-output-format mp4 https://www.youtube.com/embed/{url}'
    if not os.path.exists(save_path):
        try:
            subprocess.call(cmd, shell=True)
        except:
            return None
    return save_path

def logger_creator(video_id):
    # set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'./examples/{video_id}.log', mode='w')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":
    download_video('outcGtbnMuQ', save_dir='./examples', size=768)
