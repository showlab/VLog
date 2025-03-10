import os
import torch
import cv2
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import torch.multiprocessing as mp
import argparse

# 加载模型和处理器
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# 定义从视频中提取特征的函数
def extract_features_from_video(video_path, fps, model, processor, device):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps)

    frames = []
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % interval == 0:
            # 转换为 PIL 图像
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            frames.append(pil_image)
        
        frame_count += 1
        success, frame = cap.read()

    cap.release()

    # 使用 Siglip 模型提取特征
    batch_size = 32  # 设置批次大小，以加快处理速度
    features = []
    for i in tqdm(range(0, len(frames), batch_size), desc=f"Extracting features on {device}"):
        batch_frames = frames[i:i+batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        features.append(image_features.cpu())

    # 将所有批次的特征拼接为一个张量
    features_tensor = torch.cat(features, dim=0)
    return features_tensor

# 定义主函数来遍历目录并提取特征
def process_videos_in_directory(input_dir, output_dir, fps, total_nodes, cur_node, total_gpus):
    device = f"cuda:{cur_node % total_gpus}"
    model.to(device)
    
    # 获取所有视频文件路径
    all_videos = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                all_videos.append(os.path.join(root, file))
    
    # 根据节点号划分需要处理的视频
    videos_to_process = all_videos[cur_node::total_nodes]
    
    for video_path in videos_to_process:
        try:
            relative_path = os.path.relpath(video_path, input_dir)
            output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".pt")
    
            # 如果特征文件已存在，则跳过
            if os.path.exists(output_path):
                print(f"Node {cur_node}: Skipping existing file: {output_path}")
                continue
    
            # 创建输出目录（如果不存在）
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
            # 提取视频特征并保存
            print(f"Node {cur_node}: Processing video: {video_path}")
            features = extract_features_from_video(video_path, fps, model, processor, device)
            torch.save(features, output_path)
            print(f"Node {cur_node}: Saved features to: {output_path}")
        except:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="/blob/v-lqinghong/data/Ego_database/coin/videos", help="Path to the input directory")
    parser.add_argument('--fps', type=int, default=2, help="Frames per second to extract")
    parser.add_argument('--output_dir', default="/blob/v-lqinghong/data/Ego_database/coin/siglip-so400m-patch14-384-fps2",  help="Path to the output directory")    
    parser.add_argument('--total_nodes', type=int, default=1, help="Total number of nodes")
    parser.add_argument('--cur_node', type=int, default=0, help="Current node index (0-based)")
    parser.add_argument('--total_gpus', type=int, default=1, help="Total number of GPUs available")
    args = parser.parse_args()

    # 处理目录下的视频
    process_videos_in_directory(args.input_dir, args.output_dir, args.fps, args.total_nodes, args.cur_node, args.total_gpus)
