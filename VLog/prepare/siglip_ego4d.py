import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm
import decord
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and processor
model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")

# Define function to extract features from video using decord
def extract_features_from_video(video_path, fps, model, processor, device):
    vr = decord.VideoReader(video_path)
    video_fps = vr.get_avg_fps()
    interval = int(video_fps / fps)

    frames = []
    for i in range(0, len(vr), interval):
        frame = vr[i]
        # Convert to PIL image
        frame_rgb = frame.asnumpy()
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)

    # Use Siglip model to extract features
    batch_size = 32  # Set batch size to speed up processing
    features = []
    for i in tqdm(range(0, len(frames), batch_size), desc="Extracting features"):
        batch_frames = frames[i:i+batch_size]
        inputs = processor(images=batch_frames, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
        features.append(image_features.cpu())

    # Concatenate all batches of features into a single tensor
    features_tensor = torch.cat(features, dim=0)
    return features_tensor

# Define main function to iterate through directory and extract features in a distributed manner
def process_videos_in_directory(input_dir, output_dir, fps, total_nodes, cur_node, total_gpus):
    device = f"cuda:{cur_node % total_gpus}"
    model.to(device)
    
    # Get all video file paths
    all_videos = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".mp4"):
                all_videos.append(os.path.join(root, file))
    
    # Divide videos to process among nodes
    videos_to_process = all_videos[cur_node::total_nodes]
    
    for video_path in videos_to_process:
        try:
            relative_path = os.path.relpath(video_path, input_dir)
            output_path = os.path.join(output_dir, os.path.splitext(relative_path)[0] + ".pt")
    
            # Skip if feature file already exists
            if os.path.exists(output_path):
                print(f"Node {cur_node}: Skipping existing file: {output_path}")
                continue
    
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
            # Extract video features and save
            print(f"Node {cur_node}: Processing video: {video_path}")
            features = extract_features_from_video(video_path, fps, model, processor, device)
            torch.save(features, output_path)
            print(f"Node {cur_node}: Saved features to: {output_path}")
        except:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="/blob/xiyin1wu2_maskrcnn/data/datasets/ego4d/data/v1/clips", help="Path to the input directory")
    parser.add_argument('--fps', type=int, default=2, help="Frames per second to extract")
    parser.add_argument('--output_dir', default="/blob/v-lqinghong/data/Ego_database/ego4d/siglip-so400m-patch14-384-fps2",  help="Path to the output directory")
    parser.add_argument('--total_nodes', type=int, default=1, help="Total number of nodes")
    parser.add_argument('--cur_node', type=int, default=0, help="Current node index (0-based)")
    parser.add_argument('--total_gpus', type=int, default=1, help="Total number of GPUs available")
    args = parser.parse_args()

    # Process videos in directory
    process_videos_in_directory(args.input_dir, args.output_dir, args.fps, args.total_nodes, args.cur_node, args.total_gpus)
