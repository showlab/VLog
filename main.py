import argparse
from models.vlog import Vlogger

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default='examples/travel_in_roman.mp4')
    parser.add_argument('--alpha', default=10, type=int, help='Determine the maximum segment number for KTS algorithm, the larger the value, the fewer segments.')
    parser.add_argument('--beta', default=1, type=int, help='The smallest time gap between successive clips, in seconds.')
    parser.add_argument('--data_dir', default='./examples', type=str, help='Directory for saving videos and logs.')
    parser.add_argument('--tmp_dir', default='./tmp', type=str, help='Directory for saving intermediate files.')
    
    # * Models settings *
    parser.add_argument('--openai_api_key', default='xxx', type=str, help='OpenAI API key')
    parser.add_argument('--image_caption', action='store_true', dest='image_caption', default=True, help='Set this flag to True if you want to use BLIP Image Caption')
    parser.add_argument('--dense_caption', action='store_true', dest='dense_caption', default=True, help='Set this flag to True if you want to use Dense Caption')
    parser.add_argument('--feature_extractor', default='openai/clip-vit-base-patch32', help='Select the feature extractor model for video segmentation')
    parser.add_argument('--feature_extractor_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu')
    parser.add_argument('--image_captioner', choices=['blip2'], dest='captioner_base_model', default='blip2', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
    parser.add_argument('--image_captioner_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
    parser.add_argument('--dense_captioner_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
    parser.add_argument('--audio_translator', default='large')
    parser.add_argument('--audio_translator_device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo'], default='gpt-3.5-turbo')

    args = parser.parse_args()

    vlogger = Vlogger(args)
    vlogger.video2log(args.video_path)

    print("Let's chat with your video!")
    while True:
        question = input("Human: ")
        answer = vlogger.chat2video(question)
