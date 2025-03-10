import os
import gradio as gr
import openai
import requests
import csv
import argparse
from models.vlog import Vlogger

from utils.utils import download_video

prompt_templates = {"Default ChatGPT": ""}

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', default='examples/huaqiang.mp4')
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
parser.add_argument('--image_captioner', choices=['blip', 'blip2'], dest='captioner_base_model', default='blip2', help='blip2 requires 15G GPU memory, blip requires 6G GPU memory')
parser.add_argument('--image_captioner_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, gpu memory larger than 14G is recommended')
parser.add_argument('--dense_captioner_device', choices=['cuda', 'cpu'], default='cuda', help='Select the device: cuda or cpu, < 6G GPU is not recommended>')
parser.add_argument('--audio_translator', default='large')
parser.add_argument('--audio_translator_device', choices=['cuda', 'cpu'], default='cuda')
parser.add_argument('--gpt_version', choices=['gpt-3.5-turbo'], default='gpt-3.5-turbo')

args = parser.parse_args()

vlogger = Vlogger(args)

def get_empty_state():
    return {"total_tokens": 0, "messages": []}


def submit_message(prompt, state):
    history = state['messages']

    if not prompt:
        return gr.update(value=''), [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history)-1, 2)], state

    prompt_msg = { "role": "user", "content": prompt }
    
    try:
        history.append(prompt_msg)
        answer = vlogger.chat2video(prompt)
        history.append({"role": "system", "content": answer}) 

    
    except Exception as e:
        history.append(prompt_msg)
        history.append({
            "role": "system",
            "content": f"Error: {e}"
        })

    chat_messages = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history)-1, 2)]
    return '', chat_messages, state


def clear_conversation():
    vlogger.clean_history()
    return gr.update(value=None, visible=True), gr.update(value=None, interactive=True), None, gr.update(value=None, visible=True), get_empty_state()


def subvid_fn(vid):
    print(vid)
    save_path = download_video(vid)
    return gr.update(value=save_path)


def vlog_fn(vid_path):
    print(vid_path)
    if vid_path is None:
        log_text = "====== Please upload video or provide youtube_id 🤔====="
    else:
        log_list = vlogger.video2log(vid_path)
        log_text = "\n".join(log_list)
    return gr.update(value=log_text, visible=True)


css = """
      #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
      #video_inp {min-height: 100px}
      #chatbox {min-height: 100px;}
      #header {text-align: center;}
      #hint {font-size: 1.0em; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2em; }
      """

with gr.Blocks(css=css) as demo:
    
    state = gr.State(get_empty_state())


    with gr.Column(elem_id="col-container"):
        gr.Markdown("""## 🎞️ VLog Demo
                    Powered by BLIP2, GRIT, Whisper, ChatGPT and LangChain""",
                    elem_id="header")

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="video_input")
                gr.Markdown("Input youtube video_id in this textbox, *e.g.* *G7zJK6lcbyU*", elem_id="hint")
                with gr.Row():
                    video_id = gr.Textbox(value="", placeholder="Youtube video url", show_label=False)
                    vidsub_btn = gr.Button("Submit Youtube video")
                
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_message = gr.Textbox(show_label=False, placeholder="Enter text and press enter", visible=True).style(container=False)
                btn_submit = gr.Button("Submit")
                btn_clear_conversation = gr.Button("🔃 Start New Conversation")
            
            with gr.Column():
                vlog_btn = gr.Button("Generate Video Document")
                vlog_outp = gr.Textbox(label="Document output", lines=40)
                total_tokens_str = gr.Markdown(elem_id="total_tokens_str")

        examples = gr.Examples(
            examples=[
                ["examples/basketball_vlog.mp4"],
                ["examples/travel_in_roman.mp4"],
                ["examples/C8lMW0MODFs.mp4"],
                ["examples/huaqiang.mp4"],
                ["examples/C8lMW0MODFs.mp4"],
                ["examples/outcGtbnMuQ.mp4"],
            ],
            inputs=[video_inp],
        )

    gr.HTML('''<br><br><br><center>You can duplicate this Space to skip the queue:<a href="https://huggingface.co/spaces/anzorq/chatgpt-demo?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a><br></center>''')

    btn_submit.click(submit_message, [input_message, state], [input_message, chatbot])
    input_message.submit(submit_message, [input_message, state], [input_message, chatbot])
    btn_clear_conversation.click(clear_conversation, [], [input_message, video_inp, chatbot, vlog_outp, state])
    vlog_btn.click(vlog_fn, [video_inp], [vlog_outp])
    vidsub_btn.click(subvid_fn, [video_id], [video_inp])

    demo.load(queur=False)


demo.queue(concurrency_count=10)
demo.launch(height='800px', server_port=8899, debug=True, share=True)
