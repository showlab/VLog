

# 🎞 VLog: Video as a Long Document
Given a long video, we turn it into a doc containing visual + audio info. By sending this doc to ChatGPT, we can chat over the video! 

![vlog](figures/vlog.jpg)

### **News**

- 20/April/2023: We release our project on github and gradio demo!

### To Do List

**Done**

- [x] LLM Reasoner: ChatGPT (multilingual) + LangChain
- [x] Vision Captioner: BLIP2 + GRIT
- [x] ASR Translator: Whisper (multilingual)
- [x] Video Segmenter: KTS

**Doing** 

- [ ] Huggingface Space
- [ ] Improve Vision Models: MiniGPT-4, LLaVA, Family of Segment-anything
- [ ] Replace ChatGPT with own trained LLM
- [ ] Improve ASR Translator

## 🧸 Examples

<details open><summary>[ News - GPT4 launch event ]</summary><img src="./figures/case5.png" alt="GPT4 launch event" style="width: 100%; height: auto;">
</details>
<details open><summary>[ TV series - 征服之华强买瓜  ]</summary><img src="./figures/case2.png" alt="华强买瓜" style="width: 100%; height: auto;">
</details>

<details><summary>[ TV series - The Big Bang Theory ]</summary><img src="./figures/case4.png" alt="The Big Bang Theory" style="width: 100%; height: auto;">
</details>

<details><summary>[ Travel video - Travel in Rome ]</summary><img src="./figures/case1.png" alt="Travel in Rome" style="width: 100%; height: auto;">
</details>

<details><summary>[ Vlog - Basketball training ]</summary><img src="./figures/case3.png" alt="Basketball training" style="width: 100%; height: auto;">
</details>

## 🔨 Preparation

Please find installation instructions in [install.md](install.md).

## 🌟 Start here

### Run in cmd

```
python main.py --video_path examples/buy_watermelon.mp4 --openai_api_key xxxxx
```

The generated video document will be generated and saved in `examples/buy_watermelon.log`

### Run in Gradio

```
python gradio.py --openai_api_key xxxxx
```

## 🙋 Suggestion

The project is stay tuned 🔥

If you have more suggestions or functions need to be implemented in this codebase, feel free to drop us an email `kevin.qh.lin@gmail`, `leiwx52@gmail.com` or open an issue.

## 😊 Acknowledgment

This work is based on [ChatGPT](http://chat.openai.com), [BLIP2](https://huggingface.co/spaces/Salesforce/BLIP2), [GRIT](https://github.com/JialianW/GRiT), [KTS](https://inria.hal.science/hal-01022967/PDF/video_summarization.pdf), [Whisper](https://github.com/openai/whisper), [LangChain](https://python.langchain.com/en/latest/), [Image2Paragraph](https://github.com/showlab/Image2Paragraph).
