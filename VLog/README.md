# VLog: Video Narration as Vocabulary
> **VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary**<br>
> [Kevin Qinghong Lin](https://qinghonglin.github.io/), [Mike Zheng Shou](https://scholar.google.com/citations?user=h1-3lSoAAAAJ&hl=en)
> <br>Show Lab @ National University of Singapore<br>

**TL;DR:** A novel, efficient video narrator (GPT2-based) with Narration Vocabulary via Generative Retrieval.

- **Video Narration as Vocabulary**

<img src="assets/vlog.jpg" height="250" alt="Narration as Vocabulary">

- **Generative Retrieval**

<img src="assets/model.png" height="400" alt="Generative Retrieval">

## 🔨 Preparation
Please see [INSTALL.md](INSTALL.md)

## 🚀 Training
Please see [TRAIN.md](TRAIN.md).

Stay tune for more updates!

## ⭐ Run on your own video
Download VLog model and vocabulary [here](https://huggingface.co/KevinQHLin/VLog/tree/main).
```bash
mkdir pretrained
huggingface-cli download KevinQHLin/VLog --repo-type model --local-dir ./pretrained/
```

Then, refer to `demo.py` by providing your own video. Have fun!

## 😊 Acknowledgment
This codebase is built upon of [Fromage](https://github.com/kohjingyu/fromage).