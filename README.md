
# VLog
VLog aims to seek a new perspective for video perception.

👇 Click the branch to see more instructions.

|      | [VLog (CVPR'25)](./VLog)   |[VLog-Agent](./VLog-agent/)|
|-----------|-----------|-----------|
|**TL;DR**| Video Narration as Vocabulary|Video as Long Document|
|| <img src="VLog/assets/vlog.jpg" width="500"> | <img src="VLog-agent/figures/vlog.jpg" width="500"> |
|**Method**|A novel, efficient video narrator (GPT2-based) with Narration Vocabulary via Generative Retrieval.|Given a video, we turn it into a textual document containing visual + audio info. By sending this doc to LLM, we can chat over the video!|


## 🎓 BibTeX
If you find our work helpful, please kindly consider citing our paper.

```
@misc{lin2025vlogvideolanguagemodelsgenerative,
      title={VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary}, 
      author={Kevin Qinghong Lin and Mike Zheng Shou},
      year={2025},
      eprint={2503.09402},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.09402}, 
}
```