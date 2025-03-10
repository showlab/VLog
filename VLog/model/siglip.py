import torch
from torch import nn
from transformers import AutoProcessor, AutoModel, AutoTokenizer

class Siglip(nn.Module):
    def __init__(self, model_id, device_id):
        super().__init__()
        self.device_id = device_id
        self.model = AutoModel.from_pretrained(model_id).cuda(device_id)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def get_txt_embeds(self, text_list):
        with torch.no_grad():
            inputs = self.tokenizer(text_list, padding="max_length", return_tensors="pt").to(f'cuda:{self.device_id}')
            text_features = self.model.get_text_features(**inputs)
        return text_features