import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForConditionalGeneration

class ImageCaptioner:
    def __init__(self,  model_name="blip2-opt", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.processor, self.model = self.initialize_model()
        
    def initialize_model(self):
        if self.device == 'cpu':
            self.data_type = torch.float32
        else:
            self.data_type = torch.float16
        processor, model = None, None
        if self.model_name == "blip2-opt":
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b-coco", torch_dtype=self.data_type, low_cpu_mem_usage=True)
            
        elif self.model_name == "blip2-flan-t5":
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-flan-t5-xl", torch_dtype=self.data_type, low_cpu_mem_usage=True)
        
        # for gpu with small memory
        elif self.model_name == "blip":
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
        else:
            raise NotImplementedError(f"{self.model_name} not implemented.")
        model.to(self.device)
        
        if self.device != 'cpu':
            model.half()
        return processor, model

    def image_caption(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, self.data_type)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return generated_text
    
    def image_caption_debug(self, image_src):
        return "A dish with salmon, broccoli, and something yellow."
