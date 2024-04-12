import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from src.components.config import image_size
class From2Dto3D:
    def __init__(self,image_path):
        self.image_path=image_path

    def load_image(self,image_path):
        image=Image.open(image_path)
        return image

    def preprocess_image(self,image):
        image = image.resize(image_size)
        return image

    def load_models(self):
        self.feature_extractor = GLPNImageProcessor.from_pretrained("vinvin02/glpn-nyu")
        self.model = GLPNForDepthEstimation.from_pretrained("vinvin02/glpn-nyu")

    def feature_extraction(self,image):
        self.features= self.feature_extractor(image=image,return_tensor="pt")

    def inference(self):
        with torch.no_grad():
            output=self.model(**self.features)
            self.predicted_depth=output.predicted_depth.squeeze().cpu().numpy()
        return self.predicted_depth
    

    def postprocess_image(self,image):
        pass

    