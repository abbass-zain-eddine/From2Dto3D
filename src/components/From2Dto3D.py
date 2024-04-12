import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from src.components.config import image_size,depth_estimation_model
import numpy as np 
import open3d as o3d 

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
        self.feature_extractor = GLPNImageProcessor.from_pretrained(depth_estimation_model)
        self.model = GLPNForDepthEstimation.from_pretrained(depth_estimation_model)

    def feature_extraction(self,image):
        self.features= self.feature_extractor(images=image,return_tensors="pt")

    def inference(self):
        with torch.no_grad():
            self.output_depth=self.model(**self.features)
            self.predicted_depth=self.output_depth.predicted_depth.squeeze().cpu().numpy()*1000
        return self.predicted_depth
    

    def postprocess_image(self,image):
        pass

    def prepare_3D_image(self,image):
        depth_image= (self.predicted_depth*255 / np.max(self.predicted_depth)).astype('uint8')
        image=np.array(image)
        depth_o3d = o3d.geometry.Image(depth_image)
        image_o3d = o3d.geometry.Image(image)
        self.rgbd_image= o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d,depth_o3d,convert_rgb_to_intensity=False)
        return self.rgbd_image
    
    def Generate_3D_image_by_camera(self):
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics( image_size[0],image_size[1], 500,500, image_size[0]/2,image_size[1]/2)
        self.pcd=o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd_image,camera_intrinsic)
        return self.pcd
    
    def visualize_3D_image(self):
        o3d.visualization.draw_plotly([self.pcd])

    def __call__(self, *args: Image.Any, **kwds: Image.Any) -> Image.Any:
        image = self.load_image(self.image_path)
        image = self.preprocess_image(image)
        self.load_models()
        self.feature_extraction(image)
        self.inference()
        self.postprocess_image(image)
        self.rgbd_image=self.prepare_3D_image(image)
        self.pcd=self.Generate_3D_image_by_camera()
        return self.rgbd_image,self.pcd

