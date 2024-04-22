import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from src.components.config import image_size,depth_estimation_model,remove_statistical_outlier_nb_neighbors,remove_statistical_outlier_std_ratio,create_from_point_cloud_poisson_depth
import numpy as np 
import open3d as o3d 
import os

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
    

    def postprocess_image(self):
        #outlier removal
        cl,ind = self.pcd.remove_statistical_outlier(nb_neighbors=remove_statistical_outlier_nb_neighbors,std_ratio=remove_statistical_outlier_std_ratio)
        self.processed_pcd=self.pcd.select_by_index(ind)

        #estimate normals
        self.processed_pcd.estimate_normals()
        self.processed_pcd.orient_normals_to_align_with_direction()



    def generate_mesh(self):
        self.mesh= o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.processed_pcd,depth=create_from_point_cloud_poisson_depth,n_threads=1)[0]
        rotation=self.mesh.get_rotation_matrix_from_xyz((np.pi,0,0))
        self.mesh.rotate(rotation,center=(0,0,0))


    def prepare_3D_image(self,image):
        depth_image= (self.predicted_depth*255 / np.max(self.predicted_depth)).astype('uint8')
        image=np.array(image)
        depth_o3d = o3d.geometry.Image(depth_image)
        image_o3d = o3d.geometry.Image(image)
        self.rgbd_image= o3d.geometry.RGBDImage.create_from_color_and_depth(image_o3d,depth_o3d,convert_rgb_to_intensity=False)
        return self.rgbd_image
    
    def Generate_3D_image_by_camera(self):
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics( image_size[0],image_size[1],500,500, image_size[0]/2,image_size[1]/2)
        self.pcd=o3d.geometry.PointCloud.create_from_rgbd_image(self.rgbd_image,camera_intrinsic)
        return self.pcd
    
    def visualize_3D_image(self,pcd,is_mesh=False):
        o3d.visualization.draw_geometries([pcd],mesh_show_back_face=is_mesh)

    
    def save(self,path):
        #o3d.io.write_point_cloud(os.path.join(path,str(np.random.randint(0,100000000))+"pcd.obj"),self.pcd)
        o3d.io.write_triangle_mesh(os.path.join(path,str(np.random.randint(0,100000000))+"mesh.obj"),self.mesh)

    def __call__(self, *args: Image.Any, **kwds: Image.Any) -> Image.Any:
        image = self.load_image(self.image_path)
        image = self.preprocess_image(image)
        self.load_models()
        self.feature_extraction(image)
        self.inference()
        self.rgbd_image=self.prepare_3D_image(image)
        self.pcd=self.Generate_3D_image_by_camera()
        self.postprocess_image()
        self.generate_mesh()
        return self.rgbd_image,self.pcd,self.mesh

