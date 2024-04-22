import matplotlib
from src.components.From2Dto3D import From2Dto3D
from src.components.config import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an image and visualize it in 3D.')
    parser.add_argument('--image_path', type=str, help='Path to the input image file',default="./input/desk-plants.jpg")
    parser.add_argument('--save_path', type=str, help='To save the output in the given path (if not specified the file is not saved)')
    args = parser.parse_args()
    F2dto3D = From2Dto3D(args.image_path)
    F2dto3D()
    F2dto3D.visualize_3D_image(F2dto3D.pcd)
    F2dto3D.visualize_3D_image(F2dto3D.mesh,True)
    if args.save_path:
        F2dto3D.save(args.save_path)
