import matplotlib
from src.components.From2Dto3D import From2Dto3D
from src.components.config import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process an image and visualize it in 3D.')
    parser.add_argument('image_path', type=str, help='Path to the input image file')
    args = parser.parse_args()
    F2dto3D = From2Dto3D(args.image_path)
    F2dto3D()
    F2dto3D.visualize_3D_image()
