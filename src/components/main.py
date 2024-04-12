import matplotlib
from src.components.From2Dto3D import From2Dto3D
from src.components.config import *



if __name__ == '__main__':

    F2dto3D= From2Dto3D(image_path)
    F2dto3D()
    F2dto3D.visualize_3D_image()