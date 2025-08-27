import histomicstk as htk
import cv2
import os
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser("color_normalize_pipe.py")
parser.add_argument("--ref", help="path to the reference image, Exp: dataset/color_reference.png", type=str, required=True)
parser.add_argument("--dest", help="Folder path of the output images, Exp:dataset/Color_Normalized/", type=str, required=True)
parser.add_argument("--src", help="Folder path of the source images (will only process png files), Exp: dataset/Raw_images/", type=str, required=True)

args = parser.parse_args()


def reinhard_img(directory, des_path):
    
    """
    Apply Reinhard color normalization to images in a directory using a reference image.
    Args:
        directory (str): Path to the directory containing images to be normalized.
        des_path (str): Path to the directory where normalized images will be saved
    """
    img = cv2.imread(args.ref)
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(img)
    des_path = des_path
    try:
        os.mkdir(des_path)
    except:
        pass
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            print(filename)
            img_input = cv2.imread(os.path.join(directory, filename))
            im_nmzd = htk.preprocessing.color_normalization.reinhard(img_input, mean_ref, std_ref)
            status = cv2.imwrite(os.path.join(des_path, filename[:-3] + 'png'), im_nmzd)
            print("Image written to file-system : ", status)
        else:
            continue

reinhard_img(args.src, args.dest)