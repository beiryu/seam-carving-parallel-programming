# Example
# python convert2pnm.py inputDirectory inputFileExtension outputFileExtension
# python convert2pnm.py input .jpg .pnm

import glob
import os
import json
import sys
import numpy as np
import cv2
from PIL import Image

untreatable_formats = [".exr", ".icns", ".msp", "xbm", ".palm", ".xvthumbnails"]

argvs = sys.argv
base_dir = os.path.dirname(__file__)

print("command line arguments:", argvs)
print("where we locate:", base_dir)

# load config file
with open(os.path.join(base_dir, "config.json")) as f:
    _dict = json.load(f)

input_output_file_format_openCV = _dict["inputOutputFileFormatOpenCV"]
input_output_file_format_pillow = _dict["inputOutputFileFormatPillow"]
output_file_format_pillow = _dict["outputFileFormatPillow"]
print("input output file format OpenCV:", input_output_file_format_openCV)
print("input output file format Pillow:", input_output_file_format_pillow)

# DB path
img_paths = glob.glob(os.path.join(base_dir, argvs[1] + "/*"))
print("img paths:", img_paths)
img_paths_for_openCV = []
img_paths_for_pillow = []
for i in range(len(img_paths)):
    if (
        img_paths[i][-len(argvs[2]) :].lower() in input_output_file_format_openCV
        and img_paths[i][-len(argvs[2]) :].lower() == argvs[2]
    ):
        img_paths_for_openCV.append(img_paths[i])
    elif (
        img_paths[i][-len(argvs[2]) :].lower() in input_output_file_format_pillow
        and img_paths[i][-len(argvs[2]) :].lower() == argvs[2]
    ):
        img_paths_for_pillow.append(img_paths[i])

print("img paths for openCV:", img_paths_for_openCV)
print("img paths for pillow:", img_paths_for_pillow)

print("image list was made.")

if argvs[2] == argvs[3]:
    print("no need to convert.")
    sys.exit()

if not (
    argvs[2] in input_output_file_format_openCV
    or argvs[2] in input_output_file_format_pillow
    or argvs[2] == ".pdf"
):
    print("sorry, we can't convert from the format.")
    sys.exit()

if not (
    argvs[3] in input_output_file_format_openCV
    or argvs[3] in input_output_file_format_pillow
    or argvs[3] in output_file_format_pillow
    or argvs[3] == ".pdf"
):
    print("sorry, we can't convert to the format.")
    sys.exit()

if argvs[2] in untreatable_formats or argvs[3] in untreatable_formats:
    print("sorry, we can't deal with that format at this time.")
    sys.exit()

if not os.path.exists(os.path.join(base_dir, "output")):
    os.makedirs(os.path.join(base_dir, "output"))
os.chdir(os.path.join(base_dir, "output"))

if (
    argvs[2] in input_output_file_format_openCV
    and argvs[3] in input_output_file_format_openCV
):
    for i in range(len(img_paths_for_openCV)):
        image = cv2.imread(img_paths_for_openCV[i])
        cv2.imwrite(
            "{0}{1}".format(
                img_paths_for_openCV[i].split("/")[-1][: -len(argvs[2])], argvs[3]
            ),
            image,
        )
        print("{0}th picture was converted.".format(i + 1))

elif argvs[2] in input_output_file_format_openCV:
    for i in range(len(img_paths_for_openCV)):
        image = cv2.imread(img_paths_for_openCV[i])[:, :, ::-1]
        image = Image.fromarray(image)
        image.save(
            "{0}{1}".format(
                img_paths_for_openCV[i].split("/")[-1][: -len(argvs[2])], argvs[3]
            )
        )
        print("{0}th picture was converted.".format(i + 1))


if argvs[2] in input_output_file_format_pillow and (
    argvs[3] in input_output_file_format_pillow or argvs[3] in output_file_format_pillow
):
    for i in range(len(img_paths_for_pillow)):
        image = Image.open(img_paths_for_pillow[i])
        image.save(
            "{0}{1}".format(
                img_paths_for_pillow[i].split("/")[-1][: -len(argvs[2])], argvs[3]
            )
        )
        print("{0}th picture was converted.".format(i + 1))

elif argvs[2] in input_output_file_format_pillow:
    for i in range(len(img_paths_for_pillow)):
        image = Image.open(img_paths_for_pillow[i])
        image = np.asarray(image)
        cv2.imwrite(
            "{0}{1}".format(
                img_paths_for_pillow[i].split("/")[-1][: -len(argvs[2])], argvs[3]
            ),
            image,
        )
        print("{0}th picture was converted.".format(i + 1))
