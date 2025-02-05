from PIL import Image
import glob
import time
import argparse
import natsort

# Argument parser setup
argParser = argparse.ArgumentParser(description="Create a GIF from PNG images.")
argParser.add_argument('--path', action='store', type=str, required=True, help="Input file pattern (folder containing PNGs)")

args = argParser.parse_args()

fp_in = args.path.rstrip("/")  # Ensure no trailing slash

# List to hold images
images = []

# Get all PNG images and sort naturally (1.png before 10.png)
image_files = natsort.natsorted(glob.glob(fp_in + "/*.png"))

# Load images
for filename in image_files:
    im = Image.open(filename)
    images.append(im)

# Ensure there are images before proceeding
if not images:
    raise ValueError("No PNG files found in the specified directory.")

# Add extra copies of the last frame to extend its duration
last_frame = images[-1]
images.extend([last_frame] * 9)

out_file = fp_in + "/output.gif"

# Save as GIF
images[0].save(out_file, save_all=True, append_images=images[1:], optimize=False, duration=100, loop=0)

print("GIF successfully saved as" + out_file)
