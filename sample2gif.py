import argparse
import imageio
import numpy as np
from PIL import Image

#########################################################
# USAGE
# python sample2gif.py --img_path fakeGif_AB_005_300.png
#########################################################

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', required=True, help='path to image')
parser.add_argument('--img_size', type=int, default=64, help='image size')
parser.add_argument('--n_frames', type=int, default=10, help='number of frames')
parser.add_argument('--n_row', type=int, default=10, help='number of rows in the concatenated sample')

def get_config():
    return parser.parse_args()


def jpg2gif(img_path, img_size, n_frames, n_row):
    im = Image.open(img_path)
    im = im.convert('RGB')
    for row in range(n_row):
        frames = []
        for fr in range(n_frames):
            frame = im.crop((2*(fr+1) + fr*img_size,
                             2*(row+1) + row*img_size,
                             2*(fr+1) + (fr+1)*img_size,
                             2*(row+1) + (row+1)*img_size))
            frames.append(np.array(frame))
        imageio.mimsave('%s_%d.gif' % (img_path.split('.png')[0], row), frames)


if __name__ == "__main__":
    config = get_config()
    jpg2gif(config.img_path, config.img_size, config.n_frames, config.n_row)