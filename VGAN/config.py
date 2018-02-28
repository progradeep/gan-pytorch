import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataroot', required=True, help='path to dataset')

parser.add_argument('--image_dataset', help='specifies a separate dataset to train for images', default='')
parser.add_argument('--image_batch', type=int, default=10, help='number of images in image batch')
parser.add_argument('--video_batch', type=int, default=10, help='number of videos in video batch')

parser.add_argument('--image_size', type=int, default=64, help='resize all frames to this size')

parser.add_argument('--use_infogan', default=True, help='when specified infogan loss is used')
parser.add_argument('--use_categories', default=True ,help='when specified ground truth categories are used to train CategoricalVideoDiscriminator')
parser.add_argument('--use_reconstruct', default=True, help='when specified reconstruction is used')

parser.add_argument('--use_noise', help='when specified instance noise is used')

parser.add_argument('--noise_sigma', type=float, default=0, help='when use_noise is specified, noise_sigma controls the magnitude of the noise')

parser.add_argument('--image_discriminator', default='PatchImageDiscriminator', help='specifies image disciminator type (see vgan.py for a list of available models')
parser.add_argument('--video_discriminator', default='CategoricalVideoDiscriminator', help='specifies video discriminator type (see vgan.py for a list of available models')

parser.add_argument('--video_length', type=int, default=10, help='length of the video')
parser.add_argument('--log_interval', type=int, default=100, help='save valid gif and image')
parser.add_argument('--checkpoint_step', type=int, default=50, help='save checkpoint')
parser.add_argument('--n_channels', type=int, default=3, help='number of channels in the input data')

parser.add_argument('--every_nth', type=int, default=4, help='sample training videos using every nth frame')
parser.add_argument('--batches', type=int, default=100000, help='specify number of batches to train')

parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight_decay for adam. default=0.00001')

parser.add_argument('--dim_z_content', type=int, default=50, help='dimensionality of the content input, ie hidden space')
parser.add_argument('--dim_z_motion', type=int, default=10, help='dimensionality of the motion input')
parser.add_argument('--dim_z_category', type=int, default=5, help='dimensionality of categorical input')

parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--outf', default=None, help='folder to output images and videos ans model checkpoints')

def get_config():
    return parser.parse_args()
