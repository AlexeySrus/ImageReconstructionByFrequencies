from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import os
from shutil import copyfile


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='Path to SIDD data folder'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='Path to result folder'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    result_clear_folder = os.path.join(args.output, 'clear/')
    result_noisy_folder = os.path.join(args.output, 'noisy/')

    os.makedirs(result_clear_folder, exist_ok=True)
    os.makedirs(result_noisy_folder, exist_ok=True)

    for subfolder in tqdm(os.listdir(args.input)):
        clear_image_path = os.path.join(args.input, subfolder, 'GT_SRGB_010.PNG')
        noisy_image_path = os.path.join(args.input, subfolder, 'NOISY_SRGB_010.PNG')

        res_clear_path = os.path.join(result_clear_folder, '{}.png'.format(subfolder))
        res_noisy_path = os.path.join(result_noisy_folder, '{}.png'.format(subfolder))

        copyfile(clear_image_path, res_clear_path)
        copyfile(noisy_image_path, res_noisy_path)
