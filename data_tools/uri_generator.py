from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description='Create a file listing file URIs of a directory containing images.')
parser.add_argument('-d', '--dir', metavar='in_dir', type=str, nargs='+', default=['.'],
                    help='directory containing the images (global or relative to directory this script was invoked)')
args = parser.parse_args()

for img_dir in args.dir:
    img_dir = Path(img_dir)
    if not img_dir.is_absolute():
        img_dir = img_dir.absolute()
    uri_output = img_dir.joinpath('uri_path.csv')
    images = []
    extensions = ('*.tif', '*.png', '*.jpg', '*.jpeg')
    for ext in extensions:
        images.extend(img_dir.glob(ext))
    with open(uri_output,'w') as f:
        f.writelines([Path(image).as_uri()+'\n' for image in images])
