import os
import tensorflow as tf
from PIL import Image

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('img_dir', 'dataset3',
                           'Training data directory')
tf.app.flags.DEFINE_string('resized_img_dir', 'resized_images',
                           'Resized data directory')


def get_jpg_files(data_dir):
    """"Reads addresses and labels from `data_dir`.
    The `tf.gfile.Glob` module return a list of paths matching a pathname pattern
    
    Args:
        data_dir = Path to the `input image folder`.
    
    Returns:
        sorted_matching_files: list, address of each picture in the train folder.
    """
    cur_dir = os.getcwd()
    new_dir = os.chdir(data_dir)
    
    jpg_file = '*.JPG'
    matching_files = tf.gfile.Glob(jpg_file)
    sorted_matching_files = sorted(matching_files)
    
    # Go back to the original directory
    os.chdir(cur_dir)
    
    return sorted_matching_files


def convert_jpg_to_jpg(input_dir, filename, resized_img_dir):
    if not os.path.isdir(resized_img_dir):
        os.makedirs(resized_img_dir)
    
    image = Image.open(os.path.join(input_dir, filename))
    image = image.resize((1920,1080))
    resized_img_path = os.path.join(resized_img_dir, filename)
    image.save(resized_img_path, mode='JPEG', quality=100, optimize=True, progressive=True)
    
    return None
    
    
def main(unused_argv):
    filenames = get_jpg_files(FLAGS.img_dir)
    for filename in filenames:
        convert_jpg_to_jpg(FLAGS.img_dir, filename, FLAGS.resized_img_dir)


if __name__ == '__main__':
    tf.app.run()