import os
import shutil
import tensorflow as tf


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_directory', './train',
                           'Training data directory')


def _organize_data(data_dir):
    """Re-organizes the data folder.

    Original:
        data_dir/cat.0.jpg
        data_dir/cat.1.jpg
        data_dir/dog.0.jpg
        data_dir/dog.1.jpg

    After:
        data_dir/cat/cat.0.jpg
        data_dir/cat/cat.1.jpg
        data_dir/dog/dog.0.jpg
        data_dir/dog/dog.1.jpg

    where 'cat' and 'dog' are the labels associated with these images.

    """
    label_list = ['cat', 'dog']
    for label in label_list:
        # Creates a new folder for each label
        os.mkdir(f'{data_dir}/{label}')
        
        # Returns a list of paths matching a pathname pattern
        image_file_path = f'{data_dir}/{label}*.jpg'
        matching_files = tf.gfile.Glob(image_file_path)
        
        # Creates the new folder structure with images
        for image in matching_files:
            shutil.copy2(image, f'{data_dir}/{label}')


def main(unused_argv):
    _organize_data(FLAGS.train_directory)


if __name__ == '__main__':
    tf.app.run()
