import tensorflow as tf
import numpy as np
import os
import h5py
import sys
import argparse
from scipy import misc
from mayavi.mlab import quiver3d, draw
from mayavi import mlab

# Add paths for importing modules
sys.path.append('./utils')
sys.path.append('./models')

# Import model after adjusting the import path
import model_shape as model

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default='./train_shape', help='Directory where to write summaries and checkpoint.')
parser.add_argument('--base_dir', type=str, default='./data/ShapeNetCore_im2avatar', help='The path containing all the samples.')
parser.add_argument('--cat_id', type=str, default='02958343', help='The category id for each category: 02958343, 03001627, 03467517, 04379243')
parser.add_argument('--data_list_path', type=str, default='./data_list', help='The path containing data lists.')
parser.add_argument('--output_dir', type=str, default='./output_shape', help='Directory to save generated volume.')
parser.add_argument('--img', type=str, default='', help='Path to the input image.')
parser.add_argument('--mode', type=str, default='cube', help='Visualization mode.')
parser.add_argument('--thresh', type=float, default=0.6, help='Threshold for voxelization.')

FLAGS = parser.parse_args()

TRAIN_DIR = os.path.join(FLAGS.train_dir, FLAGS.cat_id)
OUTPUT_DIR = os.path.join(FLAGS.output_dir, 'test')
img_path = FLAGS.img

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

BATCH_SIZE = 1
IM_DIM = 128
VOL_DIM = 64

def inference():
    img_pl = tf.keras.Input(shape=(IM_DIM, IM_DIM, 3))
    is_train_pl = tf.keras.Input(shape=(), dtype=tf.bool)
    
    pred = model.get_model(img_pl, is_train_pl)
    pred = tf.sigmoid(pred)

    model_path = os.path.join(TRAIN_DIR, "trained_models")
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(tf.train.latest_checkpoint(model_path))

    img_1 = np.array(misc.imread(img_path) / 255.0)
    img_1 = img_1.reshape((1, IM_DIM, IM_DIM, 3))
    pred_res = pred(img_1, training=False)

    vol_ = pred_res[0]  # (VOL_DIM, VOL_DIM, VOL_DIM, 1)
    name_ = '001'  # FLAGS.img.strip().split('.')[0]

    save_path = OUTPUT_DIR
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path_name = os.path.join(save_path, name_+".h5")
    if os.path.exists(save_path_name):
        os.remove(save_path_name)

    with h5py.File(save_path_name, 'w') as h5_fout:
        h5_fout.create_dataset(
            'data', data=vol_,
            compression='gzip', compression_opts=4,
            dtype='float32')

    print(f'{name_}.h5 is predicted into {save_path_name}')

if __name__ == '__main__':
    inference()

# Visualization
path_shape = os.path.join(FLAGS.output_dir, 'test', '001.h5')
with h5py.File(path_shape, 'r') as f:
    voxel = f['data'][:].reshape(VOL_DIM, VOL_DIM, VOL_DIM)

voxel[voxel >= FLAGS.thresh] = 1
voxel[voxel < FLAGS.thresh] = 0

x, y, z = np.where(voxel == 1)
xx = np.ones(len(x))
yy = np.zeros(len(x))
zz = np.zeros(len(x))
scalars = np.arange(len(x))

pts = quiver3d(x, y, z, xx, yy, zz, scalars=scalars, mode=FLAGS.mode)
mlab.show()
