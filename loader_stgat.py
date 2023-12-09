import tensorflow as tf
import os
# from tensorflow.keras.utils import Sequence

from trajectories import TrajectoryDataset, seq_collate  # Assuming you have a TensorFlow version of TrajectoryDataset

def data_loader(args, path):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim)

    loader = tf.data.Dataset.from_generator(
        generator=dset,
        output_signature=(
            tf.TensorSpec(shape=(None, 2, args.obs_len), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2, args.pred_len), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2, args.obs_len), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2, args.pred_len), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(args.obs_len, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 2), dtype=tf.int64)
        )
    )
    loader = loader.shuffle(buffer_size=len(dset)).batch(args.batch_size)

    return dset, loader

def get_dset_path(dset_name, dset_type):
    _dir = os.path.dirname(__file__)
    # _dir = _dir.split("/")[:-1]
    # _dir = "/".join(_dir)
    return os.path.join(_dir, "datasets", dset_name, dset_type)

train_path = get_dset_path("eth", "train")
val_path = get_dset_path("eth", "test")

train_dset, train_loader = data_loader(args, train_path)
_, val_loader = data_loader(args, val_path)