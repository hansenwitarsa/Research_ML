import tensorflow as tf
import numpy as np
from loader import DataIntoArray
from model4 import TrajectoryPredictionModel
from sklearn.utils import shuffle


folder_path = "/Users/hanse/Documents/Research/datasets_process/eth/"

x_train, y_train, x_val, y_val, x_test, y_test_gt = DataIntoArray.process_folder(folder_path, obs_len = 8, pred_len = 12, max_ped = 64)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test_gt.shape)
# print("train shape:", train.shape)
# print("val shape:", val.shape)
# print("test shape:", test.shape)
# for i in range(20):
#     print(test[0][i][0][:])

hidden_dim_encoder = 50
gcn_units = 40
hidden_dim_temp = 30
hidden_dim_decoder = 24 
output_dense = 24
dropout_rate = 0.2 
l2_reg = 0.01
# model = TrajectoryPredictionModel(hidden_dim_encoder=hidden_dim_encoder,
#                                   hidden_dim_temp=hidden_dim_temp,
#                                   hidden_dim_decoder=hidden_dim_decoder,
#                                   gcn_units=gcn_units,
#                                   output_dense=output_dense,
#                                   dropout_rate=dropout_rate,
#                                   l2_reg=l2_reg)
# preds = model(x_train)
 
# HERE TO MAKE MASK
x_coor = y_test_gt[:, :, :, 0]
y_coor = y_test_gt[:, :, :, 1]

non_zero_mask = np.logical_and(x_coor != 0, y_coor != 0)

# Expand dimensions to match the shape of (1000, 12, 64, 1)
mask = np.expand_dims(non_zero_mask, axis=-1)
print(mask.shape)
print((tf.multiply(y_test_gt, mask)).shape)
# print(y_test_gt[100][5])
# print(mask[100][5])


# MAKE SECOND MASK
x_coor2 = x_test[:, 7, :, 0]
y_coor2 = x_test[:, 7, :, 1]

non_zero_mask2 = np.logical_and(x_coor2 != 0, y_coor2 != 0)
mask2 = np.expand_dims(non_zero_mask2, axis=(1, 3))
print(mask2.shape)
print((tf.multiply(y_test_gt, mask2)).shape)


#
# Create a mask initialized with False
mask3 = np.ones((x_test.shape[0], 1, 64, 1), dtype=bool)

for i in range(x_test.shape[2]):
    for j in range(x_test.shape[1]):
        for k in range(x_test.shape[0]):
            if x_test[k][j][i][0] == 0 and x_test[k][j][i][1] == 0:
                mask3[k][0][i] = False
print(mask3[6])
# print(x_train[100][:4])
# print(mask3[100][:5])
# Now, 'mask' has the desired shape and values based on the criteria you specified
pred_mask = tf.multiply(y_test_gt, mask3)
print(pred_mask.shape)
print(pred_mask[6][:5])
print(y_test_gt[6][:5])


# class MyLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.embedding = tf.keras.layers.Embedding(input_dim=5000, output_dim=16, mask_zero=True)
#         self.lstm = tf.keras.layers.LSTM(32)

#     def call(self, inputs):
#         x = self.embedding(inputs)
#         # Note that you could also prepare a `mask` tensor manually.
#         # It only needs to be a boolean tensor
#         # with the right shape, i.e. (batch_size, timesteps).
#         mask = self.embedding.compute_mask(inputs)
#         output = self.lstm(x, mask=mask)  # The layer will ignore the masked values
#         return output


# layer = MyLayer()
# x = np.random.random((32, 4)) * 100
# x = x.astype("int32")
# print(layer(x))