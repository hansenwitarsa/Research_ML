# folder_path = "/Users/hanse/Documents/Research/datasets/eth/"

# x_train, y_train, x_val, y_val, x_test, y_test_gt = DataIntoArray.process_folder(folder_path, obs_len = 8, pred_len = 12, max_ped = 64)
# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
# print("x_val shape:", x_val.shape)
# print("y_val shape:", y_val.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test_gt.shape)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
import time

if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. Make model
    '''
    hidden_dim_encoder = 512
    gcn_units = 256
    hidden_dim_temp = 128
    hidden_dim_decoder = 64
    output_dense = 24
    dropout_rate = 0.2
    l2_reg = 0.01
    model = TrajectoryPredictionModel(hidden_dim_encoder=hidden_dim_encoder,
                                  hidden_dim_temp=hidden_dim_temp,
                                  hidden_dim_decoder=hidden_dim_decoder,
                                  gcn_units=gcn_units,
                                  output_dense=output_dense,
                                  dropout_rate=dropout_rate,
                                  l2_reg=l2_reg)
    '''
    2. Model Learning
    '''
    criterion = losses.MeanSquaredError()

    def compute_loss(t, preds):
      t = tf.reshape(t, (-1, 12, 64 * 2))  # Reshape ground truth to (batch_size, 12, 64 * 2)
      preds = tf.reshape(preds, (-1, 12, 64 * 2))  # Reshape predictions to (batch_size, 12, 64 * 2)
      return criterion(t, preds)

    train_loss = metrics.Mean()
    val_loss = metrics.Mean()

    optimizer = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

    def train_step(x, t):
        with tf.GradientTape() as tape:
            preds = model(x)

            # Make mask here (based on the difference between gt and pred)
            x_coor = t[:, :, :, 0]
            y_coor = t[:, :, :, 1]
            non_zero_mask = np.logical_and(x_coor != 0, y_coor != 0)
            # Expand dimensions to match the shape of (1000, 12, 64, 1)
            mask = np.expand_dims(non_zero_mask, axis=-1)
            preds_mask = tf.multiply(preds, mask)

            # Make mask here (based on all 8 frame of each person in each frame)
            # Create a mask initialized with True
            mask2 = np.ones((x.shape[0], 1, 64, 1), dtype=bool)
            # Check if even one of the coordinates in all 8 frame is 0, then change it into False
            for i in range(x.shape[2]):
                for j in range(x.shape[1]):
                    for k in range(x.shape[0]):
                        if x[k][j][i][0] == 0 and x[k][j][i][1] == 0:
                            mask2[k][0][i] = False
            # Mask the prediction and the ground truth
            preds_mask2 = tf.multiply(preds_mask, mask2)
            gt_mask = tf.multiply(t, mask2)


            loss = compute_loss(gt_mask, preds_mask2)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

        return loss

    def val_step(x, t):
        preds = model(x)

        # Make mask here (based on the difference between gt and pred)
        x_coor = t[:, :, :, 0]
        y_coor = t[:, :, :, 1]
        non_zero_mask = np.logical_and(x_coor != 0, y_coor != 0)
        # Expand dimensions to match the shape of (1000, 12, 64, 1)
        mask = np.expand_dims(non_zero_mask, axis=-1)
        preds_mask = tf.multiply(preds, mask)

        # Make mask here (based on the 8th frame of each person in each frame)
        # Create a mask initialized with True
        mask2 = np.ones((x.shape[0], 1, 64, 1), dtype=bool)

        for i in range(x.shape[2]):
            for j in range(x.shape[1]):
                for k in range(x.shape[0]):
                    if x[k][j][i][0] == 0 and x[k][j][i][1] == 0:
                        mask2[k][0][i] = False

        preds_mask2 = tf.multiply(preds_mask, mask2)
        gt_mask = tf.multiply(t, mask2)


        loss = compute_loss(gt_mask, preds_mask2)
        val_loss(loss)

    epochs = 100
    batch_size = 32
    n_batches_train = x_train.shape[0] // batch_size + 1
    n_batches_val = x_val.shape[0] // batch_size + 1
    hist = {'loss': [], 'val_loss': []}
    # es = EarlyStopping(patience=10, verbose=1)

    #-------------------------------------------------------------------------------
    '''
    3. Model Training
    '''
    start_time = time.time()

    for epoch in range(epochs):
        x_, y_ = shuffle(x_train, y_train)

        for batch in range(n_batches_train):
            start = batch * batch_size
            end = start + batch_size
            train_step(x_[start:end], y_[start:end])

        for batch in range(n_batches_val):
            start = batch * batch_size
            end = start + batch_size
            val_step(x_val[start:end], y_val[start:end])

        hist['loss'].append(train_loss.result())
        hist['val_loss'].append(val_loss.result())

        print('epoch: {}, loss: {:.3}, val_loss: {:.3f}'.format(
            epoch+1,
            train_loss.result(),
            val_loss.result()
        ))
    #------------------------------------------------------------------------------
    '''
    4. Results and Evaluation
    '''
    # Predict the test data
    y_pred = model(x_test)

    # Mask the prediction
    # Make mask here (based on the difference between gt and pred)
    x_coor = y_test_gt[:, :, :, 0]
    y_coor = y_test_gt[:, :, :, 1]
    non_zero_mask = np.logical_and(x_coor != 0, y_coor != 0)
    # Expand dimensions to match the shape of (1000, 12, 64, 1)
    mask = np.expand_dims(non_zero_mask, axis=-1)
    preds_mask = tf.multiply(y_pred, mask)
    gt_mask = tf.multiply(y_test_gt, mask)

    # Make mask here (based on the 8th frame of each person in each frame)
    # Create a mask initialized with True
    mask2 = np.ones((x_test.shape[0], 1, 64, 1), dtype=bool)

    for i in range(x_test.shape[2]):
        for j in range(x_test.shape[1]):
            for k in range(x_test.shape[0]):
                if x_test[k][j][i][0] == 0 and x_test[k][j][i][1] == 0:
                    mask2[k][0][i] = False

    y_pred_mask = tf.multiply(preds_mask, mask2)
    y_gt_mask = tf.multiply(gt_mask, mask2)

    # Compute test loss
    test_loss = compute_loss(y_gt_mask, y_pred_mask)
    print("test loss:", test_loss)

    # Compute the time taken
    print("Process finished --- %s seconds ---" % (time.time() - start_time))

    
    # Calculate FDE and ADE
    y_pred_mask = tf.cast(y_pred_mask, dtype=tf.float32)
    y_gt_mask = tf.cast(y_gt_mask, dtype=tf.float32)

    def calculate_fde(prediction, ground_truth):
        # Extract final coordinates (x, y) for both prediction and ground truth
        final_coords_pred = prediction[:, -1, :]
        final_coords_gt = ground_truth[:, -1, :]

        # Calculate displacement for each pedestrian in each sample
        displacement = np.linalg.norm(final_coords_pred - final_coords_gt, axis=-1)

        # Calculate mean or sum of displacement values to get FDE
        fde = np.mean(displacement)  # Use np.sum() if you want the sum instead of the mean

        return fde

    print("FDE: ", calculate_fde(y_pred_mask, y_gt_mask))

    def calculate_ade(prediction, ground_truth):
        # Calculate displacement for each frame, pedestrian, and sample
        displacement = np.linalg.norm(prediction - ground_truth, axis=-1)

        # Average displacement across frames for each pedestrian and sample
        average_displacement = np.mean(displacement, axis=-1)

        # Calculate mean or sum of average displacement values to get ADE
        ade = np.mean(average_displacement)  # Use np.sum() if you want the sum instead of the mean

        return ade

    print("ADE: ", calculate_ade(y_pred_mask, y_gt_mask))
