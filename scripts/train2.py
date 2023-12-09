# train with mask

import tensorflow as tf
import numpy as np
from loader import DataIntoArray
from model3 import TrajectoryPredictionModel
from sklearn.utils import shuffle

import time
start_time = time.time()

folder_path = "/Users/hanse/Documents/Research/datasets/eth/"

x_train, y_train, x_val, y_val, x_test, y_test_gt = DataIntoArray.process_folder(folder_path, obs_len = 8, pred_len = 12, max_ped = 64)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test_gt.shape)

if __name__ == '__main__':
    np.random.seed(123)
    tf.random.set_seed(123)

    '''
    1. Make model
    '''
    # hidden_dim = 32
    # gcn_units = 16
    # hidden_dim_2 = 12
    # output_dim = 2
    # dropout_rate = 0.4
    # model = TrajectoryPredictionModel(hidden_dim, gcn_units, hidden_dim_2, output_dim, dropout_rate)
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    hidden_dim_encoder = 50
    gcn_units = 40
    hidden_dim_temp = 30
    hidden_dim_decoder = 24 
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
    criterion = tf.keras.losses.MeanSquaredError()

    def compute_loss(t, preds):
        # t and preds are the ground truth and predicted trajectories
        # mask is a binary mask indicating which trajectories are valid (1) and which are not (0)
    
        t = tf.reshape(t, (-1, 12, 64 * 2))  # Reshape ground truth to (batch_size, 12, 64 * 2)
        preds = tf.reshape(preds, (-1, 12, 64 * 2))  # Reshape predictions to (batch_size, 12, 64 * 2)
    
        # Compute the mean squared error loss
        loss = criterion(t, preds)
    
        return loss


    train_loss = tf.keras.metrics.Mean()
    val_loss = tf.keras.metrics.Mean()

    # Parameters for learning rate decay
    initial_learning_rate = 0.001
    decay_steps = 10000
    decay_rate = 0.9

    # Create a learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=True)

    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)


    def train_step(x, t):
        # x is the observed trajectory
        # t is the ground truth
        with tf.GradientTape() as tape:
            preds = model(x)

            # Make mask here (based on the difference between gt and pred)
            x_coor = t[:, :, :, 0]
            y_coor = t[:, :, :, 1]
            non_zero_mask = np.logical_and(x_coor != 0, y_coor != 0)
            # Expand dimensions to match the shape of (1000, 12, 64, 1)
            mask = np.expand_dims(non_zero_mask, axis=-1)
            preds_mask = tf.multiply(preds, mask)

            # Make mask here (based on the 8th frame of each person in each frame)
            # Ground truth must also be masked
            x_coor2 = x[:, 7, :, 0]
            y_coor2 = x[:, 7, :, 1]

            non_zero_mask2 = np.logical_and(x_coor2 != 0, y_coor2 != 0)
            mask2 = np.expand_dims(non_zero_mask2, axis=(1, 3))
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
        # Ground truth must also be masked
        x_coor2 = x[:, 7, :, 0]
        y_coor2 = x[:, 7, :, 1]

        non_zero_mask2 = np.logical_and(x_coor2 != 0, y_coor2 != 0)
        mask2 = np.expand_dims(non_zero_mask2, axis=(1, 3))
        preds_mask2 = tf.multiply(preds_mask, mask2)

        gt_mask = tf.multiply(t, mask2)
        
        loss = compute_loss(gt_mask, preds_mask2)
        val_loss(loss)

    # callbacks = [
    #     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    #     tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
    # ]

    epochs = 100
    batch_size = 32
    n_batches_train = x_train.shape[0] // batch_size + 1
    n_batches_val = x_val.shape[0] // batch_size + 1
    hist = {'loss': [], 'val_loss': []}
    es = tf.keras.callbacks.EarlyStopping(patience=10, verbose=1)

    #-------------------------------------------------------------------------------

    # for epoch in range(epochs):
    #     x_, y_ = shuffle(x_train, y_train)

    #     for batch in range(n_batches_train):
    #         start = batch * batch_size
    #         end = start + batch_size
    #         train_step(x_train[start:end], y_train[start:end])

    #     for batch in range(n_batches_val):
    #         start = batch * batch_size
    #         end = start + batch_size
    #         val_step(x_val[start:end], y_val[start:end])

    #     hist['loss'].append(train_loss.result())
    #     hist['val_loss'].append(val_loss.result())

    #     print('epoch: {}, loss: {:.3}, val_loss: {:.3f}'.format(
    #         epoch+1,
    #         train_loss.result(),
    #         val_loss.result()
    #     ))

        # if es(val_loss.result()):
        #     break

        # train_loss.reset_states()
        # val_loss.reset_states()



    for epoch in range(epochs):
        x_, y_ = shuffle(x_train, y_train)

        # Train on the entire dataset without batches
        train_step(x_train, y_train)

        # Validate on the entire validation set without batches
        val_step(x_val, y_val)

        hist['loss'].append(train_loss.result())
        hist['val_loss'].append(val_loss.result())

        print('epoch: {}, loss: {:.3}, val_loss: {:.3f}'.format(
            epoch + 1,
            train_loss.result(),
            val_loss.result()
        ))

        # Reset metrics for the next epoch
        train_loss.reset_states()
        val_loss.reset_states()

        # if es(val_loss.result()):
        #     break

    # y = model(x_test)
    # print(y_test_gt[0][0:4])
    # print(y[0][0:4])
    # print(y.shape)

print("Process finished --- %s seconds ---" % (time.time() - start_time))