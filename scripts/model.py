import tensorflow as tf
import spektral
from loader import DataIntoArray
# from tensorflow.keras.layers import LSTM, Dense, GraphConvolutionalNetwork

folder_path = "/Users/hanse/Documents/Research/datasets/eth/"

x_train, y_train, x_val, y_val, x_test, y_test_gt = DataIntoArray.process_folder(folder_path, obs_len = 8, pred_len = 12, max_ped = 64)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_val shape:", x_val.shape)
print("y_val shape:", y_val.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test_gt.shape)

class TrajectoryEncoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim):
        super(TrajectoryEncoder, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_dim, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         kernel_initializer='glorot_normal', 
                                         recurrent_initializer ='orthogonal', 
                                         return_sequences = True,
                                         )

    def call(self, input_trajectories):
        # Implement the trajectory encoding logic using the LSTM layer
        input_reshaped = tf.reshape(input_trajectories, (-1, 8, 2)) # shape (batch_size (64 * total number of data), 8, 2)
        #print(input_reshaped[1:10])
        #x_reshaped = tf.transpose(x_time_step, perm=[1, 0, 2]) # shape (64, 8, 2)
        encoded_traj = self.lstm(input_reshaped) # shape (..., 8, hidden_dim)
        print(encoded_traj.shape)
        #print(encoded_traj[0])
        #print(encoded_traj[1])
        #outputs.append(encoded_traj)
        #predictions = tf.stack(outputs, axis = 0)
        #encoded_trajectories = self.lstm(input_trajectories)
        return encoded_traj

class SpatialInteractionModel(tf.keras.layers.Layer):
    def __init__(self, gcn_units, l2_reg = 0.01):
        super(SpatialInteractionModel, self).__init__()
        self.gcn = spektral.layers.GCNConv(gcn_units, 
                                           activation=None, 
                                           use_bias=True, 
                                           kernel_initializer='glorot_uniform', 
                                           bias_initializer='zeros', 
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg), 
                                           bias_regularizer=None, 
                                           activity_regularizer=None, 
                                           kernel_constraint=None, 
                                           bias_constraint=None)
        self.batch_norm = tf.keras.layers.BatchNormalization()
    def call(self, encoded_trajectories):
        # Assuming encoded_trajectories has shape (batch_size, num_nodes, hidden_dim)
        num_nodes = encoded_trajectories.shape[1]

        # Expand dimensions to add a "channel" dimension
        adjacency_matrix = tf.expand_dims(tf.eye(num_nodes), axis=0)
        
        # Use GCN layer
        enhanced_representations = self.gcn([encoded_trajectories, adjacency_matrix])
        enhanced_representations = self.batch_norm(enhanced_representations)
        print(enhanced_representations.shape)
        return enhanced_representations


class TemporalInteractionModel(tf.keras.layers.Layer):
    def __init__(self, hidden_dim_2, dropout_rate):
        super(TemporalInteractionModel, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_dim_2, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         kernel_initializer='glorot_normal', 
                                         recurrent_initializer ='orthogonal', 
                                         return_sequences = True,
                                         dropout = dropout_rate)

    def call(self, enhanced_representations):
        # Implement the temporal interaction modeling logic using the LSTM layer
        final_representations = self.lstm(enhanced_representations)
        print(final_representations.shape)
        return final_representations

class TrajectoryDecoder(tf.keras.layers.Layer):
    def __init__(self, output_dim, dropout_rate):
        super(TrajectoryDecoder, self).__init__()
        self.lstm = tf.keras.layers.LSTM(output_dim, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         kernel_initializer='glorot_normal', 
                                         recurrent_initializer ='orthogonal', 
                                         return_sequences = True,
                                         dropout = dropout_rate)

    def call(self, final_representations):
        # Implement the trajectory decoding logic using the LSTM layer
        final_representations = tf.transpose(final_representations, perm=[0, 2, 1])
        predicted_trajectories = self.lstm(final_representations)
        print(predicted_trajectories.shape)
        return predicted_trajectories

# Build the complete model
class TrajectoryPredictionModel(tf.keras.Model):
    def __init__(self, hidden_dim, gcn_units, hidden_dim_2, output_dim, dropout_rate):
        super(TrajectoryPredictionModel, self).__init__()
        self.trajectory_encoder = TrajectoryEncoder(hidden_dim)
        self.spatial_interaction_model = SpatialInteractionModel(gcn_units)
        self.temporal_interaction_model = TemporalInteractionModel(hidden_dim_2, dropout_rate)
        self.trajectory_decoder = TrajectoryDecoder(output_dim, dropout_rate)

    def call(self, input_trajectories):
        # Forward pass through the entire model
        encoded_trajectories = self.trajectory_encoder(input_trajectories)
        enhanced_representations = self.spatial_interaction_model(encoded_trajectories)
        final_representations = self.temporal_interaction_model(enhanced_representations)
        predicted_trajectories = self.trajectory_decoder(final_representations)
        return predicted_trajectories



# Instantiate and compile the model
# model = TrajectoryPredictionModel(hidden_dim=..., gcn_units=..., output_dim=...)
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# test_lstm = TrajectoryEncoder(hidden_dim = 64, dropout_rate = 0.2)
# encoder_result = test_lstm.call(x_train)

# gcn_units = 32
# test_gcn = SpatialInteractionModel(gcn_units = gcn_units)
# gcn_result = test_gcn.call(encoder_result)
# print("gcn shape:", gcn_result.shape)
#print(gcn_result[0])

# test_temp = TemporalInteractionModel(hidden_dim=32)
# temp_result = test_temp.call(gcn_result)
# print("temp shape:", temp_result.shape)
# #print(temp_result[0])


# Instantiate and compile the model
model = TrajectoryPredictionModel(hidden_dim=64, gcn_units=32, hidden_dim_2=12, output_dim=2, dropout_rate=0.2)
test = model.call(x_train)
print(test.shape)
test = tf.reshape(test, (-1, 12, 64, 2))
print(test.shape)

# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# # Print the model summary
# model.summary()

# # Train the model (replace with your actual training data)
# model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))