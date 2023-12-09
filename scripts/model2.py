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
    def __init__(self, hidden_dim, dropout_rate):
        super(TrajectoryEncoder, self).__init__()
        self.lstm = tf.keras.layers.LSTM(hidden_dim, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         kernel_initializer='glorot_normal', 
                                         recurrent_initializer ='orthogonal', 
                                         return_sequences = True,
                                         dropout = dropout_rate)

    def call(self, input_trajectories):
        # Implement the trajectory encoding logic using the LSTM layer
        input_reshaped = tf.reshape(input_trajectories, (-1, 8, 2)) # shape (batch_size (64 * total number of data), 8, 2)
        encoded_traj = self.lstm(input_reshaped) # shape (..., 8, hidden_dim)

        return encoded_traj

class SpatialInteractionModel(tf.keras.layers.Layer):
    def __init__(self, gcn_units, l2_reg = 0.01):
        super(SpatialInteractionModel, self).__init__()
        self.gcn_units = gcn_units
        self.l2_reg = l2_reg
        self.gcn = BatchGraphConvolution(gcn_units,
                                         activation='relu',
                                         use_bias=True,
                                         kernel_initializer='glorot_uniform',
                                         bias_initializer='zeros',
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_reg))

    def call(self, encoded_trajectories, lap):
        # Assuming encoded_trajectories has shape (batch_size, num_nodes, hidden_dim)
        num_nodes = encoded_trajectories.shape[1]
        lap = tf.expand_dims(lap, axis=0) # Add a batch dimension
        # Flatten encoded_trajectories for batch processing
        flat_encoded_trajectories = tf.reshape(encoded_trajectories, (-1, encoded_trajectories.shape[-1]))

        # Use BatchGraphConvolution layer
        enhanced_representations = self.gcn(flat_encoded_trajectories, lap)

        # Reshape back to the original shape
        enhanced_representations = tf.reshape(enhanced_representations, encoded_trajectories.shape)

        return enhanced_representations

    
class BatchGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, gcn_units, activation=None, use_bias=True, **kwargs):
        super(BatchGraphConvolution, self).__init__(**kwargs)
        self.gcn_units = gcn_units
        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.weight = self.add_weight("weight",
                                       shape=(input_dim, self.gcn_units),
                                       initializer=tf.keras.initializers.glorot_uniform(),
                                       trainable=True)
        if self.use_bias:
            self.bias = self.add_weight("bias",
                                       shape=(self.gcn_units, ),
                                       initializer=tf.keras.initializers.zeros(),
                                       trainable=True)
        else:
            self.bias = None
        super(BatchGraphConvolution, self).build(input_shape)

    def call(self, inputs, lap):
        support = tf.matmul(inputs, self.weight)
        output = tf.matmul(lap, support)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output


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
        # print(final_representations.shape)
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
        # print(predicted_trajectories.shape)
        return predicted_trajectories

# Build the complete model
class TrajectoryPredictionModel(tf.keras.Model):
    def __init__(self, hidden_dim, gcn_units, hidden_dim_2, output_dim, dropout_rate):
        super(TrajectoryPredictionModel, self).__init__()
        self.trajectory_encoder = TrajectoryEncoder(hidden_dim, dropout_rate)
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
test_lstm = TrajectoryEncoder(hidden_dim = 64, dropout_rate = 0.2)
encoder_result = test_lstm.call(x_train)

gcn_units = 32
test_gcn = SpatialInteractionModel(gcn_units = gcn_units)
gcn_result = test_gcn.call(encoder_result)
print("gcn shape:", gcn_result.shape)
#print(gcn_result[0])
