import tensorflow as tf
import spektral
from loader import DataIntoArray
# from tensorflow.keras.layers import LSTM, Dense, GraphConvolutionalNetwork

# folder_path = "/Users/hanse/Documents/Research/datasets/eth/"

# x_train, y_train, x_val, y_val, x_test, y_test_gt, train, val, test = DataIntoArray.process_folder(folder_path, obs_len = 8, pred_len = 12, max_ped = 64)
# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
# print("x_val shape:", x_val.shape)
# print("y_val shape:", y_val.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test_gt.shape)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, hidden_dim_encoder, dropout_rate):
        super(Encoder, self).__init__()
        self.hidden_dim_encoder = hidden_dim_encoder
        self.dropout_rate = dropout_rate
        self.encoder = tf.keras.layers.LSTM(hidden_dim_encoder,
                                               activation='tanh',
                                               recurrent_activation='sigmoid',
                                               kernel_initializer='glorot_normal',
                                               recurrent_initializer='orthogonal'
                                               )
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.batch_norm = tf.keras.layers.BatchNormalization()

    def call(self, input):
        # input shape (..., 8, 64, 2)
        # encoder_outputs = []
        encoder_outputs = tf.TensorArray(tf.float32, size=input.shape[0])
        for i in range(input.shape[0]):
            input_timestep = input[i]
            # input_timestep shape (8, 64, 2)
            input_timestep_reshaped = tf.transpose(input_timestep, perm=[1, 0, 2])
            # intput timestep reshaped shape (64, 8, 2)
            encode = self.encoder(input_timestep_reshaped)
            # encode shape (64, hidden_dim_encoder)
            # encoder_outputs.append(encode)
            encoder_outputs = encoder_outputs.write(i, encode)
        encoder_outputs = encoder_outputs.stack()
        # (..., 64, hidden_dim_encoder)
        return encoder_outputs
    

class SpatialInteractionModel(tf.keras.layers.Layer):
    def __init__(self, gcn_units, l2_reg):
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
        self.spatial = tf.keras.layers.LSTM(gcn_units,
                                               activation='tanh',
                                               recurrent_activation='sigmoid',
                                               kernel_initializer='glorot_normal',
                                               recurrent_initializer='orthogonal'
                                               )
    def call(self, encoded_trajectories):
        # (..., 64, hidden_dim_encoder)
        num_nodes = encoded_trajectories.shape[1]
        
        # Expand dimensions to add a "channel" dimension
        adjacency_matrix = tf.expand_dims(tf.eye(num_nodes), axis=0)
        
        # Use GCN layer
        enhanced_representations = self.gcn([encoded_trajectories, adjacency_matrix])
        # enhanced_representations = self.batch_norm(enhanced_representations)
        
        return enhanced_representations


class TemporalInteractionModel(tf.keras.layers.Layer):
    def __init__(self, hidden_dim_temp, hidden_dim_decoder, dropout_rate, output_dense):
        super(TemporalInteractionModel, self).__init__()
        self.hidden_dim_temp = hidden_dim_temp
        self.hidden_dim_decoder = hidden_dim_decoder
        self.dropout_rate = dropout_rate
        self.output_dense = output_dense
        self.temp = tf.keras.layers.LSTM(hidden_dim_temp, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         kernel_initializer='glorot_normal', 
                                         recurrent_initializer ='orthogonal', 
                                         )
        self.decoder = tf.keras.layers.LSTM(hidden_dim_decoder,
                                               activation='tanh',
                                               recurrent_activation='sigmoid',
                                               kernel_initializer='glorot_normal',
                                               recurrent_initializer='orthogonal')
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.dense1 = tf.keras.layers.Dense(output_dense,
                                         activation='linear')
        self.dense2 = tf.keras.layers.Dense(output_dense,
                                         activation='linear')
        self.dense3 = tf.keras.layers.Dense(output_dense,
                                         activation='linear')

    def call(self, gcn_output):
        # gcn output shape = (..., 64, gcn_units)
        # input this gcn_output into next LSTM layer with loop
        decoder_outputs = tf.TensorArray(tf.float32, size=gcn_output.shape[0])
        for i in range(gcn_output.shape[0]):
            temp_timestep = gcn_output[i]
            # temp timestep shape (64, gcn_units)
            
            dense_1 = self.dense1(temp_timestep) # (64, 24)
            dense_1_reshaped = tf.reshape(dense_1, (64, 12, 2))

            temporal = self.temp(dense_1_reshaped) # (64, hidden_dim_temp)
            dense_2 = self.dense2(temporal) # (64, 24)
            dense_2_reshaped = tf.reshape(dense_2, (64, 12, 2))
            
            decode = self.decoder(dense_2_reshaped) # (64, hidden_dim_decoder)
            dense_3 = self.dense3(decode) # (64, 24)
            dense_3_reshaped = tf.reshape(dense_3, (64, 12, 2))

            final = tf.transpose(dense_3_reshaped, perm=[1, 0, 2]) # reshape into (12, 64, 2)
            decoder_outputs = decoder_outputs.write(i, final)
            # decoder outputs shape (..., 12, 64, 2)
        decoder_outputs = decoder_outputs.stack()

        return decoder_outputs

# Build the complete model
class TrajectoryPredictionModel(tf.keras.Model):
    def __init__(self, 
                 hidden_dim_encoder, 
                 hidden_dim_temp, 
                 hidden_dim_decoder, 
                 gcn_units, 
                 output_dense, 
                 dropout_rate, 
                 l2_reg):
        super(TrajectoryPredictionModel, self).__init__()
        self.hidden_dim_encoder = hidden_dim_encoder
        self.hidden_dim_temp = hidden_dim_temp
        self.hidden_dim_decoder = hidden_dim_decoder
        self.gcn_units = gcn_units
        self.output_dense = output_dense
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.Dense = tf.keras.layers.Dense(output_dense,
                                         activation='linear')
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
        self.encoder_model = Encoder(hidden_dim_encoder, dropout_rate)
        self.spatial_interaction_model = SpatialInteractionModel(gcn_units, l2_reg)
        self.temporal_interaction_model = TemporalInteractionModel(hidden_dim_temp, hidden_dim_decoder, 
                                                                   dropout_rate, output_dense)

    def call(self, input_traj):
        # input shape (..., 8, 64, 2)
        encode_traj = self.encoder_model(input_traj)
        # (..., 64, hidden_dim_encoder)
        spatial_traj = self.spatial_interaction_model(encode_traj)
        # (..., 64, gcn_units)
        temporal_traj = self.temporal_interaction_model(spatial_traj)
        # (..., 64, hidden_dim_temp)
        # (..., 12, 64, 2)
        return temporal_traj
