import tensorflow as tf
import spektral
from loader import DataIntoArray
# from tensorflow.keras.layers import LSTM, Dense, GraphConvolutionalNetwork

# folder_path = "/Users/hanse/Documents/Research/datasets/eth/"

# x_train, y_train, x_val, y_val, x_test, y_test_gt = DataIntoArray.process_folder(folder_path, obs_len = 8, pred_len = 12, max_ped = 64)
# print("x_train shape:", x_train.shape)
# print("y_train shape:", y_train.shape)
# print("x_val shape:", x_val.shape)
# print("y_val shape:", y_val.shape)
# print("x_test shape:", x_test.shape)
# print("y_test shape:", y_test_gt.shape)

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
    def call(self, encoded_trajectories):
        # Assuming encoded_trajectories has shape (batch_size, num_nodes, hidden_dim)
        num_nodes = encoded_trajectories.shape[1]

        # Expand dimensions to add a "channel" dimension
        adjacency_matrix = tf.expand_dims(tf.eye(num_nodes), axis=0)
        
        # Use GCN layer
        enhanced_representations = self.gcn([encoded_trajectories, adjacency_matrix])
        enhanced_representations = self.batch_norm(enhanced_representations)
        #print(enhanced_representations.shape)
        return enhanced_representations

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
        self.encoder = tf.keras.layers.LSTM(hidden_dim_encoder, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         kernel_initializer='glorot_normal', 
                                         recurrent_initializer ='orthogonal', 
                                         )
        self.Dense = tf.keras.layers.Dense(output_dense,
                                         activation='linear')
        self.decoder = tf.keras.layers.LSTM(hidden_dim_decoder, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         kernel_initializer='glorot_normal', 
                                         recurrent_initializer ='orthogonal', 
                                         )
        self.temp = tf.keras.layers.LSTM(hidden_dim_temp, 
                                         activation='tanh', 
                                         recurrent_activation='sigmoid', 
                                         kernel_initializer='glorot_normal', 
                                         recurrent_initializer ='orthogonal', 
                                         )
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
        self.spatial_interaction_model = SpatialInteractionModel(gcn_units, l2_reg)

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
            # print(encode.shape)
            # encode shape (64, hidden_dim_encoder = 100)
            # encoder_outputs.append(encode)
            encoder_outputs = encoder_outputs.write(i, encode)
        encoder_outputs = encoder_outputs.stack()
        #print("encoder:", encoder_outputs.shape)
        # encoder_outputs shape = (..., 64, hidden_dim_encoder = 100)
        # input this encode to GCN layer
        gcn_output = self.spatial_interaction_model(encoder_outputs)
        #print("gcn:",gcn_output.shape)
        # gcn output shape = (..., 64, gcn_units = 50)
        # input this gcn_output into next LSTM layer with loop
        decoder_outputs = tf.TensorArray(tf.float32, size=gcn_output.shape[0])
        for i in range(gcn_output.shape[0]):
            temp_timestep = gcn_output[i]
            # temp timestep shape (64, gcn_units = 50)
            flatten_timestep = tf.keras.layers.Flatten()(temp_timestep)
            # Reshape to (batch_size=64, timesteps=1, features=50)
            timestep_reshaped = tf.reshape(flatten_timestep, (64, 1, self.gcn_units))
            # temp reshaped shape (64, 1, gcn_units = 50)
            temporal = self.temp(timestep_reshaped)
            # temporal hshape (64, hidden_dim_temp = 32)
            flatten_temporal = tf.keras.layers.Flatten()(temporal)
            temporal_reshaped = tf.reshape(flatten_temporal, (64, 1, self.hidden_dim_temp))
            # temporal reshaped shape (64, 1, hidden_dim_temp = 32)
            decode = self.decoder(temporal_reshaped)
            # decode shape (64, hidden_dim_decoder = 24)
            decode_reshaped = tf.reshape(decode, (64, 12, 2))
            # decode reshaped shape (64, 12, 2)
            decode_final = tf.transpose(decode_reshaped, perm=[1, 0, 2])
            # decode final shape (12, 64, 2)
            #decoder_outputs.append(decode_final)
            decoder_outputs = decoder_outputs.write(i, decode_final)
            # decoder outputs shape (..., 12, 64, 2)
        decoder_outputs = decoder_outputs.stack()
        #print("decoder:", decoder_outputs.shape)
        # Stack the outputs along the batch axis to form the final prediction sequence
        #predictions = tf.stack(decoder_outputs, axis=0)

        return decoder_outputs


# hidden_dim_encoder = 64
# hidden_dim_temp = 30
# hidden_dim_decoder = 24 
# gcn_units = 40
# output_dense = 24
# dropout_rate = 0.2 
# l2_reg = 0.01


# # Instantiate and compile the model

# model = TrajectoryPredictionModel(hidden_dim_encoder=hidden_dim_encoder,
#                                   hidden_dim_temp=hidden_dim_temp,
#                                   hidden_dim_decoder=hidden_dim_decoder,
#                                   gcn_units=gcn_units,
#                                   output_dense=output_dense,
#                                   dropout_rate=dropout_rate,
#                                   l2_reg=l2_reg)
# test = model.call(x_train)



# print(test.shape)
# # test = tf.reshape(test, (-1, 12, 64, 2))
# # print(test.shape)

