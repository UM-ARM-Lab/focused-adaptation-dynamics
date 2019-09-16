import tensorflow as tf
import tensorflow.keras.layers as layers


def action_smear_layer(action, h, w):
    _, input_sequence_length, action_dim = action.shape

    reshape = layers.Reshape(target_shape=[input_sequence_length, 1, 1, action_dim], name='smear_reshape')
    smear = layers.Lambda(function=lambda action_reshaped: tf.tile(action_reshaped, [1, 1, h, w, 1]),
                          name='spatial_tile')

    def forward(action):
        action_reshaped = reshape(action)
        action_smear = smear(action_reshaped)
        return action_smear

    return forward
