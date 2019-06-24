from __future__ import division, print_function, absolute_import

import keras
import numpy as np
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Flatten, MaxPool2D, Concatenate
from keras.models import Model

from link_bot_models.base_model import BaseModel


class RasterCNNModel(BaseModel):

    def __init__(self, args_dict, sdf_shape, N):
        super(RasterCNNModel, self).__init__(args_dict, N)
        self.sdf_shape = sdf_shape

        sdf = Input(shape=(sdf_shape[0], sdf_shape[1], 1), dtype='float32', name='sdf')
        rope_image = Input(shape=(sdf_shape[0], sdf_shape[1], 3), dtype='float32', name='rope_image')
        combined_image = Concatenate()([sdf, rope_image])

        self.conv_filters = [
            (32, (5, 5)),
            (32, (5, 5)),
            (16, (3, 3)),
            (16, (3, 3)),
        ]

        self.fc_layer_sizes = [
            256,
            256,
        ]

        conv_h = combined_image
        for conv_filter in self.conv_filters:
            n_filters = conv_filter[0]
            filter_size = conv_filter[1]
            conv_z = Conv2D(n_filters, filter_size, activation='relu')(conv_h)
            conv_h = MaxPool2D(2)(conv_z)

        conv_output = Flatten()(conv_h)

        fc_h = conv_output
        for fc_layer_size in self.fc_layer_sizes:
            fc_h = Dense(fc_layer_size, activation='relu')(fc_h)
        predictions = Dense(1, activation='sigmoid', name='combined_output')(fc_h)

        self.model_inputs = [sdf, rope_image]
        keras_model = Model(inputs=self.model_inputs, outputs=predictions)
        keras_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # self.keras_model = multi_gpu_model(keras_model, gpus=args_dict['n_gpus'])
        self.keras_model = keras_model

    def metadata(self, label_types):
        extra_metadata = {
            'conv_filters': self.conv_filters,
            'fc_layer_sizes': self.fc_layer_sizes,
            'sdf_shape': self.sdf_shape,
        }
        return super().metadata(label_types).update(extra_metadata)

    def violated(self, observations, sdf_data):
        m = observations.shape[0]
        rope_configuration = observations
        sdf = np.tile(np.expand_dims(sdf_data.sdf, axis=2), [m, 1, 1, 1])
        sdf_gradient = np.tile(sdf_data.gradient, [m, 1, 1, 1])
        sdf_origin = np.tile(sdf_data.origin, [m, 1])
        sdf_resolution = np.tile(sdf_data.resolution, [m, 1])
        sdf_extent = np.tile(sdf_data.extent, [m, 1])
        inputs_dict = {
            'rope_configuration': rope_configuration,
            'sdf': sdf,
            'sdf_gradient': sdf_gradient,
            'sdf_origin': sdf_origin,
            'sdf_resolution': sdf_resolution,
            'sdf_extent': sdf_extent
        }

        predicted_violated = (self.keras_model.predict(inputs_dict) > 0.5).astype(np.bool)
        return predicted_violated

    def __str__(self):
        return "keras constraint raster cnn"
