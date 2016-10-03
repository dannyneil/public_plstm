import os
import lasagne
import cPickle as pickle
from lasagne.utils import floatX
from lasagne.random import get_rng
from lasagne.init import Initializer
from lasagne import nonlinearities
from lasagne import init
from lasagne.layers.base import Layer
import numpy as np

def replace_updates_nans_with_zero(updates):
    import theano.tensor as T
    import numpy as np
    # Replace all nans with zeros
    for k,v in updates.items():
        k = T.switch(T.eq(v, np.nan), float(0.), v)
    return updates

def save_model(filename, suffix, model, log=None, announce=True):
    # Build filename
    filename = '{}_{}'.format(filename, suffix)
    # Acquire Data
    data = lasagne.layers.get_all_param_values(model)
    # Store in separate directory
    filename = os.path.join('./models/', filename)
    # Inform user
    if announce:
        print('Saving to: {}'.format(filename))
    # Generate parameter filename and dump
    param_filename = '%s.params' % (filename)
    with open(param_filename, 'w') as f:
        pickle.dump(data, f)
    # Generate log filename and dump
    if log is not None:
        log_filename = '%s.log' % (filename)
        with open(log_filename, 'w') as f:
            pickle.dump(log, f)

def load_model(filename, model):
    # Build filename
    filename = os.path.join('./models/', '%s.params' % (filename))
    with open(filename, 'r') as f:
        data = pickle.load(f)
    lasagne.layers.set_all_param_values(model, data)
    return model

def load_log(filename, append_dir=True):
    if append_dir:
        filename = os.path.join('./models/', '%s.log' % (filename))
    with open(filename, 'r') as f:
        log = pickle.load(f)
    return log

def store_in_log(log, kv_pairs):
    # Quick helper function to append values to keys in a log
    for k,v in kv_pairs.items():
        log[k].append(v)
    return log

def non_flattening_dense(l_in, batch_size, seq_len, *args, **kwargs):
    # Flatten down the dimensions for everything but the features
    l_flat = lasagne.layers.ReshapeLayer(l_in, (-1, [2]))
    # Make a dense layer connected to it
    l_dense = lasagne.layers.DenseLayer(l_flat, *args, **kwargs)
    # Reshape it back out - this could be done implicitly, but I want to throw an error if not matching
    l_reshaped = lasagne.layers.ReshapeLayer(l_dense, (batch_size, seq_len, l_dense.output_shape[1]))
    return l_reshaped

class ExponentialUniformInit(Initializer):
    """
    """
    def __init__(self, range):
        self.range = range

    def sample(self, shape):
        return floatX(np.exp(get_rng().uniform(low=self.range[0],
            high=self.range[1], size=shape)))


class MultiResEmbeddingLayer(Layer):
    """
    """
    def __init__(self, incoming, input_size, output_size_per,
                 W=init.Normal(mean=0,std=0.2), res_levels=[1], **kwargs):
        super(MultiResEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size_per = output_size_per
        self.res_levels = res_levels

        self.W = []
        for idx, level in enumerate(res_levels):
            self.W.append( self.add_param(W, (input_size/level, output_size_per), name="W_{}".format(idx)) )

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size_per*len(self.res_levels), )

    def get_output_for(self, input, **kwargs):
        list_of_embeds = []
        for idx, level in enumerate(self.res_levels):
            list_of_embeds.append(self.W[idx][input//level])
        return T.concatenate(list_of_embeds, axis=-1)