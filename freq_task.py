import time
import sys
import lasagne
import theano
import os
import argparse
from collections import defaultdict
from lasagne_utils import save_model, store_in_log, load_model, load_log, replace_updates_nans_with_zero, ExponentialUniformInit, non_flattening_dense
import theano.tensor as T
import numpy as np
from plstm import PLSTMLayer, PLSTMTimeGate
from bnlstm import LSTMWBNLayer

# Remove these:
from lasagne.layers.recurrent import Gate
from lasagne import nonlinearities
from lasagne import init

def non_flattening_dense(l_in, batch_size, seq_len, *args, **kwargs):
    # Flatten down the dimensions for everything but the features
    l_flat = lasagne.layers.ReshapeLayer(l_in, (-1, [2]))
    # Make a dense layer connected to it
    l_dense = lasagne.layers.DenseLayer(l_flat, *args, **kwargs)
    # Reshape it back out
    l_reshaped = lasagne.layers.ReshapeLayer(l_dense, (batch_size, seq_len, l_dense.output_shape[1]))
    return l_reshaped


def get_layer_output_fn(fn_inputs, network, on_unused_input='raise'):
    import theano
    outs = []
    for layer in lasagne.layers.get_all_layers(network):
        outs.append(lasagne.layers.get_output(layer, deterministic=True))
    out_fn = theano.function(fn_inputs, outs, on_unused_input=on_unused_input)
    return out_fn


def get_train_and_val_fn(inputs, target_var, network, replace_nans=True, use_time=False):
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                  dtype=theano.config.floatX)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    # updates = lasagne.updates.nesterov_momentum(
    #         loss, params, learning_rate=0.01, momentum=0.9)
    # updates = lasagne.updates.adam(loss, params, learning_rate=3e-5)
    updates = lasagne.updates.adam(loss, params, learning_rate=1e-3)
    #updates = PartialAdam(loss, params, update_every=update_every, learning_rate=1e-4)

    #rmsprop(loss, params, learning_rate=1e-4)
    #u pdates = lasagne.updates.rmsprop(loss, params)
    if replace_nans:
        updates = replace_updates_nans_with_zero(updates)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    if not use_time:
        on_unused_input='warn'
    else:
        on_unused_input='raise'

    out_fn = get_layer_output_fn(inputs, network, on_unused_input=on_unused_input)
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    fn_inputs = inputs + [target_var]
    train_fn = theano.function(fn_inputs, [loss, train_acc], updates=updates, on_unused_input=on_unused_input)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(fn_inputs, [test_loss, test_acc], on_unused_input=on_unused_input)

    return train_fn, val_fn, out_fn

def get_rnn(input_var, mask_var, time_var, arch_size, GRAD_CLIP=100, bn=False, use_time=False, model_type='plstm'):
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(None, None, 1), input_var=input_var) #L0?
    # Mask as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_mask = lasagne.layers.InputLayer(shape=(None, None), input_var=mask_var) #l6
    # Time as matrices of dimensionality (N_BATCH, MAX_LENGTH)
    l_t = lasagne.layers.InputLayer(shape=(None, None), input_var=time_var) #l5

    # Allows arbitrary sizes
    batch_size, seq_len, _ = input_var.shape

    if model_type=='plstm':
        print('Using PLSTM.')
        # RNN layer 1
        l_forward = PLSTMLayer(
            l_in, time_input=l_t,
            num_units=arch_size[1],
            mask_input=l_mask,
            ingate=Gate(b=lasagne.init.Constant(-0.1)),
            forgetgate=Gate(b=lasagne.init.Constant(0), nonlinearity=nonlinearities.sigmoid),
            cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
            outgate=Gate(),
            nonlinearity=nonlinearities.tanh,
            grad_clipping=GRAD_CLIP,
            bn=bn,
            timegate=PLSTMTimeGate(
                Period=ExponentialUniformInit((1,3)),
                Shift=init.Uniform( (0., 100)),
                On_End=init.Constant(0.05))
            )

    else:
        print('Using LSTM, with BN: {}'.format(bn))
        # RNN layers
        l_forward = LSTMWBNLayer(lasagne.layers.ConcatLayer([l_in, lasagne.layers.ReshapeLayer(l_t,[batch_size, seq_len, 1])], axis=2),
                    num_units=arch_size[1],
                    mask_input=l_mask, grad_clipping=GRAD_CLIP,
                    ingate=Gate(b=lasagne.init.Constant(-0.1)),
                    forgetgate=Gate(b=lasagne.init.Constant(0), nonlinearity=nonlinearities.sigmoid),
                    cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                    outgate=Gate(),
                    nonlinearity=nonlinearities.tanh,
                    bn=bn)

    # Need to slice off the last layer now
    l_slice = lasagne.layers.SliceLayer(l_forward, -1, axis=1) #l11

    # Softmax
    l_dense = lasagne.layers.DenseLayer(l_slice, num_units=arch_size[2],nonlinearity=lasagne.nonlinearities.leaky_rectify)
    l_out = lasagne.layers.NonlinearityLayer(l_dense, nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

# Special Data Iterator
# ----------------------------------------------------
class SinWaveIterator(object):
    """
    """
    def flow(self, sample_regularly, sample_res, min_period=1, max_period=100, min_spec_period=5, max_spec_period=6,
                batch_size=32, nb_examples=10000, min_duration=10, max_duration=200,
                # NOTE CHANGED FOR REVIEW -- MAX_NUM_POINTS ORIGINALLY 500
                min_num_points=10, max_num_points=200):
        # Get some constants
        num_examples = nb_examples
        nb_batch = int(np.ceil(float(num_examples)/batch_size))

        b = 0
        min_log_period, max_log_period = np.log(min_period), np.log(max_period)
        while b < nb_batch:
            num_points = np.random.uniform(low=min_num_points,high=max_num_points,size=(batch_size))
            duration = np.random.uniform(low=min_duration, high=max_duration, size=batch_size)
            start = np.random.uniform(low=0, high=max_duration-duration, size=batch_size)
            periods = np.exp(np.random.uniform(low=min_log_period, high=max_log_period, size=(batch_size)))

            # Ensure always at least half

            periods[:len(periods)/2] = np.random.uniform(low=min_spec_period, high=max_spec_period, size=len(periods)/2)
            shifts = np.random.uniform(low=0,high=duration,size=(batch_size))

            all_t = []
            all_masks = []
            all_wavs = []
            for idx in range(batch_size):
                if sample_regularly:
                    t = np.arange(start[idx],start[idx]+duration[idx],step=sample_res)
                else:
                    t = np.sort(np.random.random(int(num_points[idx])))*duration[idx]+start[idx]
                wavs = np.sin(2*np.pi/periods[idx]*t-shifts[idx])
                mask = np.ones(wavs.shape)
                all_t.append(t)
                all_masks.append(mask)
                all_wavs.append(wavs)

            lengths = [len(item) for item in all_masks]
            max_length = np.max(lengths)
            bXt = np.zeros((batch_size, max_length))
            bXmask = np.zeros((batch_size, max_length))
            bX = np.zeros((batch_size, max_length, 1))
            for idx in range(batch_size):
                bX[idx, max_length-lengths[idx]:, 0] = all_wavs[idx]
                bXmask[idx, max_length-lengths[idx]:] = all_masks[idx]
                bXt[idx, max_length-lengths[idx]:] = all_t[idx]


            bY = np.zeros(batch_size)
            bY[(periods>=min_spec_period)*(periods<=max_spec_period)] = 1
            #print('\t\tStart: {}, Duration: {}'.format(start, duration))

            yield bX.astype('float32'), bXmask.astype('bool'), bXt.astype('float32'), bY.astype('int32')
            b += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a timeful RNN using PLSTM.')
    # File and path naming stuff
    parser.add_argument('--run_id',   default=os.environ.get('LSB_JOBID',''), help='ID of the run, used in saving.')
    parser.add_argument('--filename', default='freq_task_30.09', help='Filename to save model and log to.')
    parser.add_argument('--resume',   default=None, help='Filename to load model and log from.')
    # Control meta parameters
    parser.add_argument('--seed',       default=42,   type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--grad_clip',  default=10.,  type=float, help='Clip the gradient to prevent it from blowing up.')
    parser.add_argument('--batch_size', default=32,   type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--num_epochs', default=100,  type=int, help='Number of epochs to train for.')
    parser.add_argument('--patience',   default=100,  type=int, help='How long to wait for an increase in validation error before quitting.')
    parser.add_argument('--save_every', default=1000, type=int, help='How many epochs to wait between a save.')
    # Control architecture and run data
    parser.add_argument('--model_type',     default='plstm', help='Choose which model type to use.')
    parser.add_argument('--use_time',       default=1, type=int, help='Whether to use time or just ignore and use standard LSTM.')
    parser.add_argument('--batch_norm',     default=0, type=int, help='Batch normalize.')
    parser.add_argument('--sample_regularly', default=0, type=int, help='Whether to sameple regularly or irregularly.')
    parser.add_argument('--sample_res',     default=0.5, type=float, help='Resolution at which to sample.')
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    # Constants
    num_train = 5000
    num_test = 500
    arch_size = [None, 110, 2]
    # theano.config.mode='FAST_COMPILE'

    #  Set filename
    comb_filename    = '{}_{}_bn_{}_reg_samp_{}_samp_res_{}_{}'.format(args.filename, args.model_type,
    	args.batch_norm, args.sample_regularly, args.sample_res, args.seed)
    if args.run_id!='':
    	comb_filename += '_{}'.format(args.run_id)

    # Create symbolic vars
    input_var       = T.ftensor3('my_input_var')
    mask_var        = T.bmatrix('my_mask')
    target_var      = T.ivector('my_targets')
    time_var        = T.fmatrix('my_timevar')

    # Build model
    print("Building network ...")
    #   Get input dimensions
    network = get_rnn(input_var, mask_var, time_var, arch_size, args.grad_clip,
    	bn=args.batch_norm, model_type=args.model_type)
    # Instantiate log
    log = defaultdict(list)
    print("Built.")

    # Resume if desired
    if args.resume:
        print('RESUMING: {}'.format(args.resume))
        load_model(args.resume, network)
        log = load_log(args.resume)

    # Compile the learning functions
    print('Compiling functions...')
    train_fn, val_fn, out_fn = get_train_and_val_fn([input_var, mask_var, time_var], target_var, network)

    # Instantiate data generator
    d = SinWaveIterator()
    # Save result
    save_model(comb_filename, 'pretrain', network, log)

    # Precalc for announcing
    num_train_batches = int(np.ceil(float(num_train)/args.batch_size))
    num_test_batches = int(np.ceil(float(num_test)/args.batch_size))

    # Finally, launch the training loop.
    print("Starting training...")
    for epoch in range(args.num_epochs):
        print("Starting {} of {}.".format(epoch + 1, args.num_epochs))

        # Clear out epoch variables each iteration
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        failed = 0

        # Call the data generator
        for data in d.flow(batch_size=args.batch_size, nb_examples=num_train,
                           sample_regularly=args.sample_regularly, sample_res=args.sample_res):
            bX, maskX, bXt, bY = data
            # Do a training batch
            err, acc = train_fn(bX, maskX, bXt, bY)
            train_err += err # Accumulate error
            train_acc += acc
            train_batches += 1 # Accumulate count so we can calculate mean later
            # Log and print
            log = store_in_log(log, {'b_train_err': err, 'b_train_acc' : acc})
            print("\tBatch {} of {} (FF: {:.2f}%): Loss: {} | Accuracy: {}".format(
                train_batches, num_train_batches, np.mean(maskX)*100., err, acc*100.))
            # Force it to go to output now rather than holding
            sys.stdout.flush()
        print("Training loss:\t\t{:.6f}".format(train_err / train_batches))

        # Test the accuracy
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for data in d.flow(batch_size=args.batch_size, nb_examples=num_test,
                           sample_regularly=args.sample_regularly, sample_res=args.sample_res):
            # Pass a batch through the test function
            bX, maskX, bXt, bY = data
            err,acc = val_fn(bX, maskX, bXt, bY)
            val_err += err # Accumulate error
            val_acc += acc
            val_batches += 1  # Accumulate count so we can calculate mean later
            # Log and print
            log = store_in_log(log, {'b_val_err': err, 'b_val_acc' : acc})
            print("\t\tVAL batch {} of {} (FF: {:.2f}%): Loss: {} | Acc: {}".format(
                val_batches, num_test_batches, np.mean(maskX)*100., err, acc*100.))
            sys.stdout.flush()

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, args.num_epochs, time.time() - start_time))
        # And we store
        log = store_in_log(log, {'val_err': val_err / val_batches,
                             'train_err': train_err / train_batches,
                             'val_acc':val_acc / val_batches*100.,
                             'train_acc':train_acc / train_batches*100.} )

        print("\t Training loss:\t\t{:.6f}".format(log['train_err'][-1]))
        print("\t Validation loss:\t\t{:.6f}".format(log['val_err'][-1]))
        print("\t Training accuracy:\t\t{:.2f}".format(log['train_acc'][-1]))
        print("\t Validation accuracy:\t\t{:.2f}".format(log['val_acc'][-1]))
        print("\t Run failures: {}".format(failed))

        # Save result
        if (epoch+1 % args.save_every) == 0:
            print('Saving....')
            save_model(comb_filename, 'recent', network, log)

        # End if there's no improvement in validation error
        best_in_last_set = np.min(log['val_err'][-(args.patience-1):])
        # Drop out if our best round was not in the last set, i.e., no improvement
        if len(log['val_err']) > args.patience and log['val_err'][-args.patience] <= best_in_last_set:
            break

    # Save result
    save_model(comb_filename, 'final', network, log)
    print('Completed.')
