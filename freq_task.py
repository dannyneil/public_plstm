import time
import sys
import lasagne
import theano
import os
import argparse
from collections import defaultdict
from lasagne_utils import save_model, store_in_log, load_model, load_log, \
                          ExponentialUniformInit, non_flattening_dense, get_layer_output_fn
import theano.tensor as T
import numpy as np
from plstm import PLSTMLayer, PLSTMTimeGate
from bnlstm import LSTMWBNLayer
from lasagne.layers.recurrent import Gate

def get_train_and_val_fn(inputs, target_var, network):
    # Get network output
    prediction = lasagne.layers.get_output(network)
    # Calculate training accuracy
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                  dtype=theano.config.floatX)
    # Calculate crossentropy between predictions and targets
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Fetch trainable parameters
    params = lasagne.layers.get_all_params(network, trainable=True)
    # Calculate updates for the parameters given the loss
    updates = lasagne.updates.adam(loss, params, learning_rate=1e-3)

    # Fetch network output, using deterministic methods
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    # Again calculate crossentropy, this time using (test-time) determinstic pass
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    # Also, create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Get the raw output activations, for every layer
    out_fn = get_layer_output_fn(inputs, network)

    # Add in the targets to the function inputs
    fn_inputs = inputs + [target_var]
    # Compile a train function with the updates, returning loss and accuracy
    train_fn = theano.function(fn_inputs, [loss, train_acc], updates=updates)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function(fn_inputs, [test_loss, test_acc])

    return train_fn, val_fn, out_fn

def get_rnn(input_var, mask_var, time_var, arch_size, GRAD_CLIP=100, bn=False, model_type='plstm'):
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
            forgetgate=Gate(b=lasagne.init.Constant(0), nonlinearity=lasagne.nonlinearities.sigmoid),
            cell=Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
            outgate=Gate(),
            nonlinearity=lasagne.nonlinearities.tanh,
            grad_clipping=GRAD_CLIP,
            bn=bn,
            learn_time_params=[True, True, True],
            timegate=PLSTMTimeGate(
                Period=ExponentialUniformInit((1,3)),
                Shift=lasagne.init.Uniform( (0., 100)),
                On_End=lasagne.init.Constant(0.05))
            )

    else:
        print('Using LSTM, with BN: {}'.format(bn))
        # RNN layers
        l_forward = LSTMWBNLayer(lasagne.layers.ConcatLayer([l_in, lasagne.layers.ReshapeLayer(l_t,[batch_size, seq_len, 1])], axis=2),
                    num_units=arch_size[1],
                    mask_input=l_mask, grad_clipping=GRAD_CLIP,
                    ingate=Gate(b=lasagne.init.Constant(-0.1)),
                    forgetgate=Gate(b=lasagne.init.Constant(0), nonlinearity=lasagne.nonlinearities.sigmoid),
                    cell=Gate(W_cell=None, nonlinearity=lasagne.nonlinearities.tanh),
                    outgate=Gate(),
                    nonlinearity=lasagne.nonlinearities.tanh,
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
                batch_size=32, num_examples=10000, min_duration=15, max_duration=125,
                min_num_points=15, max_num_points=125):
        # Calculate constants
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        min_log_period, max_log_period = np.log(min_period), np.log(max_period)
        b = 0
        while b < num_batches:
            # Choose curve and sampling parameters
            num_points = np.random.uniform(low=min_num_points,high=max_num_points,size=(batch_size))
            duration = np.random.uniform(low=min_duration, high=max_duration, size=batch_size)
            start = np.random.uniform(low=0, high=max_duration-duration, size=batch_size)
            periods = np.exp(np.random.uniform(low=min_log_period, high=max_log_period, size=(batch_size)))
            shifts = np.random.uniform(low=0,high=duration,size=(batch_size))

            # Ensure always at least half is special class
            periods[:len(periods)/2] = np.random.uniform(low=min_spec_period, high=max_spec_period, size=len(periods)/2)

            # Define arrays of data to fill in
            all_t = []
            all_masks = []
            all_wavs = []
            for idx in range(batch_size):
                if sample_regularly:
                    # Synchronous condition
                    t = np.arange(start[idx],start[idx]+duration[idx],step=sample_res)
                else:
                    # Asynchronous condition
                    t = np.sort(np.random.random(int(num_points[idx])))*duration[idx]+start[idx]
                wavs = np.sin(2*np.pi/periods[idx]*t-shifts[idx])
                mask = np.ones(wavs.shape)
                all_t.append(t)
                all_masks.append(mask)
                all_wavs.append(wavs)

            # Now pack all the data down into masked matrices
            lengths = [len(item) for item in all_masks]
            max_length = np.max(lengths)
            bXt = np.zeros((batch_size, max_length))
            bXmask = np.zeros((batch_size, max_length))
            bX = np.zeros((batch_size, max_length, 1))
            for idx in range(batch_size):
                bX[idx, max_length-lengths[idx]:, 0] = all_wavs[idx]
                bXmask[idx, max_length-lengths[idx]:] = all_masks[idx]
                bXt[idx, max_length-lengths[idx]:] = all_t[idx]

            # Define and calculate labels
            bY = np.zeros(batch_size)
            bY[(periods>=min_spec_period)*(periods<=max_spec_period)] = 1

            # Yield data
            yield bX.astype('float32'), bXmask.astype('bool'), bXt.astype('float32'), bY.astype('int32')
            b += 1

# Special Data Iterator
# ----------------------------------------------------
class SinWaveComboIterator(object):
    """
    """
    def flow(self, sample_regularly, sample_res, min_period=1, max_period=100,
                min_spec_period=5, max_spec_period=6, min_spec_period_2=13, max_spec_period_2=15,
                batch_size=32, num_examples=10000, min_duration=1, max_duration=100,
                min_num_points=100, max_num_points=1000):
        # Calculate constants
        num_batches = int(np.ceil(float(num_examples)/batch_size))
        min_log_period, max_log_period = np.log(min_period), np.log(max_period)
        b = 0
        while b < num_batches:
            # Choose curve and sampling parameters
            num_points = np.random.uniform(low=min_num_points,high=max_num_points,size=(batch_size))
            duration = np.random.uniform(low=min_duration, high=max_duration, size=batch_size)
            start = np.random.uniform(low=0, high=max_duration-duration, size=batch_size)
            periods = np.exp(np.random.uniform(low=min_log_period, high=max_log_period, size=(batch_size)))
            periods2 = np.exp(np.random.uniform(low=min_log_period, high=max_log_period, size=(batch_size)))
            shifts = np.random.uniform(low=0,high=duration,size=(batch_size))
            shifts2 = np.random.uniform(low=0,high=duration,size=(batch_size))

            # Ensure always at least half is special class
            periods[:len(periods)/2] = np.random.uniform(low=min_spec_period, high=max_spec_period, size=len(periods)/2)
            periods2[:len(periods)/2] = np.random.uniform(low=min_spec_period_2, high=max_spec_period_2, size=len(periods)/2)

            # Define arrays of data to fill in
            all_t = []
            all_masks = []
            all_wavs = []
            for idx in range(batch_size):
                # Asynchronous condition
                t = np.sort(np.random.random(int(num_points[idx])))*duration[idx]+start[idx]
                wavs = np.sin(1./periods[idx]*t-shifts[idx]) + np.sin(1./periods2[idx]*t-shifts2[idx])
                mask = np.ones(wavs.shape)
                all_t.append(t)
                all_masks.append(mask)
                all_wavs.append(wavs)

            # Now pack all the data down into masked matrices
            lengths = [len(item) for item in all_masks]
            max_length = np.max(lengths)
            bXt = np.zeros((batch_size, max_length))
            bXmask = np.zeros((batch_size, max_length))
            bX = np.zeros((batch_size, max_length, 1))
            for idx in range(batch_size):
                bX[idx, max_length-lengths[idx]:, 0] = all_wavs[idx]
                bXmask[idx, max_length-lengths[idx]:] = all_masks[idx]
                bXt[idx, max_length-lengths[idx]:] = all_t[idx]

            # Define and calculate labels
            bY = np.zeros(batch_size)
            bY[(periods>=min_spec_period)*(periods<=max_spec_period)*(periods2>=min_spec_period_2)*(periods2<=max_spec_period_2)] = 1

            # Yield data
            yield bX.astype('float32'), bXmask.astype('bool'), bXt.astype('float32'), bY.astype('int32')
            b += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a timeful RNN using PLSTM.')
    # File and path naming stuff
    parser.add_argument('--run_id',   default=os.environ.get('LSB_JOBID',''), help='ID of the run, used in saving.')
    parser.add_argument('--filename', default='freq_task', help='Filename to save model and log to.')
    parser.add_argument('--resume',   default=None, help='Filename to load model and log from.')
    # Control meta parameters
    parser.add_argument('--exp',        default='task1', help='Choose whether to run "task1" (single freq) or "task2" (freq combo) experiment.')
    parser.add_argument('--seed',       default=42,   type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--grad_clip',  default=10.,  type=float, help='Clip the gradient to prevent it from blowing up.')
    parser.add_argument('--batch_size', default=64,   type=int, help='Initialize the random seed of the run (for reproducibility).')
    parser.add_argument('--num_epochs', default=100,  type=int, help='Number of epochs to train for.')
    parser.add_argument('--patience',   default=100,  type=int, help='How long to wait for an increase in validation error before quitting.')
    parser.add_argument('--save_every', default=1000, type=int, help='How many epochs to wait between a save.')
    parser.add_argument('--log_only',   default=0,    type=int, help='Whether to save parameters.')
    # Control architecture and run data
    parser.add_argument('--model_type',       default='plstm', help='Choose which model type to use.')
    parser.add_argument('--batch_norm',       default=0, type=int, help='Batch normalize.')
    parser.add_argument('--sample_regularly', default=0, type=int, help='Whether to sameple regularly or irregularly.')
    parser.add_argument('--sample_res',       default=0.5, type=float, help='Resolution at which to sample.')
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # Constants
    num_train = 5000
    num_test = 500
    arch_size = [None, 110, 2]

    #  Set filename
    if args.exp=='task2':
        comb_filename = '{}_task2_{}_bn_{}_{}'.format(args.filename, args.model_type, args.batch_norm, args.seed)
    else:
        comb_filename = '{}_task1_{}_bn_{}_reg_samp_{}_samp_res_{}_{}'.format(args.filename, args.model_type,
            args.batch_norm, args.sample_regularly, args.sample_res, args.seed)
    if args.run_id != '':
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
    if args.exp=='task2':
        print('Performing Task 2, choosing asynchronous...')
        d = SinWaveComboIterator()
    else:
        d = SinWaveIterator()
    # Save result
    if not args.log_only:
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

        # Call the data generator
        for data in d.flow(batch_size=args.batch_size, num_examples=num_train,
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
        for data in d.flow(batch_size=args.batch_size, num_examples=num_test,
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

        # Save result
        if (epoch+1 % args.save_every) == 0 and not args.log_only:
            print('Saving....')
            save_model(comb_filename, 'recent', network, log)

        # End if there's no improvement in validation error
        best_in_last_set = np.min(log['val_err'][-(args.patience-1):])
        # Drop out if our best round was not in the last set, i.e., no improvement
        if len(log['val_err']) > args.patience and log['val_err'][-args.patience] <= best_in_last_set:
            break

    # Save result
    if args.log_only:
        save_log(comb_filename, 'final', network, log)
    else:
        save_model(comb_filename, 'final', network, log)
    print('Completed.')
