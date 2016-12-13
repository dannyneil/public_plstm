# Phased LSTM
This is the official repository of "Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences," presented as an oral presentation at NIPS 2016, by Daniel Neil, Michael Pfeiffer, and Shih-Chii Liu.

## Rule of Thumb
In general, **if you are using ~1000 timesteps or more in your input sequence, you can benefit from PLSTM.**

If you're only answering bAbI tasks or doing negative log-likelihood on some paragraph of text, you're unlikely to see improvement from this model.  However, for long sequences, or sequences which are fusing input from multiple sensors with different timing (e.g., one going at 3 Hz and the other at 25 Hz), this model is both natural and efficient.

# Freq Task 1
To run the first task, run the shell script [a_freq_task.sh](/a_freq_task.sh).  It should load the first task with default parameters, training each model under each condition for 70 epochs.  Afterwards, you can open [A_Freq_Task.ipynb](/A_Freq_Task.ipynb) to render the results, which should show the following:

![Freq Task A](/images/task1_acc_bar.png)

# Freq Task 2
To run the second task, run the shell script [b_freq_combo_task.sh](/b_freq_combo_task.sh).  It should load the second task with default parameters, training each model with the more complex stimuli for 300 epochs (a long time!).  Afterwards, you can open [B_Freq_Combo_Task.ipynb](/B_Freq_Combo_Task.ipynb) to render the results, which should show the following:

![Freq Task A](/images/task2_acc.png)

It runs the same Python file as in task 1, but the data iterator is changed to be the more complex version.

# PLSTM Notes
The essence of the PLSTM code ([plstm.py](/plstm.py#L355-L370)) is the following lines:

```python
def calc_time_gate(time_input_n):
    # Broadcast the time across all units
    t_broadcast = time_input_n.dimshuffle([0,'x'])
    # Get the time within the period
    in_cycle_time = T.mod(t_broadcast + shift_broadcast, period_broadcast)
    # Find the phase
    is_up_phase = T.le(in_cycle_time, on_mid_broadcast)
    is_down_phase = T.gt(in_cycle_time, on_mid_broadcast)*T.le(in_cycle_time, on_end_broadcast)
    # Set the mask
    sleep_wake_mask = T.switch(is_up_phase, in_cycle_time/on_mid_broadcast,
                        T.switch(is_down_phase,
                            (on_end_broadcast-in_cycle_time)/on_mid_broadcast,
                                off_slope*(in_cycle_time/period_broadcast)))

    return sleep_wake_mask
```
This creates the rhythmic mask based on some `time_input_n` which is a vector of times, one time for each item in the batch.  The timestamp is broadcast to form a 2-tensor of size `[batch_size, num_neurons]` which contains the timestamp at each neuron for each item in the batch (at one timestep), and stores this in `t_broadcast`.  We calculate the `in_cycle_time`, which ranges between 0 and the period length for each neuron.  Then, subsequently, we use that `in_cycle_time` to figure out if it is in the `is_up_phase`, `is_down_phase`, or just the off_phase.  Then, we just use `T.switch` to apply the correct transformation for each phase.

Once the mask is generated, we simply mask the cell state with the sleep-wake cycle ([plstm.py](/plstm.py#L380-L381)):
```python
def step_masked(input_n, time_input_n, mask_n, cell_previous, hid_previous, *args):
    cell, hid = step(input_n, time_input_n, cell_previous, hid_previous, *args)

    # Get time gate openness
    sleep_wake_mask = calc_time_gate(time_input_n)

    # Sleep if off, otherwise stay a bit on
    cell = sleep_wake_mask*cell + (1.-sleep_wake_mask)*cell_previous
    hid = sleep_wake_mask*hid + (1.-sleep_wake_mask)*hid_previous
```

# Implementation notes
PLSTM was originally written in Theano.  There are some subtle differences between e.g., Theano and Tensorflow.  Some issues worth keeping in mind are:

 * Make sure r_on can't be negative
 * Make sure the period can't be negative
 * Check to see what mod(-1, 5) is to make sure it lines up with your intuition (e.g., negative symmetric or cyclical)
 * Think about whether or not you want to `abs` the phase shift

Also note that this doesn't take advantage of any sparse BLAS code.  The latest TensorFlow code has some good CuSPARSE support, and the gemvi sparse instructions are great for computing the `dense_matrix x sparse` vector operations we need for Phased LSTM, and should absolutely offer speedups at the sparsity levels that are shown here.  But, as far as I know, no one has yet publicly implemented this.

# Default parameters
Generally, for "standard" tasks, you have an input of several hundred to a couple thousand steps and your neurons tend to be overcomplete.  For this situation, the default parameters given here are pretty good:

 * Period drawn from `np.exp(np.random.uniform(1, 6))`, i.e., (2.71, 403) timesteps per cycle, where 5e is as likely as 50e.
 * An on ratio of around 5%; sometimes, for hard problems, you'll need to either turn on learning for this parameter, which gradually expands r_on towards 100% (because why not; the neuron will always decrease loss if it is on more often.  Hint: think about adding an L2 cost to this, which is equivalent to having SGD find an accurate solution while minimizing compute cost, which is its own interesting topic).  Alternatively, you can fix it at 10%, which generally seems like a good number so far.
 * A phase shift drawn from all possible phase shifts.  If you don't cover all phase shifts, or don't have enough neurons, you'll have "holes" in time where no neurons are paying attention.
 * The "timestamp" for a standard input is the integer time index, ranging from 0 to num_timesteps.

# Other Tasks
Other tasks are coming soon, when I can clean them up.

# Citation
Please use this citation, if the code or paper was useful in your work:

```
@inproceedings{neil2016phased,
  title={Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences},
  author={Neil, Daniel and Pfeiffer, Michael and Liu, Shih-Chii},
  booktitle={Advances In Neural Information Processing Systems},
  pages={3882--3890},
  year={2016}
}
```
