# Phased LSTM
This is the official repository of "Phased LSTM: Accelerating Recurrent Network Training for Long or Event-based Sequences," presented as an oral presentation at NIPS 2016, by Daniel Neil, Michael Pfeiffer, and Shih-Chii Liu.

# Freq Task 1
To run the first task, run the shell script "a_freq_task.sh."  It should load the first task with default parameters, training each model under each condition for 70 epochs.  Afterwards, you can open "A_Freq_Task.ipynb" to render the results, which should show the following:

![Freq Task A](/images/task1_acc_bar.png)

# Freq Task 2
To run the second task, run the shell script "b_freq_combo_task.sh."  It should load the second task with default parameters, training each model with the more complex stimuli for 300 epochs (a long time!).  Afterwards, you can open "B_Freq_Combo_Task.ipynb" to render the results, which should show the following:

![Freq Task A](/images/task2_acc.png)

It runs the same Python file as in task 1, but the data iterator is changed to be the more complex version.

# PLSTM Notes
The essence of the PLSTM code is the following lines:

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
This creates the rhythmic mask based on some time_input_n which is a vector of times, one time for each item in the batch.  The timestamp is broadcast to form a 2-tensor of size [batch_size, num_neurons] which contains the timestamp at each neuron for each item in the batch (at one timestep), and stores this in t_broadcast.  We calculate the in_cycle_time, which ranges between 0 and the period length for each neuron.  Then, subsequently, we use that in_cycle_time to figure out if it is in the is_up_phase, is_down_phase, or just the off_phase.  Then, we just use T.switch to apply the correct transformation for each phase.

Once the mask is generated, we simply mask the cell state with the sleep-wake cycle:
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
