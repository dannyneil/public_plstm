#!/bin/bash
set -e

EPOCHS=70
for i in `seq 1 5`
do
    # Async
    python freq_task.py --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type plstm --batch_norm 0 --batch_size 32 --sample_regularly 0 --sample_res 0.0
    python freq_task.py --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type lstm --batch_norm 1 --batch_size 32 --sample_regularly 0 --sample_res 0.0
    python freq_task.py --run_id regular_0_seed_${i} --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type lstm --batch_norm 0 --batch_size 32 --sample_regularly 0 --sample_res 0.0
    # Resolution 1
    python freq_task.py --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type plstm --batch_norm 0 --batch_size 32 --sample_regularly 1 --sample_res 1.0
    python freq_task.py --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type lstm --batch_norm 1 --batch_size 32 --sample_regularly 1 --sample_res 1.0
    python freq_task.py --run_id regular_0_seed_${i} --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type lstm --batch_norm 0 --batch_size 32 --sample_regularly 1 --sample_res 1.0

    # Resolution .1
    python freq_task.py --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type plstm --batch_norm 0 --batch_size 32 --sample_regularly 1 --sample_res 0.1
    python freq_task.py --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type lstm --batch_norm 1 --batch_size 32 --sample_regularly 1 --sample_res 0.1
    python freq_task.py --run_id regular_0_seed_${i} --seed $i --num_epochs $EPOCHS --patience $EPOCHS \
                        --model_type lstm --batch_norm 0 --batch_size 32 --sample_regularly 1 --sample_res 0.1
done