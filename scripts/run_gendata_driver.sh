num_epochs=100
batch_size=128
augment_data=True
save_by_minibatch=True
save_freq=3
use_adam=False
use_bn=False
max_saved=150
lr=0.0001
dr=0.01

time python ../GenData/driver.py \
    --output_path="/y/vfatemeh/NN/ExpWinter2021/exp/tmp" \
    --lr=${lr} \
    --dr=${dr} \
    --num_epochs=${num_epochs} \
    --batch_size=${num_epochs} \
    --save_freq=${save_freq} \
    --momentum=0.9 \
    --use_adam=False \
    --save_by_minibatch=${save_by_minibatch} \
    --augment_data=${augment_data} \
    --use_bn=${use_bn} \
    --max_saved=${max_saved} \
    --rep_num=0

# echo "***************************************"
# echo "*     FINISHED DRIVER 1               *"
# echo "**************************************"

# num_epochs=100
# batch_size=128
# augment_data=True
# save_by_minibatch=True
# save_freq=3
# use_adam=False
# use_bn=False
# max_saved=150

# for lr in 0.000002 0.00002 0.0002 0.002 0.02 0.2 0.0000025 0.000025 0.00025 0.0025 0.025 0.25
    # do 
    # for dr in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    # do
        # echo "***************************************"
        # echo "*             Driver 2 {$lr} -- {$dr} *"
        # echo "**************************************"

        # time python /z/pujat/projects_summer_2020/NN_GenData/driver.py \
            # --output_path="/y/vfatemeh/NN/ExpWinter2021/exp/" \
            # --lr=${lr} \
            # --dr=${dr} \
            # --num_epochs=${num_epochs} \
            # --batch_size=${num_epochs} \
            # --save_freq=${save_freq} \
            # --momentum=0.9 \
            # --use_adam=False \
            # --save_by_minibatch=${save_by_minibatch} \
            # --augment_data=${augment_data} \
            # --use_bn=${use_bn} \
            # --max_saved=${max_saved} \
            # --rep_num=0
    # done 
# done

# echo "***************************************"
# echo "*     FINISHED DRIVER 2               *"
# echo "**************************************"

# num_epochs=100
# batch_size=128
# augment_data=True
# save_by_minibatch=True
# save_freq=3
# use_adam=False
# use_bn=False
# max_saved=150

# for lr in 0.000003 0.00003 0.0003 0.003 0.03 0.3 0.0000025 0.000035 0.00035 0.0035 0.035 0.35
    # do 
    # for dr in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    # do
        # echo "***************************************"
        # echo "*     Driver 3   {$lr} -- {$dr}       *"
        # echo "**************************************"

        # time python /z/pujat/projects_summer_2020/NN_GenData/driver.py \
            # --output_path="/y/vfatemeh/NN/ExpWinter2021/exp/" \
            # --lr=${lr} \
            # --dr=${dr} \
            # --num_epochs=${num_epochs} \
            # --batch_size=${num_epochs} \
            # --save_freq=${save_freq} \
            # --momentum=0.9 \
            # --use_adam=False \
            # --save_by_minibatch=${save_by_minibatch} \
            # --augment_data=${augment_data} \
            # --use_bn=${use_bn} \
            # --max_saved=${max_saved} \
            # --rep_num=0
    # done 
# done 


# echo "***************************************"
# echo "*     FINISHED DRIVER 3               *"
# echo "**************************************"

# num_epochs=100
# batch_size=128
# augment_data=True
# save_by_minibatch=True
# save_freq=3
# use_adam=False
# use_bn=False
# max_saved=150

# for lr in 0.000004 0.00004 0.0004 0.004 0.04 0.4 0.0000025 0.000045 0.00045 0.0045 0.045 0.45
    # do 
    # for dr in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    # do
        # echo "***************************************"
        # echo "*     Driver 4 {$lr} -- {$dr}       *"
        # echo "**************************************"

        # time python /z/pujat/projects_summer_2020/NN_GenData/driver.py \
            # --output_path="/y/vfatemeh/NN/ExpWinter2021/exp/" \
            # --lr=${lr} \
            # --dr=${dr} \
            # --num_epochs=${num_epochs} \
            # --batch_size=${num_epochs} \
            # --save_freq=${save_freq} \
            # --momentum=0.9 \
            # --use_adam=False \
            # --save_by_minibatch=${save_by_minibatch} \
            # --augment_data=${augment_data} \
            # --use_bn=${use_bn} \
            # --max_saved=${max_saved} \
            # --rep_num=0
    # done 
# done 

# echo "***************************************"
# echo "*     FINISHED ALL DRIVERS            *"
# echo "**************************************"
