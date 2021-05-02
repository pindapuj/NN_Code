for exp in exp_lr_00015_momentum_9_dr_4_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00015_momentum_9_dr_5_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00015_momentum_9_dr_6_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00015_momentum_9_dr_7_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00015_momentum_9_dr_8_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00015_momentum_9_dr_9_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00025_momentum_9_dr_4_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00025_momentum_9_dr_5_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00025_momentum_9_dr_6_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00025_momentum_9_dr_7_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00025_momentum_9_dr_8_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00025_momentum_9_dr_9_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00035_momentum_9_dr_6_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00035_momentum_9_dr_7_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00035_momentum_9_dr_8_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00035_momentum_9_dr_9_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00045_momentum_9_dr_2_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00045_momentum_9_dr_3_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00045_momentum_9_dr_4_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00045_momentum_9_dr_5_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00045_momentum_9_dr_6_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00045_momentum_9_dr_7_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00045_momentum_9_dr_8_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_00045_momentum_9_dr_9_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0015_momentum_9_dr_0_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0015_momentum_9_dr_1_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0015_momentum_9_dr_2_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0025_momentum_9_dr_0_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0025_momentum_9_dr_1_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0025_momentum_9_dr_2_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0035_momentum_9_dr_0_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0035_momentum_9_dr_1_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0035_momentum_9_dr_2_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0035_momentum_9_dr_3_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0035_momentum_9_dr_4_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0\
    exp_lr_0045_momentum_9_dr_0_bs_128_ne_100_sf_10_sgd_AUGDATA_sbmb_0

do
	exp_path="/y/vfatemeh/NN/ExpWinter2021/exp/"
	exp_path+=${exp}
	time python /z/pujat/projects_summer_2020/NN_GenGraph/driver-fv.py \
		--input_path=${exp_path} \
		--glob='False' \
		--output_path=${exp_path} \
		--weight_transform='identity' \
		--num_convert=25 
done 

