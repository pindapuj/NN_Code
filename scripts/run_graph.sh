exp_path="/cluster/home/it_stu158/iris/vgg1"
time python3 -u /cluster/home/it_stu158/iris/GenGraph/driver.py --glob True\
	--input_path=${exp_path} \
	--glob='True' \
	--weight_transform='identity' \
	--num_convert=25
