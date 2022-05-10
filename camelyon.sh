nohup python rejectron/train.py -g 0 -b "$BATCHSIZE" -d camelyon -w 24 -t 10 >camelyon-10.log &
nohup python rejectron/train.py -g 1 -b "$BATCHSIZE" -d camelyon -w 24 -t 20 >camelyon-20.log &
nohup python rejectron/train.py -g 0 -b 512 -d camelyon -w 24 -t 50 >camelyon-50.log &
# default $BATCHSIZE = 512
