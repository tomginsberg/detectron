nohup python rejectron/train.py -g 0 -b 512 -d camelyon -w 12 -t 10 > cam10.log &
nohup python rejectron/train.py -g 1 -b 512 -d camelyon -w 24 -t 20 >cam20.log &
nohup python rejectron/train.py -g 2 -b 512 -d camelyon -w 12 -t 50 >cam50.log &

