nohup python rejectron/train.py -g 0 -b 1024 -d cifar -w 12 -t 10 -dc True > cifar10dom.log &
nohup python rejectron/train.py -g 1 -b 1024 -d cifar -w 12 -t 20 > cifar20.log &
nohup python rejectron/train.py -g 2 -b 1024 -d cifar -w 12 -t 50 > cifar50.log &