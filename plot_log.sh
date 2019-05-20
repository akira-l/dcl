#!/bin/bash

recent=$(ls -t *.log |head -1)
echo $recent
log_folder="./grep_log"
if [ ! -d $log_folder ]; then
    mkdir $log_folder
fi
grep " trainval-acc@" $recent |awk '{print $6, $8, $10, $12}' > $log_folder/train_acc
grep " val-acc@" $recent |awk '{print $6, $8, $10, $12}'> $log_folder/val_acc
grep " loss=" $recent |awk '{print $6, $8, $10, $12}' > $log_folder/loss

python utils/plot_log.py --dir $log_folder


