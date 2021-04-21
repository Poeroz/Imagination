modelfile=$1
subset=$2
dataset=$3
python fairseq/scripts/average_checkpoints.py --inputs checkpoints/$modelfile --num-update-checkpoints 5 --output checkpoints/$modelfile/average-model.pt
fairseq-generate data-bin/$dataset --gen-subset $subset --path checkpoints/$modelfile/average-model.pt --beam 5 --remove-bpe --lenpen 0.6 --no-progress-bar --log-format json --left-pad-source False > result/pred_{$modelfile}_{$subset}_avg
