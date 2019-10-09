mode=admm

# check folder 
DIRECTORY=./log_admm
if [ ! -d "$DIRECTORY" ]; then
  mkdir log_admm
fi

model=resnet18
python wot.py --pretrained --arch ${model} --batch-size 128 --gpu 0 --mode ${mode} --lr 0.0001  |& tee ./log_admm/${model}_${mode}.txt


