mode=WOT

# check folder 
DIRECTORY=./log_wot
if [ ! -d "$DIRECTORY" ]; then
  mkdir log_wot
fi

## batch size: 128 
# model=alexnet
model=resnet18
# model=resnet34
# model=squeezenet1_0
python wot.py --pretrained --arch ${model} --batch-size 128 --gpu 0 --mode ${mode} --lr 0.0001  |& tee ./log_wot/${model}_${mode}.txt

## batch size: 64
# model=resnet50
# python wot.py --pretrained --arch ${model} --batch-size 64 --gpu 0 --mode ${mode} --lr 0.0001  |& tee ./log_wot/${model}_${mode}.txt

## lr: 1e-5 
# model=vgg16
# python wot.py --pretrained --arch ${model} --batch-size 64 --gpu 0 --mode ${mode} --lr 0.00001  |& tee ./log_wot/${model}_${mode}.txt

## batch size: 32
# model=vgg16_bn
# model=resnet152
# python wot.py --pretrained --arch ${model} --batch-size 32 --gpu 0 --mode ${mode} --lr 0.0001  |& tee ./log_wot/${model}_${mode}.txt
