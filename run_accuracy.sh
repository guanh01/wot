
# check folder 
DIRECTORY=./log_acc
if [ ! -d "$DIRECTORY" ]; then
  mkdir log_acc 
fi

# measure accuracy 
for model in alexnet 
# for model in googlenet inception_v3 mobilenet_v2 resnet101 resnet152 
# for model in resnet18 resnet34 resnet50 squeezenet1_0 squeezenet1_1
# for model in vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
do 
  python accuracy.py --original-acc --quantized-acc --w-dist --model-name $model |& tee ./log_acc/${model}_acc.txt
done 
