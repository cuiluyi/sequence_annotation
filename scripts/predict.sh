export CUDA_VISIBLE_DEVICES=7

python predict.py --model_path ckpts/model_epoch_5.pth \
> logs/predict_epoch_5.log 2>&1