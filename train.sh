#### Ablation study train examples
### Loss function 
# python train.py --network IReDNet --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_NegativeSSIM --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet --loss SSIM --save_path logs/Rain100L/IReDNet_SSIM --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet --loss MSE --save_path logs/Rain100L/IReDNet_MSE --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam

# python train.py --network LightIReDNet --loss NegativeSSIM --save_path logs/Rain100L/LightIReDNet_NegativeSSIM --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet --loss SSIM --save_path logs/Rain100L/LightIReDNet_SSIM --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet --loss MSE --save_path logs/Rain100L/LightIReDNet_MSE --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam

### Network architecture
## Recurrent Layers Analysis
python train.py --network IteDNet --loss NegativeSSIM --save_path logs/Rain100L/IteDNet --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet_LSTM --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_LSTM --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet_GRU --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_GRU --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet_BiRNN --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_BiRNN --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet_IndRNN --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_IndRNN --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet_ConvLSTM --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_ConvLSTM --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet_QRNNs --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_QRNNs --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network IReDNet --loss NegativeSSIM --save_path logs/Rain100L/IReDNet --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam

# python train.py --network LightIteDNet --loss NegativeSSIM --save_path logs/Rain100L/LightIteDNet --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet_LSTM --loss NegativeSSIM --save_path logs/Rain100L/LightIReDNet_LSTM --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet_GRU --loss NegativeSSIM --save_path logs/Rain100L/LightIReDNet_GRU --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet_BiRNN --loss NegativeSSIM --save_path logs/Rain100L/LightIReDNet_BiRNN --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet_IndRNN --loss NegativeSSIM --save_path logs/Rain100L/LightIReDNet_IndRNN --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet_ConvLSTM --loss NegativeSSIM --save_path logs/Rain100L/LightIReDNet_ConvLSTM --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet_QRNN --loss NegativeSSIM --save_path logs/Rain100L/LightIReDNet_QRNN --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --network LightIReDNet --loss NegativeSSIM --save_path logs/Rain100L/LightIReDNet --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam


## Effects of recursive stage numbers
# python train.py --recurrent_iter 2 --network IReDNet --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_2 --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --recurrent_iter 3 --network IReDNet --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_3 --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --recurrent_iter 4 --network IReDNet --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_4 --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --recurrent_iter 5 --network IReDNet --loss NegativeSSIM --save_path logs/Rain100L/IReDNet_5 --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam

# python train.py --recurrent_iter 2 --network LightIteDNet --loss NegativeSSIM --save_path logs/Rain100L/LightIteDNet_2 --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --recurrent_iter 3 --network LightIteDNet --loss NegativeSSIM --save_path logs/Rain100L/LightIteDNet_3 --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --recurrent_iter 4 --network LightIteDNet --loss NegativeSSIM --save_path logs/Rain100L/LightIteDNet_4 --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
# python train.py --recurrent_iter 5 --network LightIteDNet --loss NegativeSSIM --save_path logs/Rain100L/LightIteDNet_5 --data_path datasets/train/RainTrainL --batch_size 4 --epochs 25 --milestone 8 16 20 --optimizer Adam
