import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks import *
from light_networks import *
from custom_adam import Adam
import csv

parser = argparse.ArgumentParser(description="Proposed1_train")
parser.add_argument("--network", type=str, default="IReDNet", help='name of network')
parser.add_argument("--loss", type=str, default="SSIM", help='loss function')
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate", nargs='+')
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/Proposed1_test", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="datasets/train/Rain12600",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument("--optimizer", type=str, default="CustomAdam", help='Optimizer Adam/SGD/RMSProp/CustomAdam')

opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    print("Network:", opt.network)
    print("Loss:", opt.batch_size)
    print("Optimizer:", opt.optimizer)
    print("Recurrent iter:", opt.recurrent_iter)

    print("Batch size:", opt.batch_size)
    print("Number of epochs:", opt.epochs)
    print("Learning rate decay milestone:", opt.milestone)
    print("Learning rate:", opt.lr)
    print("Log save path:", opt.save_path)
    print("Dath path:", opt.data_path)
    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    if (opt.network == "IteDNet"):
        model = IteDNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "IReDNet"):
        model = IReDNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "IReDNet_LSTM"):
        model = IReDNet_LSTM(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "IReDNet_GRU"):
        model = IReDNet_GRU(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "IReDNet_BiRNN"):
        model = IReDNet_BiRNN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "IReDNet_IndRNN"):
        model = IReDNet_IndRNN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "IReDNet_ConvLSTM"):
        model = IReDNet_ConvLSTM(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "IReDNet_QRNN"):
        model = IReDNet_QRNN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
        
    elif (opt.network == "LightIteDNet"):
        model = LightIteDNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "LightIReDNet"):
        model = LightIReDNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "LightIReDNet_LSTM"):
        model = LightIReDNet_LSTM(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "LightIReDNet_GRU"):
        model = LightIReDNet_GRU(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "LightIReDNet_BiRNN"):
        model = LightIReDNet_BiRNN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "LightIReDNet_IndRNN"):
        model = LightIReDNet_IndRNN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "LightIReDNet_ConvLSTM"):
        model = LightIReDNet_ConvLSTM(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    elif (opt.network == "LightIReDNet_QRNN"):
        model = LightIReDNet_QRNN(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    else:
        raise Exception("Invalid network name.")


    print_network(model)

    # loss function
    if (opt.loss == "MSE"):
        criterion = nn.MSELoss(size_average=False)
    else:
        criterion = SSIM()


    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    if (opt.optimizer == "SGD"):
        print("Use SGD as optimizer")
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    elif (opt.optimizer == "RMSProp"):
        print("Use RMSProp as optimizer")
        optimizer = optim.RMSprop(model.parameters(), lr=opt.lr)
    elif (opt.optimizer == "Adam"):
        print("Use Adam as optimizer")
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    else:
        print("Use CustomAdam as optimizer")
        optimizer = Adam(model.parameters(), lr=opt.lr)

    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))
    
    with open(opt.save_path+"/log.csv", 'w', encoding='UTF8') as f:
        csv_writer = csv.writer(f)
        # write a row to the csv file
        header = ['epoch', 'loss', 'pixel_metric', 'PSNR']
        csv_writer.writerow(header)

        # start training
        step = 0
        for epoch in range(initial_epoch, opt.epochs):
            scheduler.step(epoch)
            for param_group in optimizer.param_groups:
                print('learning rate %f' % param_group["lr"])

            ## epoch training start
            for i, (input_train, target_train) in enumerate(loader_train, 0):
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                input_train, target_train = Variable(input_train), Variable(target_train)

                if opt.use_gpu:
                    input_train, target_train = input_train.cuda(), target_train.cuda()

                out_train, _ = model(input_train)
                pixel_metric = criterion(target_train, out_train)
                if (opt.loss == "NegativeSSIM"):
                    loss = -pixel_metric
                else:
                    loss = pixel_metric

                loss.backward()
                optimizer.step()

                # training curve
                model.eval()
                out_train, _ = model(input_train)
                out_train = torch.clamp(out_train, 0., 1.)
                psnr_train = batch_PSNR(out_train, target_train, 1.)
                
                csv_writer.writerow([epoch+1, loss.item(), pixel_metric.item(), psnr_train])
                print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                    (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

                if step % 10 == 0:
                    # Log the scalar values
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('PSNR on training data', psnr_train, step)
                step += 1
            ## epoch training end

            # log the images
            model.eval()
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
            im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
            im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
            writer.add_image('clean image', im_target, epoch+1)
            writer.add_image('rainy image', im_input, epoch+1)
            writer.add_image('deraining image', im_derain, epoch+1)

            # save model
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
            if epoch % opt.save_freq == 0:
                torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            print(opt.data_path.find('RainTrainH'))
            prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('RainTrainL') != -1:
            prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
        elif opt.data_path.find('Rain12600') != -1:
            prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')
    main()
