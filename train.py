#import pytorch modules and submodules
import torch 
from torch import nn,optim 
import torch.nn.functional as F 

#General system libraries 
import time 
import os 
import argparse 
import json

#torchvision related 
from torchvision import datasets, models, transforms, utils 

#supportive ML lib
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image 

#support script
import support 


def model_select(arguments):
    model_name = arguments.arch
    print("\nModel architecture chosen : ",model_name,)

    if model_name == "dense_net":
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features
    elif model_name == "vgg_net":
        model = models.vgg16(pretrained= True)
        in_features = model.classifier[6].in_features

    for param in model.parameters():
        param.requires_grad = False

    print("In features to classifier block : ",in_features)

    classifier_part = support.model_classifier_dyanamics(arguments.classifier_hidden_units_layers,in_features)

    model.classifier = classifier_part

    print("New classifier block attached to the pre trained model is : \n")
    print(model.classifier)

    return model

def plot_losses_accur(train_losses,valid_losses,accur):

    fig, (ax1, ax2) = plt.subplots(1, 2,figsize = (13.5,4.7))


    ax1.plot(train_losses, label='Training loss')
    ax1.plot(valid_losses, label='Validation loss')
    ax1.set_ylim([0,6])
    ax1.set_title('Train & Validation Loss')

    ax2.plot(accur, label = 'Validation accuracy')
    ax2.set_ylim([0,1])
    ax2.set_title('Validation accuracy')

    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    plt.show()


def train_model(model,arguments,device,image_datasets,dataloaders):

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr= arguments.learning_rate)
    model.to(device)

    epochs = arguments.epochs
    steps = 0
    running_loss = 0
    print_every = 10

    train_losses, valid_losses , accur = [],[],[]

    print("\nTraining starts on the model ....\n")

    
    try:
        for e in range(epochs):

            running_loss = 0
            start = time.time()

            print("--------------------------------------------------------------------------------------------------------")
            print("|     Epoch      |     Train loss       |       Validation loss       |      Validation accuracy       |")
            print("|------------------------------------------------------------------------------------------------------|")

            for images,labels in dataloaders['train_loader']:

                steps += 1
                images, labels = images.to(device),labels.to(device)
                optimizer.zero_grad()

                logps = model(images)
                loss = criterion(logps,labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:

                    model.eval()

                    with torch.no_grad():
                        
                        validation_loss,accuracy = support.validate_test(model,dataloaders['validate_loader'],criterion,device)

                    train_losses.append(running_loss/print_every)
                    valid_losses.append(validation_loss/len(dataloaders['validate_loader']))
                    accur.append(accuracy/len(dataloaders['validate_loader']))

                    print(f"|--> Epoch {e+1}/{epochs}   | "
                        f"--> Train loss: {running_loss/print_every:.3f} | "
                        f"--> Validation loss: {validation_loss/len(dataloaders['validate_loader']):.3f}  | "
                        f"--> Validation accuracy: {accuracy/len(dataloaders['validate_loader']):.3f} |")

                    running_loss = 0

                    model.train()


            epoch_dur = time.time() - start
            print("|-----------------------------------------------------------------------------------------------------|")
            print('Epoch "{}" Time taken: {:.0f}m {:.0f}s'.format(e+1,epoch_dur // 60, epoch_dur % 60))

        else:
            print("|-----------------------------------------------------------------------------------------------------|")
            print("\n\nTraining process is completed after {} epochs ".format(epochs))

            support.save_checkpoint(model,arguments,dataloaders['train_loader'],optimizer)

    except KeyboardInterrupt:
        print("\n|-----------------------------------------------------------------------------------------------------|")
        print("\nTraining process interrupted by Keypress")
        print("\nTraining process is completed after {} epochs ".format(e+1))

        support.save_checkpoint(model,arguments,dataloaders['train_loader'],optimizer)

    return model,train_losses,valid_losses,accur

def test_model(model,arguments,device,image_datasets,dataloaders):

    test_loss,test_accuracy = 0,0
    criterion = nn.NLLLoss()

    model.eval()

    with torch.no_grad():
        test_loss,test_accuracy = support.validate_test(model,dataloaders['test_loader'],criterion,device)
    
    len_tl = len(dataloaders['test_loader'])
    print(f'| --> Test Loss: {test_loss/len_tl:.3f} | --> Test Accuracy: {test_accuracy/len_tl:.3f} |')

    model.train()

    return test_loss,test_accuracy


def main():

    #Arguments block 
    
    parser = argparse.ArgumentParser(description= " 102 flower image classifier training script")

    parser.add_argument('--data_dir',type=str,default='/home/shank/.pytorch/flower_102_class/flower_data',help="Train,Validate,Test Dataset path/directory",required=False)
    parser.add_argument("--save_dir",type=str,default="trained_models/model_checkpoint.pth",help="Checkpoint model save directory",required=False)
    
    parser.add_argument('--arch',type=str,default='dense_net',help=' Pretrained model to use "vgg_net16" - "dense_net161"',required=False)

    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate',required=False)
    parser.add_argument('--classifier_hidden_units_layers', type=str, default='512,256,102', help=' Classifier append fc units',required=False)
    parser.add_argument('--epochs', type=int, default=5, help='no of epochs',required=False)
    parser.add_argument('--batch_size',type=int,default=32,help="Batch size in training process")

    parser.add_argument('--gpu',type= bool, default=True,help='Set True to use GPU, if available, Else use CPU',required=False)
    parser.add_argument('--test',type=bool,default=False,help="Run the model on test images and print the parameters",required=False)
    parser.add_argument('--load_model',type = bool,default=False,help="Load a pretrained model checkpoint and train, False to create a new pretrained model ",required=False)
    parser.add_argument('--plot_details',type= bool,default=False,help="Plot the accuracy / losses while training and validation")
    arguments = parser.parse_args()

    #Pre-training block

    print("\nStarting Pre-training process ... \n")

    image_datasets,dataloaders = support.prepare_dataset(arguments)

    cat_to_name = support.cat_to_name('cat_to_name.json')

    device = support.device_select(arguments)

    if arguments.load_model == True:
        print("Loading a pre trained checkpoint\n")
        model = support.load_checkpoint(arguments.save_dir)
    else:
        print('Creating a new pretrained model\n')
        model = model_select(arguments)

    #Training block

    trained_model,train_losses,valid_losses,accur = train_model(model,arguments,device,image_datasets,dataloaders)

    if arguments.plot_details == True:
        plot_losses_accur(train_losses,valid_losses,accur)

    if arguments.test == True:
        test_model(model,arguments,device,image_datasets,dataloaders)


if __name__ == "__main__":
    main()