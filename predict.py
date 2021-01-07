#import pytorch modules and submodules
import torch 

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
import seaborn as sns 

#support script
import support 


def pre_process_image(image):

    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    img = Image.open(image) 

    img.thumbnail((255,255), Image.ANTIALIAS) 

    width, height = img.size

    left,top,right,bottom = (width - 224)/2, (height - 224)/2, (width + 224)/2 , (height + 224)/2
    img = img.crop((left, top, right, bottom)) 

    np_image = np.array(img)/255 
    
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    final_return = (np_image - mean)/std

    final_return = final_return.transpose((2, 0, 1)) 

    return final_return



def imshow(image, ax=None, title=None):

    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()
        
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.asarray(image).transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.set_title(title)
    
    return ax


def predict(image_path, model,topk,device):

    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    pred_image = torch.from_numpy(np.expand_dims(pre_process_image(image_path), axis=0)).type(torch.FloatTensor)

    model.to(device)
    model.eval()

    with torch.no_grad():
        pred_image = pred_image.to(device)

        log_ps = model.forward(pred_image)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(topk, dim=1)

    model.train()

    idx_to_class = {idx:clas for clas,idx in model.class_to_idx.items()}

    top_p,top_class = top_p.reshape(-1).tolist(),top_class.reshape(-1).tolist()

    top_class = [idx_to_class[item] for item in top_class] 

    return top_p,top_class


def plot_show_inference(image_path,classes_top,probs):
    
    plt.figure(figsize=(8,8))

    ax = plt.subplot(2,1,1)    
    imshow(pre_process_image(image_path), ax, classes_top[0])

    ax = plt.subplot(2,1,2)
    sns.barplot(x=probs, y=classes_top, palette=sns.color_palette(n_colors=1))

    plt.show()



def main():
    parser = argparse.ArgumentParser(description= " 102 flower image classifier prediction script")

    parser.add_argument('--image_path',type=str,default='pred_images/grape_hyacinth.jpeg',help="Path for Image to be predicted",required=False)
    parser.add_argument("--model_dir",type=str,default="trained_models/model_checkpoint.pth",help="Checkpoint model directory",required=False)
    
    parser.add_argument('--top_k',type=int,default=3,help="K classes to return after prediction",required=False)

    parser.add_argument('--cat_to_name_json', type=str, default="cat_to_name.json", help=' json file path for category to names ',required=False)
    parser.add_argument('--gpu',type=bool,default=True,help=" Use GPU for inference")
    parser.add_argument('--plot_show_inference',type=bool,default=False,help="Plot & show inference")

    arguments = parser.parse_args()

    #load model

    cat_to_name = support.cat_to_name(arguments.cat_to_name_json)

    print("\nLoading pre trained model checkpoint\n")
    model = support.load_checkpoint(arguments.model_dir)

    device = support.device_select(arguments)

    probs, classes = predict(arguments.image_path, model,arguments.top_k,device)

    classes_top = [cat_to_name[item] for item in classes]

    print(classes_top,probs)

    if arguments.plot_show_inference == True:
        plot_show_inference(arguments.image_path,classes_top,probs)


if __name__ == "__main__":
    main()