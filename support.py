from torchvision import transforms,datasets,models
import torch
import json 
from collections import OrderedDict
from torch import nn 

def prepare_dataset(arguments):

    data_dir = arguments.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    transforms_train = transforms.Compose([transforms.RandomRotation(35),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ])

    transforms_validate_test = transforms.Compose([transforms.Resize(255),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                                    ])

    #Load the datasets with ImageFolder

    image_datasets = {}

    image_datasets['train_data'] = datasets.ImageFolder(train_dir,transform =  transforms_train)
    image_datasets['valid_data'] = datasets.ImageFolder(valid_dir,transform = transforms_validate_test)
    image_datasets['test_data'] = datasets.ImageFolder(test_dir,transform = transforms_validate_test)

    #Using the image datasets and the trainforms, define the dataloaders

    dataloaders = {}

    dataloaders['train_loader'] = torch.utils.data.DataLoader(image_datasets['train_data'], batch_size = arguments.batch_size, shuffle = True)
    dataloaders['validate_loader'] = torch.utils.data.DataLoader(image_datasets['valid_data'], batch_size = arguments.batch_size, shuffle = True)
    dataloaders['test_loader'] = torch.utils.data.DataLoader(image_datasets['test_data'], batch_size = arguments.batch_size, shuffle = True)

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train_data', 'valid_data', 'test_data']}
    class_names = image_datasets['train_data'].classes

    print ("Dataset Size: "+ str(dataset_sizes) + "\n")
    n_class = len(class_names)
    print ("Number of classes: \n"+ str(n_class) + "\n")
    print ("Classes: "+ str(class_names) + "\n")

    print("Batch size :",arguments.batch_size)
    print("Batches split :",(dataset_sizes['train_data']//arguments.batch_size + 1),"\n")

    return image_datasets,dataloaders



def cat_to_name(json_path = 'cat_to_name.json'):

    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)

    print("first ten classes & their Names \n")

    for i in range(10):
        print(i+1," : ",cat_to_name[str(i+1)])

    return cat_to_name


def device_select(arguments):

    print("\nDevice to train on ...")

    if arguments.gpu == True:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if device.type == 'cuda':
            print("\nTraning on ",torch.cuda.get_device_name()) 
            print('CUDA is available in the system :) \n')

        else:
            print('CUDA is not available in the system .... Sorry :(  Training on CPU ...')

    else:

        print("Training on CPU ...")

        device = torch.device('cpu')

    return device


def model_classifier_dyanamics(classifier_hidden_units_layers,in_features):

    hidden_units_layers = classifier_hidden_units_layers.split(",")
    hidden_units_layers = [int(item) for item in hidden_units_layers]
    len_hidden = len(hidden_units_layers)

    print("New Classifier_layers no : ",len_hidden,"\n")

    layers = OrderedDict()

    layers['fc1'] = nn.Linear(in_features, hidden_units_layers[0])
    
    for i in range(len_hidden-1):
        layers['relu'+str(i+1)] = nn.ReLU()
        layers['dropout'+str(i+1)] = nn.Dropout(p= 0.4)
        layers['fc'+str(i+2)] = nn.Linear(hidden_units_layers[i],hidden_units_layers[i+1])

    layers['output'] = nn.LogSoftmax(dim=1)

    classifier_part = nn.Sequential(layers)

    return classifier_part


def validate_test(model,loader,criterion,device):
    total_loss,accuracy  = 0,0

    for images, labels in loader:

        images,labels = images.to(device),labels.to(device)

        logps = model(images)
        loss = criterion(logps,labels)
        total_loss += loss.item()

        ps = torch.exp(logps)
        top_ps,top_class = ps.topk(1, dim = 1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor))

    return total_loss,accuracy

def save_checkpoint(model,arguments,train_loader,optimizer):

    model.class_to_idx = train_loader.dataset.class_to_idx

    model.epochs = arguments.epochs
    checkpoint = {'input_size': [3, 224, 224],
                  'output_size': 102,
                  'arch':arguments.arch,
                  'classifier_hidden_units_layers':arguments.classifier_hidden_units_layers,
                  'batch_size': train_loader.batch_size,
                  'state_dict': model.state_dict(),
                  'optimizer_state_dict':optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'classifier':model.classifier,
                  'epoch': model.epochs}

    torch.save(checkpoint, arguments.save_dir)

def load_checkpoint(file_path):

    checkpoint = torch.load(file_path)

    print("\nModel architecture chosen : ",checkpoint['arch'])

    if checkpoint['arch'] == "dense_net":
        model = models.densenet161(pretrained=True)
        in_features = model.classifier.in_features
    elif checkpoint['arch'] == "vgg_net":
        model = models.vgg16(pretrained= True)
        in_features = model.classifier[6].in_features

    for param in model.parameters():
        param.requires_grad = False

    print("In features to classifier block : ",in_features)

    classifier_part = model_classifier_dyanamics(checkpoint['classifier_hidden_units_layers'],in_features)

    model.classifier = classifier_part
    model.load_state_dict(checkpoint['state_dict'])

    print("New classifier block attached to the pre trained model is : \n")
    print(model.classifier)

    return model



if __name__ == "__main__":
    pass