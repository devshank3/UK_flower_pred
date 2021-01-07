### ML with pytorch Udacity project-deeplearning

#### 102 UK flower classification 
---
Dataset link:

www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html

Jupyter notebooks and HTML submission

#### Scripts
---
train.py

arguments:
```
  -h, --help :   show this help message and exit

  --data_dir  :  Train,Validate,Test Dataset path/directory

  --save_dir  :  Checkpoint model save directory

  --arch ARCH :  Pretrained model to use "vgg_net16" - "dense_net161"

  --learning_rate  : learning rate

  --classifier_hidden_units_layers  : Classifier append fc units

  --epochs  :   no of epochs

  --batch_size  : Batch size in training process

  --gpu GPU   :   Set True to use GPU, if available, Else use CPU

  --test TEST  :  Run the model on test images and print the parameters

  --load_model LOAD_MODEL : Load a pretrained model checkpoint and train, False to create a new pretrained model

  --plot_details PLOT_DETAILS : Plot the accuracy / losses while training and validation
  ```

support.py

supporting script for training and prediction

predict.py
```
-h, --help    :   show this help message and exit

--image_path  :  Path for Image to be predicted

--model_dir   :  Checkpoint model directory

--top_k       :  K classes to return after prediction

--cat_to_name_json :  json file path for category to names

--gpu      :    Use GPU for inference

--plot_show_inference  :  Plot & show inference
```

