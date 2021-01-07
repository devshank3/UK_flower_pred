102 category flower classifier


train.py

102 flower image classifier training script

arguments:

  -h, --help :           show this help message and exit

  --data_dir DATA_DIR :  Train,Validate,Test Dataset path/directory

  --save_dir SAVE_DIR :  Checkpoint model save directory

  --arch ARCH      :     Pretrained model to use "vgg_net16" - "dense_net161"

  --learning_rate LEARNING_RATE : learning rate

  --classifier_hidden_units_layers CLASSIFIER_HIDDEN_UNITS_LAYERS : Classifier append fc units

  --epochs EPOCHS    :   no of epochs

  --batch_size BATCH_SIZE : Batch size in training process

  --gpu GPU       :     Set True to use GPU, if available, Else use CPU

  --test TEST     :      Run the model on test images and print the parameters

  --load_model LOAD_MODEL : Load a pretrained model checkpoint and train, False to create a new pretrained model

  --plot_details PLOT_DETAILS : Plot the accuracy / losses while training and validation