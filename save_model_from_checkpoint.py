import os
import tensorflow as tf
from datetime import datetime
from model import UNet

if __name__ == '__main__':
    root_dir = 'D:/Work/01_Knowledge_Distillation/'
    print('==================teacher model training==================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    learning_rate = 0.001
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model_save_dir = root_dir + 'models/model(' + start_time + ')'
    checkpoint_path = root_dir + 'checkpoints/2022_08_17-14_03_37/e064-acc0.9821-val_acc0.9797-val_loss0.0685.hdf5'

    ##build model
    print('-----------------------build model------------------------')
    model = UNet(input_layer, classes=classes, init_depth=32, model='unet_logit')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(checkpoint_path)

    ##save
    model.save(model_save_dir)