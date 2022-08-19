import tensorflow as tf
from datetime import datetime
from model import UNet
from data import load_dataset, make_one_hot
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

if __name__ == '__main__':
    root_dir = 'D:/Work/99_Unet_TFKeras/'
    print('==================model training==================')
    start_time = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    classes = 2
    epochs = 500
    batch_size = 2
    learning_rate = 0.001
    input_layer = tf.keras.layers.Input([640, 640, 1])
    model_save_dir = root_dir + 'models/model(' + start_time + ')'

    ##load dataset
    print('-----------------------load dataset-----------------------')
    data_dir = root_dir + 'data/train'
    train_images, train_labels, valid_images, valid_labels = load_dataset(data_dir, valid_ratio=0.1)
    img_num = train_images.shape[0]
    val_img_num = valid_images.shape[0]
    steps_per_epoch = int(img_num/batch_size) + bool(img_num%batch_size)
    validation_steps = int(val_img_num/batch_size) + bool(val_img_num%batch_size)

    train_images = tf.data.Dataset.from_tensor_slices(train_images)
    train_images = train_images.map(lambda x: tf.image.random_brightness(x, 0.15),
                                    num_parallel_calls=AUTOTUNE)
    train_labels = tf.data.Dataset.from_tensor_slices(train_labels)
    train_labels = train_labels.map(lambda x: make_one_hot(x, tf.constant(classes, dtype=tf.int32)),
                                    num_parallel_calls=AUTOTUNE)
    train_dataset = tf.data.Dataset.zip((train_images, train_labels))
    train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)

    valid_images = tf.data.Dataset.from_tensor_slices(valid_images)
    valid_images = valid_images.map(lambda x: tf.image.random_brightness(x, 0.15),
                                    num_parallel_calls=AUTOTUNE)
    valid_labels = tf.data.Dataset.from_tensor_slices(valid_labels)
    valid_labels = valid_labels.map(lambda x: make_one_hot(x, tf.constant(classes, dtype=tf.int32)),
                                    num_parallel_calls=AUTOTUNE)
    valid_dataset = tf.data.Dataset.zip((valid_images, valid_labels))
    valid_dataset = valid_dataset.batch(batch_size).prefetch(AUTOTUNE)
    

    ##build model
    print('-----------------------build model------------------------')
    model = UNet(input_layer, classes=classes, init_depth=32, model='unet_logit')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    ##callbacks
    #early stoping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, mode='auto')
    #reduce learning rate on plateau
    reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.7, patience=5, cooldown=10,
                                                  min_lr=0.00001, mode='auto')
    #add callbacks
    callbacks_list = [early_stopping, reduce]

    ##train
    print('--------------------------train---------------------------')
    model.summary()
    history = model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=valid_dataset,
                        validation_steps=validation_steps, callbacks=callbacks_list, class_weight=None, initial_epoch=0)

    model.save(model_save_dir)