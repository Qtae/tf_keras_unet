import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self, input_layer, classes, init_depth, model='unet_logit'):
        super(UNet, self).__init__()
        self.classes = classes
        if (model == 'unet'):
            self.model = self.unet(input_layer, classes, init_depth=init_depth)
        else:
            self.model = self.unet_logit(input_layer, classes, init_depth=init_depth)

    def compile(self, optimizer, metrics, loss):
        super(UNet, self).compile(optimizer=optimizer, metrics=metrics, loss=loss)
        self.model.compile(optimizer=optimizer, metrics=metrics, loss=loss)

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss = self.compiled_loss(y, logits)

        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.compiled_metrics.update_state(y, tf.nn.softmax(logits))
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def test_step(self, data):
        x, y = data

        logits = self.model(x, training=False)
        loss = self.compiled_loss(y, logits)
        self.compiled_metrics.update_state(y, tf.nn.softmax(logits))
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        return results

    def unet(self, input_layer, classes, init_depth=16):
        conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(input_layer)
        conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv1)
        conv2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv2)
        conv2 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv2)
        conv3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv3)
        conv3 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv3)
        conv4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv4)
        conv4 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv4)
        conv5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(init_depth * 16, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv5)
        conv5 = tf.keras.layers.Conv2D(init_depth * 16, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv5)
        # up1 = tf.keras.layers.Conv2DTranspose(init_depth*16, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
        up1 = tf.keras.layers.Conv2D(init_depth * 16, 3, padding='same', activation='relu',
                                     kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)))
        concat1 = tf.keras.layers.concatenate([conv4, up1], axis=3)

        conv6 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(concat1)
        conv6 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv6)
        # up2 = tf.keras.layers.Conv2DTranspose(init_depth*8, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
        up2 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                     kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)))
        concat2 = tf.keras.layers.concatenate([conv3, up2], axis=3)

        conv7 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(concat2)
        conv7 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv7)
        # up3 = tf.keras.layers.Conv2DTranspose(init_depth*4, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
        up3 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                     kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)))
        concat3 = tf.keras.layers.concatenate([conv2, up3], axis=3)

        conv8 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(concat3)
        conv8 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv8)
        # up4 = tf.keras.layers.Conv2DTranspose(init_depth*2, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
        up4 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                     kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)))
        concat4 = tf.keras.layers.concatenate([conv1, up4], axis=3)

        conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(concat4)
        conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv9)

        output_conv = tf.keras.layers.Conv2D(classes, 1, padding='same', kernel_initializer='he_normal')(conv9)

        output_sftmx = tf.keras.activations.softmax(output_conv)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_sftmx)
        return model

    def unet_logit(self, input_layer, classes, init_depth=16):
        conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(input_layer)
        conv1 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv1)
        conv2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv1)

        conv2 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv2)
        conv2 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv2)
        conv3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv2)

        conv3 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv3)
        conv3 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv3)
        conv4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv3)

        conv4 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv4)
        conv4 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv4)
        conv5 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(conv4)

        conv5 = tf.keras.layers.Conv2D(init_depth * 16, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv5)
        conv5 = tf.keras.layers.Conv2D(init_depth * 16, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv5)
        # up1 = tf.keras.layers.Conv2DTranspose(init_depth*16, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
        up1 = tf.keras.layers.Conv2D(init_depth * 16, 3, padding='same', activation='relu',
                                     kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv5)))
        concat1 = tf.keras.layers.concatenate([conv4, up1], axis=3)

        conv6 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(concat1)
        conv6 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv6)
        # up2 = tf.keras.layers.Conv2DTranspose(init_depth*8, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
        up2 = tf.keras.layers.Conv2D(init_depth * 8, 3, padding='same', activation='relu',
                                     kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)))
        concat2 = tf.keras.layers.concatenate([conv3, up2], axis=3)

        conv7 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(concat2)
        conv7 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv7)
        # up3 = tf.keras.layers.Conv2DTranspose(init_depth*4, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
        up3 = tf.keras.layers.Conv2D(init_depth * 4, 3, padding='same', activation='relu',
                                     kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)))
        concat3 = tf.keras.layers.concatenate([conv2, up3], axis=3)

        conv8 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(concat3)
        conv8 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv8)
        # up4 = tf.keras.layers.Conv2DTranspose(init_depth*2, 3, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
        up4 = tf.keras.layers.Conv2D(init_depth * 2, 3, padding='same', activation='relu',
                                     kernel_initializer='he_normal')((tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)))
        concat4 = tf.keras.layers.concatenate([conv1, up4], axis=3)

        conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(concat4)
        conv9 = tf.keras.layers.Conv2D(init_depth, 3, padding='same', activation='relu',
                                       kernel_initializer='he_normal')(conv9)

        output_conv = tf.keras.layers.Conv2D(classes, 1, padding='same', kernel_initializer='he_normal')(conv9)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_conv)
        return model

    def summary(self):
        self.model.summary()
