import tensorflow as tf
from base.base_model import BaseModel
from base.base_trainer import focal_loss
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Activation, Dropout, GlobalAveragePooling2D, Concatenate, concatenate, MaxPooling2D
from keras.models import Model
from keras import optimizers
import os

class CustomModel(BaseModel):
    def __init__(self, config):
        super(CustomModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        print 'Building model'
        input_shape = (196, )
        inputX = Input(shape = input_shape)
        x = Dense(256, activation='relu')(inputX)
        x = Dropout(self.config.trainer.dropout)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.config.trainer.dropout)(x)
        outputY = Dense(2, activation='softmax')(x)


        '''
        input_shape = (14, 14, 1)
        inputX = Input(shape = input_shape)

        x = Conv2D(64, kernel_size=(4, 4), strides=2, activation='relu')(inputX)
        x = Dropout(self.config.trainer.dropout)(x)
        x = Conv2D(128, kernel_size=(4, 4), strides=2, activation='relu')(x)
        x = Dropout(self.config.trainer.dropout)(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(self.config.trainer.dropout)(x)
        outputY = Dense(2, activation='softmax')(x)
        '''
        self.model = Model(inputs=inputX, outputs=outputY)
        self.model.summary()


        if self.config.trainer.optimizer == 'sgd':
            optimizer = optimizers.SGD(lr=self.config.LRFinder.learning_rate,
                                        decay=self.config.LRFinder.decay_rate)
            print '-'*50
            print 'Compiling model with SGD'
            print '-'*50
        else:
            optimizer = optimizers.Adam(lr=self.config.LRFinder.learning_rate,
                                        decay=self.config.LRFinder.decay_rate)
            print '-'*50
            print 'Compiling model with Adam'
            print '-'*50

        if self.config.use_focal_loss:
            loss = [focal_loss(config=self.config)]
            print 'Loss : Focal Loss'
        else:
            loss = 'categorical_crossentropy'
            print 'Loss : ' + loss

        self.model.compile(optimizer=optimizer,
                           loss=loss,
                           metrics=['accuracy'])

        if self.config.load_checkpoint != '':
            self.model.load_weights(self.config.load_checkpoint)
            print 'Successfully loaded weights fom file : %s' % (self.config.load_checkpoint)
