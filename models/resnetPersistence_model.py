import tensorflow as tf
from base.base_model import BaseModel
from base.base_trainer import focal_loss
import keras
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Activation, Dropout, GlobalAveragePooling2D, Concatenate, concatenate
from keras.models import Model
from keras import optimizers
from keras import backend as K
import os

from utils.metrics import auc, f1, precision, recall


class ResNetPersistenceModel(BaseModel):
    def __init__(self, config):
        super(ResNetPersistenceModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        print 'Building model'
        input_shape = (224, 224, 3)
        inputX = Input(shape = input_shape)
        print 'ResNet initialization : ', self.config.trainer.weights
        if self.config.trainer.weights == 'None':
            self.config.trainer.weights = None
        resnet = keras.applications.ResNet50(weights=self.config.trainer.weights, include_top=False, pooling='avg', input_shape=input_shape)
        x = resnet(inputX)
        x = Dropout(self.config.trainer.dropout)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.config.trainer.dropout)(x)
        x = Dense(2)(x)
        outputY = Activation('softmax')(x)
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
                           metrics=[precision, recall, f1, auc, 'accuracy'])

        if self.config.load_checkpoint != '':
            self.model.load_weights(self.config.load_checkpoint)
            print 'Successfully loaded weights fom file : %s' % (self.config.load_checkpoint)
