from base.base_model import BaseModel
from base.base_trainer import focal_loss
from keras.layers import Input, Dense, Conv2D, Activation, Dropout, GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras import optimizers
import os
import numpy as np
from keras import backend as K
from sklearn import metrics

from utils.metrics import auc, f1, precision, recall

class PersistenceModel(BaseModel):
    def __init__(self, config):
        super(PersistenceModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        print 'Building PersistenceModel'
        input_shape = (32, 32, 1)
        inputX = Input(shape = input_shape)
        x = Conv2D(128, (4, 4), strides=2, activation='relu')(inputX)
        x = BatchNormalization(name='batchnorm1')(x)
        x = Conv2D(128, (4, 4), strides=2, activation='relu')(x)
        x = BatchNormalization(name='batchnorm2')(x)
        x = GlobalAveragePooling2D(name='AvgPoolPer')(x)

        top_model = Model(inputs=inputX, outputs=x)
        top_model.name = 'persistenceTop'

        x = top_model(inputX)

        x = Dropout(self.config.trainer.dropout)(x)
        x = Dense(128, activation='relu', name='densePer')(x)
        x = Dropout(self.config.trainer.dropout)(x)
        x = Dense(2)(x)
        outputY = Activation('softmax')(x)


        self.model = Model(inputs=top_model.inputs, outputs=outputY)


        top_model.summary()
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
