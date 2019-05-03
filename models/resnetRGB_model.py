from base.base_model import BaseModel
from base.base_trainer import focal_loss
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Activation, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
import keras
from keras.utils import multi_gpu_model

from utils.metrics import auc, f1, precision, recall

class ResNetRGBModel(BaseModel):
    def __init__(self, config):
        super(ResNetRGBModel, self).__init__(config)
        self.blocks = ['input_1', 'add_1', 'add_2', 'add_3', 'add_4', 'add_5', 'add_6', 'add_7', 'add_8', \
                       'add_9', 'add_10', 'add_11', 'add_12', 'add_13', 'add_14', 'add_15', 'add_16']
        self.block_ids = [0, 16, 26, 36, 48, 58, 68, 78, 90, 100, 110, 120, 130, 140, 152, 162, 172]

        self.build_model()

    def build_model(self):
        print 'Building model'
        input_shape = (256, 256, 3)
        inputX = Input(shape = input_shape)
        print 'ResNet initialization : ', self.config.trainer.weights
        if self.config.trainer.weights == 'None':
            self.config.trainer.weights = None
        resnet = keras.applications.ResNet50(weights=self.config.trainer.weights, include_top=False, pooling='avg', input_shape=input_shape)
        x = resnet(inputX)
        x = Dropout(self.config.trainer.dropout)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(self.config.trainer.dropout)(x)
        outputY = Dense(2, activation='softmax')(x)
        self.model = Model(inputs=inputX, outputs=outputY)



        for block_num in range(self.config.model.freeze_RGB):
            block_start = self.block_ids[block_num]
            block_end = self.block_ids[block_num+1]
            for i in range(block_start, block_end):
                self.model.get_layer('resnet50').layers[i].trainable = False


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
            print 'Successfully loaded weights fom file %s' % (self.config.load_checkpoint)
