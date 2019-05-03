
from base.base_model import BaseModel
from base.base_trainer import focal_loss
import keras
from keras.layers import Input, Dense, Conv2D, Activation, Dropout, GlobalAveragePooling2D, concatenate, BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.utils import multi_gpu_model
import os

from utils.metrics import auc, f1, precision, recall

class ResNetCombinedModel(BaseModel):
    def __init__(self, config):
        super(ResNetCombinedModel, self).__init__(config)
        self.input_shape_RGB = (256, 256, 3)
        self.input_shape_Per = (32, 32, 1)
        self.blocks = ['input_1', 'add_1', 'add_2', 'add_3', 'add_4', 'add_5', 'add_6', 'add_7', 'add_8', \
                       'add_9', 'add_10', 'add_11', 'add_12', 'add_13', 'add_14', 'add_15', 'add_16']
        self.block_ids = [0, 16, 26, 36, 48, 58, 68, 78, 90, 100, 110, 120, 130, 140, 152, 162, 172]
        self.build_model()

    def build_model(self):
        print 'Building ResNetCombinedModel'

        inputRGB = Input(shape = self.input_shape_RGB)
        inputPer = Input(shape = self.input_shape_Per)

        print '-'*50
        print 'RGB initialization : ', self.config.model.weights_rgb
        print '-'*50
        if self.config.model.weights_rgb == "None":
            self.config.model.weights_rgb = None
        resnetRGB = keras.applications.ResNet50(weights=self.config.model.weights_rgb, include_top=False, pooling='avg', input_shape=self.input_shape_RGB)
        resnetRGB.name = 'resnetRGB'
        xRGB = resnetRGB(inputRGB)
        xRGB = Dropout(self.config.trainer.dropout)(xRGB)
        xRGB = Dense(128, activation='relu', name='denseRGB')(xRGB)


        xPer = Conv2D(128, (4, 4), strides=2, activation='relu')(inputPer)
        xPer = BatchNormalization()(xPer)
        xPer = Conv2D(128, (4, 4), strides=2, activation='relu')(xPer)
        xPer = BatchNormalization()(xPer)
        xPer = GlobalAveragePooling2D(name='AvgPoolPer')(xPer)
        
        top_model = Model(inputs=inputPer, outputs=xPer)
        top_model.name = 'persistenceTop'
        xPer = top_model(inputPer)
        xPer = Dropout(self.config.trainer.dropout)(xPer)
        xPer = Dense(128, activation='relu', name='densePer')(xPer)

        concat = keras.layers.concatenate([xRGB, xPer], axis=1)
        concat = Dropout(self.config.trainer.dropout)(concat)

        output = Dense(2)(concat)
        output = Activation('softmax')(output)
        self.model = Model(inputs=[inputRGB, inputPer], outputs=output)


        for block_num in range(self.config.model.freeze_RGB):
            block_start = self.block_ids[block_num]
            block_end = self.block_ids[block_num+1]
            for i in range(block_start, block_end):
                self.model.get_layer('resnetRGB').layers[i].trainable = False


        if self.config.model.freeze_Persistence == True:
            for layer in self.model.get_layer('persistenceTop').layers:
                layer.trainable = False


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
            self.config.checkpoint_RGB = ''
            self.config.checkpoint_Persistence = ''
        else:
            print 'No checkpoint found for combined model'


        if self.config.checkpoint_RGB != '' and os.path.isfile(self.config.checkpoint_RGB):
            print 'Loading RGB weights from: %s' % (self.config.checkpoint_RGB)
            inputX = Input(shape = self.input_shape_RGB)
            resnet = keras.applications.ResNet50(weights='imagenet', include_top=False, pooling='avg', input_shape=self.input_shape_RGB)
            x = resnet(inputX)
            x = Dropout(self.config.trainer.dropout)(x)
            x = Dense(128, activation='relu', name='denseRGB')(x)
            x = Dropout(self.config.trainer.dropout)(x)
            x = Dense(2)(x)
            outputY = Activation('softmax')(x)
            dummy_model_rgb = Model(inputs=inputX, outputs=outputY)

            dummy_model_rgb.load_weights(self.config.checkpoint_RGB)

            weights_list = dummy_model_rgb.get_layer('resnet50').get_weights()
            self.model.get_layer('resnetRGB').set_weights(weights_list)
            self.model.get_layer('denseRGB').set_weights( dummy_model_rgb.get_layer('denseRGB').get_weights() )
            del dummy_model_rgb
            print 'Successfully loaded RGB layer weights in combined model'


        if self.config.checkpoint_Persistence != '' and os.path.isfile(self.config.checkpoint_Persistence):
            print 'Loading Persistence weights from: %s' % (self.config.checkpoint_Persistence)
            inputX = Input(shape = self.input_shape_Per)
            x = Conv2D(128, (4, 4), strides=2, activation='relu')(inputX)
            x = BatchNormalization()(x)
            x = Conv2D(128, (4, 4), strides=2, activation='relu')(x)
            x = BatchNormalization()(x)
            x = GlobalAveragePooling2D(name='AvgPool')(x)

            top_model = Model(inputs=inputX, outputs=x)
            top_model.name='persistenceTop'

            x = top_model(inputX)

            x = Dropout(self.config.trainer.dropout)(x)
            x = Dense(128, activation='relu', name='densePer')(x)
            x = Dropout(self.config.trainer.dropout)(x)
            x = Dense(2)(x)
            outputY = Activation('softmax')(x)
            dummy_model_persistence = Model(inputs=inputX, outputs=outputY)

            dummy_model_persistence.load_weights(self.config.checkpoint_Persistence)

            weights_list = dummy_model_persistence.get_layer('persistenceTop').get_weights()
            self.model.get_layer('persistenceTop').set_weights(weights_list)
            self.model.get_layer('densePer').set_weights( dummy_model_persistence.get_layer('densePer').get_weights() )
            del dummy_model_persistence
            print 'Successfully loaded Persistence layer weights in combined model'
