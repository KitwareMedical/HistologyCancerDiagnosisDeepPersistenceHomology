from base.base_trainer import BaseTrainer, TrainValTensorBoard
import os, time
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.applications.resnet50 import preprocess_input as preprocess_resnet
from utils.KerasOneCycle.clr import LRFinder, OneCycleLR
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense


class ResNetCombinedTrainer(BaseTrainer):
    def __init__(self, model, config):
        super(ResNetCombinedTrainer, self).__init__(model, config)

        self.callbacks = []
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

        self.init_callbacks()

    def init_callbacks(self):

        if not os.path.isdir(self.config.checkpoint_dir):
            os.mkdir(self.config.checkpoint_dir)
            print 'Created checkpoint directory : %s' % os.path.abspath(self.config.checkpoint_dir)

        self.callbacks.append(
            ModelCheckpoint(
                os.path.join(self.config.checkpoint_dir, '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose
            )
        )

        self.callbacks.append(
            TrainValTensorBoard(self.config, **self.args)
        )


        if self.config.trainer.optimizer == 'sgd':
            print '-'*50
            print 'SGD optimizer : Using OneCycleLR'
            print '-'*50
            self.callbacks.append(
                OneCycleLR(
                    num_samples=self.config.len_train,
                    num_epochs=self.config.trainer.num_epochs,
                    batch_size=self.config.trainer.batch_size,
                    max_lr=self.config.LRFinder.learning_rate
                )
            )
        else:
            print '-'*50
            print 'Adam optimizer : Using ReduceLROnPlateau'
            print '-'*50

            self.callbacks.append(
               ReduceLROnPlateau(
                    monitor = self.config.LROnPlateau.monitor,
                    factor = self.config.LROnPlateau.factor,
                    patience = self.config.LROnPlateau.patience,
                    verbose = self.config.LROnPlateau.verbose,
                    mode = self.config.LROnPlateau.mode,
                    min_delta = self.config.LROnPlateau.min_delta,
                    cooldown = self.config.LROnPlateau.cooldown,
                    min_lr = self.config.LROnPlateau.min_lr
                    )
                )

    def trainModel(self, TrainGenerator, CVGenerator):


        print 'Fitting Generator'
        len_train = len(TrainGenerator.train_files)
        len_CV = len(CVGenerator.cv_files)
        history = self.model.fit_generator(
                  generator = TrainGenerator,
                  steps_per_epoch=int(len_train/self.config.trainer.batch_size),
                  epochs=self.config.trainer.num_epochs,
                  verbose=self.config.trainer.verbose_training,
                  validation_data= CVGenerator,
                  validation_steps=int(len_CV/self.config.trainer.batch_size),
                  class_weight=self.config.class_weights,
                  workers=self.config.fit_generator.workers,
                  use_multiprocessing=self.config.fit_generator.use_multiprocessing,
                  max_queue_size=self.config.fit_generator.max_queue_size,
                  callbacks=self.callbacks
	    )
        self.train_loss.extend(history.history['loss'])
        self.train_acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])




    def trainLRFinder(self, TrainGenerator, CVGenerator):
        print 'Training for one epoch only'
        lr_callback = LRFinder(
                self.config.len_train,
                self.config.trainer.batch_size,
                self.config.LRFinder.minimum_lr,
                self.config.LRFinder.maximum_lr,
                lr_scale='exp',
                save_dir=self.config.LRFinder.save_dir
                )

        history = self.model.fit_generator(
                  generator = TrainGenerator,
                  steps_per_epoch=int(self.config.len_train/self.config.trainer.batch_size),
                  epochs=1,
                  verbose=self.config.trainer.verbose_training,
                  class_weight=self.config.class_weights,
                  callbacks=[lr_callback],
                  workers=self.config.fit_generator.workers,
                  use_multiprocessing=self.config.fit_generator.use_multiprocessing,
                  max_queue_size=self.config.fit_generator.max_queue_size
                  )

        lr_callback.plot_schedule()


    def train(self, TrainGenerator, CVGenerator):
        print 'Training model...'

        def print_params():
            print '\nLearning Rate : ', self.config.LRFinder.learning_rate
            print 'Batch Size : ', self.config.trainer.batch_size
            print 'Epochs : ', self.config.trainer.num_epochs
            print 'Dropout : ', self.config.trainer.dropout
            print 'Workers : ', self.config.fit_generator.workers
            print 'use_multiprocessing : ', self.config.fit_generator.use_multiprocessing
            print 'max_queue_size : ', self.config.fit_generator.max_queue_size
            print 'Class weights : ', self.config.class_weights
            print '# Train Malignant : ', len(TrainGenerator.mal_paths_train)
            print '# Train Benign    : ', len(TrainGenerator.ben_paths_train)
            print '# CV Malignant    : ', len(CVGenerator.mal_paths_cv_per)
            print '# CV Benign       : ', len(CVGenerator.ben_paths_cv_per)
            if self.config.use_focal_loss:
                print 'Loss : Focal Loss'
                print '  gamma : ', self.config.loss.gamma
                print '  alpha : ', self.config.loss.alpha
            else:
                print 'Loss : ', 'categorical_crossentropy'
            print '\n'

        print_params()

        if self.config.findLR:
            self.trainLRFinder(TrainGenerator, CVGenerator)
        else:
            self.trainModel(TrainGenerator, CVGenerator)




def ResNetCombinedTest(model, data, config):
    # https://github.com/keras-team/keras/issues/6499#issuecomment-301562885
    X_RGB, X_Per, Y_test = data

    print 'RGB : ', X_RGB.shape
    print 'Per : ', X_Per.shape
    print 'Y : ', len(Y_test)

    metrics_vals = model.evaluate([X_RGB, X_Per] , to_categorical(Y_test, num_classes=2) , batch_size=config.trainer.batch_size)
    metrics_names = model.metrics_names
    metrics = dict(zip(metrics_names, metrics_vals))
    print metrics

    print '-'*60
    print 'Test Loss : ', metrics['loss']
    print 'Test Acc  : ', metrics['acc']

    Y_prob = model.predict([X_RGB, X_Per])

    return Y_prob, metrics['loss'], metrics['acc']
