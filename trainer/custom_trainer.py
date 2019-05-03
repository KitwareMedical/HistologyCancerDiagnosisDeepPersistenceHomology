from base.base_trainer import BaseTrainer, TrainValTensorBoard
import os, time
from keras.callbacks import ModelCheckpoint, TensorBoard
from utils.KerasOneCycle.clr import LRFinder, OneCycleLR
import tensorflow as tf
from keras.utils import to_categorical



class CustomTrainer(BaseTrainer):
    def __init__(self, model, config):
        super(CustomTrainer, self).__init__(model, config)

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
            TrainValTensorBoard(self.config, write_graph=False)
        )

        if self.config.trainer.optimizer == 'sgd':
            self.callbacks.append(
                OneCycleLR(
                    num_samples=self.config.len_train,
                    num_epochs=self.config.trainer.num_epochs,
                    batch_size=self.config.trainer.batch_size,
                    max_lr=self.config.LRFinder.learning_rate
                )
            )
            print '-'*50
            print 'SGD optimizer : Using OneCycleLR'
            print '-'*50
        else:
            print '-'*50
            print 'Adam optimizer : Not using OneCycleLR'
            print '-'*50



    def trainModel(self, TrainGenerator, CVGenerator):

        print 'Fitting Generator'
        history = self.model.fit_generator(
                  generator = TrainGenerator,
                  epochs=self.config.trainer.num_epochs,
                  steps_per_epoch = int(self.config.len_train/self.config.trainer.batch_size),
                  verbose=self.config.trainer.verbose_training,
                  validation_data=CVGenerator,
                  validation_steps=int(self.config.len_CV/self.config.trainer.batch_size),
                  class_weight=self.config.class_weights,
                  callbacks=self.callbacks,
                  workers=self.config.fit_generator.workers,
                  use_multiprocessing=self.config.fit_generator.use_multiprocessing,
                  max_queue_size=self.config.fit_generator.max_queue_size
        )
        self.train_loss.extend(history.history['loss'])
        self.train_acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])



    def trainLRFinder(self, TrainGenerator, CVGenerator, val_data=None):
        print 'Training for one epoch only'
        if val_data == None:
            lr_callback = LRFinder(
                    self.config.len_train,
                    self.config.trainer.batch_size,
                    self.config.LRFinder.minimum_lr,
                    self.config.LRFinder.maximum_lr,
                    lr_scale='exp',
                    save_dir=self.config.LRFinder.save_dir
                    )
        else:
            X_CV, Y_CV = val_data
            print X_CV.shape
            print Y_CV.shape
            lr_callback = LRFinder(
                    self.config.len_train,
                    self.config.trainer.batch_size,
                    self.config.LRFinder.minimum_lr,
                    self.config.LRFinder.maximum_lr,
                    validation_data = (X_CV, Y_CV),
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




    def train(self, TrainGenerator, CVGenerator, val_data=None):
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
            if self.config.use_focal_loss:
                print 'Loss : Focal Loss'
                print '  gamma : ', self.config.loss.gamma
                print '  alpha : ', self.config.loss.alpha
            else:
                print 'Loss : ', 'categorical_crossentropy'
            print '\n'

        print_params()

        if self.config.findLR:
            self.trainLRFinder(TrainGenerator, CVGenerator, val_data)
        else:
            self.trainModel(TrainGenerator, CVGenerator)
