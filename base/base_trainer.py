from keras.callbacks import ModelCheckpoint, TensorBoard
import os
import tensorflow as tf
import keras
from keras import backend as K
import numpy as np

class BaseTrainer(object):
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.args = dict( zip( self.config.tensorboard.keys(), self.config.tensorboard.values() ) )

    def train(self, train_generator, cv_generator):
        raise NotImplementedError



class TrainValTensorBoard(TensorBoard):
    def __init__(self, config, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'

        self.config = config
        training_log_dir = os.path.join(self.config.callbacks.tensorboard_log_dir, 'training')
        print training_log_dir
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(self.config.callbacks.tensorboard_log_dir, 'validation')
        print self.val_log_dir

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        print 'Inside epoch_end'
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        print 'val_logs = ', val_logs
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()



class TrainValTensorBoard_visualize(TensorBoard):
    def __init__(self, config, batch_gen, **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'

        self.config = config
        training_log_dir = os.path.join(self.config.callbacks.tensorboard_log_dir, 'training')
        print training_log_dir
        super(TrainValTensorBoard_visualize, self).__init__(training_log_dir, **kwargs)
        self.batch_gen = batch_gen # The generator.

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(self.config.callbacks.tensorboard_log_dir, 'validation')
        print self.val_log_dir

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard_visualize, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        print 'Inside epoch_end'
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        print 'val_logs = ', val_logs
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}


        #https://github.com/keras-team/keras/issues/3358#issuecomment-422826820


        if self.config.exp.name == 'Persistence':
            len_CV = self.config.len_CV
            imgs = []
            tags = []

            total_batches = len_CV // self.config.trainer.batch_size

            for batch_num in range(total_batches):
                img, label = self.batch_gen[batch_num]
                for i in range(img.shape[0]):
                    imgs.append(img[i])
                    tags.append(label[i])
            imgs = np.array(imgs)
            tags = np.array(tags)

            print imgs.shape
            print tags.shape

            self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]


        return super(TrainValTensorBoard_visualize, self).on_epoch_end(epoch, logs)


    def on_train_end(self, logs=None):
        super(TrainValTensorBoard_visualize, self).on_train_end(logs)
        self.val_writer.close()





def focal_loss(config):


    gamma = config.loss.gamma
    alpha = config.loss.alpha

    def Focal_Loss(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))


    def Focal_loss_fixed(y_true, y_pred):
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)

        alpha_factor = keras.backend.ones_like(y_true) * alpha
        alpha_factor = tf.where(keras.backend.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
        return  K.sum(cls_loss)

    return Focal_loss_fixed
