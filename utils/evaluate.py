import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from data_loader.rgb_data_loader import RGBTestData
from models.resnetRGB_model import ResNetRGBModel
from trainer.resnetRGB_trainer import ResNetRGBTest

from data_loader.persistence_data_loader import PersistenceTestData
from models.Persistence_model import PersistenceModel
from trainer.Persistence_trainer import PersistenceTest

from data_loader.combined_data_loader import CombinedTestData
from models.resnetCombined_model import ResNetCombinedModel
from trainer.resnetCombined_trainer import ResNetCombinedTest

import numpy as np
from sklearn import metrics
import os
import cPickle as pickle



def compute_metrics(Y_pred, Y_test, save_path, score, acc, display_result=False):

    pred_classes = np.argmax(Y_pred, axis=1)

    Y_score = Y_pred.T[1]

    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_score, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    print 'AUC       : ', roc_auc

    pscore = metrics.precision_score(Y_test, pred_classes, average='binary')
    print 'Precision : ', pscore

    recall_score = metrics.recall_score(Y_test, pred_classes, average='binary')
    print 'Recall    : ', recall_score

    f1 = metrics.f1_score(Y_test, pred_classes, average='binary')
    print 'F1 score  : ', f1

    cm = metrics.confusion_matrix(Y_test, pred_classes)
    print 'Confusion Matrix : '
    print cm

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig( os.path.join(save_path, 'ROC.png') )
    #print 'Figure saved!'

    if display_result:
        plt.show()

    model_metrics = {}
    model_metrics['AUC'] = roc_auc
    model_metrics['Precision'] = pscore
    model_metrics['Recall'] = recall_score
    model_metrics['f1_score'] = f1
    model_metrics['test_loss'] = score
    model_metrics['test_acc'] = acc
    model_metrics['confusion_matrix'] = cm

    with open( os.path.join(save_path, 'metrics.pkl'), 'wb' ) as f:
        pickle.dump(model_metrics, f)

    #print 'Metrics saved!'
    print '-'*60
    print(metrics.classification_report(Y_test, pred_classes, labels=[0, 1], target_names=['Benign', 'Malignant']))
    print '-'*60

def test_combined(config):

    print 'Loading Combined test data'
    data = CombinedTestData(config)
    Y_test = data[2]

    print 'Building model'
    model = ResNetCombinedModel(config).model

    print 'Predicting'
    Y_pred, score, acc = ResNetCombinedTest(model, data, config)

    compute_metrics(Y_pred, Y_test, config.config_dir, score, acc)



def test_persistence(config):

    print 'Loading Persistence test data...'
    X_test, Y_test = PersistenceTestData(config)
    print X_test.shape

    print 'Building model'
    model = PersistenceModel(config).model
    #model = ResNetPersistenceModel(config).model

    print 'Predicting'
    Y_prob, score, acc = PersistenceTest(model, [X_test, Y_test], config)

    compute_metrics(Y_prob, Y_test, config.config_dir, score, acc)



def test_rgb(config):

    print 'Loading RGB Test data...'
    X_test, Y_test = RGBTestData(config)
    print X_test.shape
    print len(Y_test)

    print 'Building model'
    model = ResNetRGBModel(config).model

    print 'Predicting'
    Y_pred, score, acc = ResNetRGBTest(model, [X_test, Y_test], config)

    compute_metrics(Y_pred, Y_test, config.config_dir, score, acc)
