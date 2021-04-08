#!/usr/bin/python
#-*- coding: utf-8 -*-

import os
import glob
import sys
import time
from sklearn import metrics
import numpy as np
import pdb
from operator import itemgetter

import matplotlib.pyplot as plt
from matplotlib.offsetbox import (TextArea, AnnotationBbox)


def tuneThresholdfromScore(scores, labels, target_fa, target_fr = None):
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = [];
    if target_fr:
        for tfr in target_fr:
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    #?
    for tfa in target_fa:
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
    
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr)))
    eer  = max(fpr[idxE],fnr[idxE])*100
    
    return (tunedThreshold, eer, fpr, fnr,thresholds);

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
      sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
      sorted_labels = []
      labels = [labels[i] for i in sorted_indexes]
      fnrs = []
      fprs = []

      # At the end of this loop, fnrs[i] is the number of errors made by
      # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
      # is the total number of times that we have correctly accepted scores
      # greater than thresholds[i].
      for i in range(0, len(labels)):
          if i == 0:
              fnrs.append(labels[i])
              fprs.append(1 - labels[i])
          else:
              fnrs.append(fnrs[i-1] + labels[i])
              fprs.append(fprs[i-1] + 1 - labels[i])
      fnrs_norm = sum(labels)
      fprs_norm = len(labels) - fnrs_norm

      # Now divide by the total number of false negative errors to
      # obtain the false positive rates across all thresholds
      fnrs = [x / float(fnrs_norm) for x in fnrs]

      # Divide by the total number of corret positives to get the
      # true positive rate.  Subtract these quantities from 1 to
      # get the false positive rates.
      fprs = [1 - x / float(fprs_norm) for x in fprs]
      return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


google_label = np.load('google_command_label.npy')
google_score = np.load('google_command_score.npy')
#google_score = google_score[:,1:]

vox_label = np.load('voxceleb_label.npy')
vox_score = np.load('voxceleb_score.npy')
vox_score = vox_score.reshape(-1,1)

def metrics_eer(labels,scores,pos_label = [1],save_path = './results', target_fa = None, target_fr = None):
    '''
               real
              1    0
    pred 1   TP   FP
         0   FN   TN
    accuracy = (TP + TN)/(all)     
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1_score = 
    FPR = FP/(FP+TN) = FP/real negative
    FNR = FN/(FN+TP) = FN/real positive 
    TPR = 1 - FNR = TP/(TP + FN) = recall
    
    kws: false reject(FNR) y-axis: false reject: a keyword is not present but kws system is positive
         false alarm(FPR) x-axis: false alarm: a keyword is present but kws system is negative 
    
    '''        
    if not os.path.exists(save_path):
        os.mkdir(save_path)
            
    file = open(save_path + '/results.txt','w')
    
    if scores.shape[1] == 1:
        fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=pos_label[0])
        fnr = 1 - tpr    
        idxE = np.nanargmin(np.absolute((fnr - fpr)))
        eer  = max(fpr[idxE],fnr[idxE])*100
        
        tunedThreshold = [];
        
        if target_fa:
            for tfr in target_fr:
                idx = np.nanargmin(np.absolute((tfr - fnr)))
                tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);
        #?
        elif target_fr:
            for tfa in target_fa:
                idx = np.nanargmin(np.absolute((tfa - fpr))) # numpy.where(fpr<=tfa)[0][-1]
                tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]]);

        #
        fig, ax = plt.subplots(1, 1,figsize = (5,5))
        ax.plot(thresholds,fpr,label = 'false positive rate')
        ax.plot(thresholds,fnr,label = 'false negative rate')
        ax.set_xlabel('thresholds')
        
        offsetbox = TextArea('eer:%.3f' % eer + '%')
        xy = (thresholds[idxE],eer/100)
        
        ab = AnnotationBbox(offsetbox, xy,
                            xybox=(60, 40),
                            xycoords='data',
                            boxcoords="offset points",
                            arrowprops=dict(arrowstyle="->"))
        ax.add_artist(ab)
        ax.legend()
        fig.savefig(save_path + '/eer.png')
        
        fig, ax = plt.subplots(1, 1,figsize = (5,5))
        ax.plot(fpr,fnr)      
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('false negative rate')        
        fig.savefig(save_path + '/fpr-fnr.png')
        plt.show()       
        
 

    elif scores.shape[1] > 1:

        fig, ax = plt.subplots(1, 1,figsize = (8,8))    
        
        preds = np.argmax(scores,axis = 1)
    
        for i in range(scores.shape[1]):
            if i > 0:
                score = scores[:,i]
                fpr, tpr, thresholds = metrics.roc_curve(labels, score, pos_label=pos_label[i])
                fnr = 1 - tpr    
                idxE = np.nanargmin(np.absolute((fnr - fpr)))
                eer  = max(fpr[idxE],fnr[idxE])*100
                
                pred = preds == pos_label[i]
                label = labels == pos_label[i]
                
                precision = metrics.precision_score(label,pred)
                recall = metrics.recall_score(label,pred)
                f1_score = metrics.f1_score(label,pred)
                
                line =  "label %d"%pos_label[i] + ' : precision:%.5f' % precision\
                        + ' : recall:%.5f' % recall \
                        + ' : f1 score:%.5f' % f1_score + '\n'
                                
                file.write(line)
                
                            
                label = "label %d"%pos_label[i] + ' : eer:%.3f' % eer + '%'
                ax.plot(fpr,fnr,label = label)
        
        
        macro_score = metrics.f1_score(labels,preds,average = 'macro')
        micro_score = metrics.f1_score(labels,preds,average = 'micro')
 
        line =  "macro f1 score: %.5f "% macro_score + '\n' + \
                 "micro f1 score: %.5f "% micro_score + '\n'
          
                        
        file.write(line)
        
        ax.set_xlabel('false positive rate')
        ax.set_ylabel('false negative rate')        
        fig.savefig(save_path + '/fpr-fnr_muti.png')
        ax.legend()
        plt.show()    
            
    
        
    else:
        print('scores shape error!')
    

    file.close()   

#demo
metrics_eer(vox_label,vox_score)
metrics_eer(google_label,google_score,pos_label = [0,1,2,3,4,5,6,7,8,9,10,11])






