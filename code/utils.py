import mxnet as mx
import numpy as np

import os
from sklearn.preprocessing import label_binarize
from sklearn import metrics


def type_auc(label, softmax):
    y_true = label.reshape(-1, )
    valid_idx = np.flatnonzero(y_true != -1)
    num_classes = softmax.shape[1]
    if num_classes == 2:
        try:
            auc = metrics.roc_auc_score(mx.nd.one_hot(mx.nd.array(y_true), 2).asnumpy()[valid_idx], softmax[valid_idx])
        except ValueError as error:
            auc = 0
    else:
        y = label_binarize(np.array(y_true), classes=range(num_classes))
        try:
            auc = metrics.roc_auc_score(y[valid_idx], softmax[valid_idx], 'macro', multi_class='ovo')
        except ValueError as error:
            auc = 0
    return auc


def custom_metric_type_acc(label, pred):
    y_pred = np.argmax(pred, 1).reshape(-1, )
    y_true = label.reshape(-1, )
    valid_idx = np.flatnonzero(y_true != -1)
    return metrics.accuracy_score(y_true[valid_idx], y_pred[valid_idx])
def load_params(prefix, epoch):
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params


def changeArr(input1):
    # Copy input array into newArray
    newArray = input1.copy()

    # Sort newArray[] in ascending order
    newArray.sort()

    # Dictionary to store the rank of
    # the array element
    ranks = {}

    rank = 1

    for index in range(len(newArray)):
        element = newArray[index]

        # Update rank of element
        if element not in ranks:
            ranks[element] = rank
            rank += 1

    # Assign ranks to elements
    for index in range(len(input1)):
        element = input1[index]
        input1[index] = ranks[input1[index]]
    return input1

def tag2bow(tag_array):
    tags_list = []
    tags_num = []
    for i in range(len(tag_array)):
        for j in range(tag_array.shape[1]):
            tag_ij = tag_array[i, j]
            try:
                temp = [int(float(xx)) for xx in tag_ij.split(';')]
                tags_list.append(temp)
                tags_num.append(max(temp))

            except:
                tags_list.append(-1)
    tags_num = max(tags_num) + 1
    tags_embedding = []
    for t in tags_list:
        embed_ij = np.zeros(tags_num)
        if t != -1:
            # print(t)
            embed_ij[t] = 1
        tags_embedding.append(embed_ij)

    tags_embedding = np.array(tags_embedding).reshape(tag_array.shape[0], tag_array.shape[1], tags_num)

    return tags_embedding,tags_num

def set_paths(pars,dt_string):

    for dir in ['model','results']:

        if not os.path.isdir(os.path.join('../',dir)):
            os.makedirs(os.path.join('../',dir))
        if not os.path.isdir(os.path.join('../',dir, pars.save_path)):
            os.makedirs(os.path.join('../',dir, pars.save_path))

        if not os.path.isdir(os.path.join('../',dir, pars.save_path, pars.ablation_type)):
            os.makedirs(os.path.join('../',dir, pars.save_path, pars.ablation_type))

        if pars.save_train_log:
            # if not os.path.isdir(os.path.join('../',dir, pars.save_path, pars.ablation_type, dt_string)):
            #         os.makedirs(os.path.join('../',dir, pars.save_path, pars.ablation_type, dt_string))

            if pars.use_nongraded == False:
                if not os.path.isdir(os.path.join('../',dir, pars.save_path, pars.ablation_type, 'nongraded', dt_string)):
                    os.makedirs(os.path.join('../',dir, pars.save_path,  pars.ablation_type, 'nongraded', dt_string))
            else:
                if not os.path.isdir(os.path.join('../',dir, pars.save_path, pars.ablation_type, 'fulldata', dt_string)):
                    os.makedirs(os.path.join('../',dir, pars.save_path, pars.ablation_type, 'fulldata', dt_string))

    return


def plot_fig(mv,params,batch,pars,tags_num):
    tags = batch.data[-3]
    quiz_input = batch.data[2]

    read_contac_W = params.get('read_concat_weight')
    read_contac_b = params.get('read_concat_bias')
    result_fc_W = params.get('result_fc_weight')
    result_fc_b= params.get('result_fc_bias')
    quiz_pred_W= params.get('quiz_pred_weight')
    quiz_pred_b= params.get('quiz_pred_bias')
    Concept_pred = []
    Mv = mx.nd.array(mv.reshape(pars.batch_size,pars.sequence_length,pars.dv,pars.memory_size))
    for i in range(pars.sequence_length):
        quiz_embedding = mx.nd.one_hot(quiz_input[:, i], pars.quiz_embedding_size)
        tags_embedding = mx.nd.one_hot(tags[:,i], tags_num)
        quiz_embedding = mx.nd.Concat(quiz_embedding, tags_embedding, dim=1)
        mv = Mv[:,i,:,:]
        mv = mx.nd.transpose(mv,[0,2,1])
        cor = mx.nd.array([np.eye(pars.memory_size)]*pars.batch_size)
        # read_content = mx.nd.batch_dot(cor,mv)
        concept_pred = []
        for j in range(pars.memory_size):
            corj = cor[:,j]
            corj = mx.nd.expand_dims(corj,1)
            read_content = mx.nd.batch_dot(corj, mv)
            read_content = mx.nd.squeeze(read_content,axis = 1)

            knowledge_quiz_concat = mx.nd.Concat(read_content, quiz_embedding, dim=1,
                                                 name='knowledge quiz concatenated')

            knowledge_quiz_embedding = mx.nd.FullyConnected(data=knowledge_quiz_concat,
                                                         weight=read_contac_W,
                                                         bias=read_contac_b,
                                                         num_hidden=pars.knowledge_hidden,
                                                         name='knowledge quiz embedding')

            knowledge_quiz_embedding = mx.nd.Activation(data=knowledge_quiz_embedding,
                                                         act_type='relu',
                                                         name='knowledge quiz embedding tanh')

            quiz_response = mx.nd.FullyConnected(data=knowledge_quiz_embedding, num_hidden=pars.quiz_hidden,
                                                  weight=result_fc_W,
                                                  bias=result_fc_b,
                                                  name='quiz prediction')

            quiz_pred = mx.nd.FullyConnected(data=quiz_response, num_hidden=2,
                                              weight=quiz_pred_W,
                                              bias=quiz_pred_b,
                                              )

            quiz_pred = mx.nd.softmax(quiz_pred, name='quiz_pred')
            concept_pred.append(quiz_pred[:,0].asnumpy())
        Concept_pred.append(np.array(concept_pred))
    return Concept_pred