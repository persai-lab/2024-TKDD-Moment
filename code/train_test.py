import numpy as np
import mxnet as mx
from sklearn import metrics
import logging
import os
import pandas as pd
logging.getLogger().setLevel(logging.INFO)
from datetime import datetime
def norm_clipping(params_grad, threshold):
    norm_val = 0.0
    for i in range(len(params_grad[0])):
        norm_val += np.sqrt(
            sum([mx.nd.norm(grads[i]).asnumpy()[0] ** 2
                 for grads in params_grad]))
    norm_val /= float(len(params_grad[0]))

    if norm_val > threshold:
        ratio = threshold / float(norm_val)
        for grads in params_grad:
            for grad in grads:
                grad[:] *= ratio

    return norm_val

def train_test_run(net,data_iter,pars,train_test,print_results = True):
    all_type_preds = []
    all_quiz_preds = []
    all_paddings = []
    all_type_labels = []
    all_quiz_labels = []
    all_time_labels = []
    all_time_preds = []

    print(f'======================================= start {train_test} ============================================')
    # now = datetime.now()
    bi = 1

    for batch in data_iter:
        f = True if train_test == 'training' else False
        net.forward(batch, is_train=f)
        # print(net.get_input_grads()[0].asnumpy())
        preds = [x.asnumpy() for x in net.get_outputs()]
        quiz_pred,type_pred,time_f = preds
        quiz_pred = quiz_pred.reshape(-1,)
        type_pred = type_pred.reshape(pars.batch_size*pars.sequence_length,-1)
        quiz_pred = quiz_pred.reshape(-1,)

        net.backward()
        norm_clipping(net._exec_group.grad_arrays, 500)
        net.update()
        # quiz_pred = np.argmax(quiz_pred, axis=1)
        # true_type = batch.label[1].reshape(-1,)
        # true_quiz = batch.label[2].reshape(-1,)
        # true_time = batch.label[0].reshape(-1,)
        true_type = batch.label[1].reshape(-1,)
        true_quiz = batch.label[2].reshape(-1,)
        true_time = batch.label[0].reshape(-1,)
        padd = batch.data[0].reshape(-1,)
        valid_idx = np.flatnonzero(padd!= 0)

        # if pars.dataset == 'TK4_subset':
        #     valid_type_pred = type_pred[valid_idx]
        #     valid_type_label = true_type[valid_idx].asnumpy()
        #     valid_type_pred = np.argmax(valid_type_pred, axis=1)
        #     valid_quiz_label = np.array([0 if x==-1 else x for x in true_quiz.asnumpy()])
        #     valid_quiz_pred = np.array(np.array([1 if x>=pars.t else 0 for x in quiz_pred]))
        # # elif pars.dataset == 'Junyi_Academy':
        # else:
        valid_idx = np.flatnonzero(padd != 0)
        valid_type_pred = type_pred[valid_idx]
        valid_type_label = true_type[valid_idx].asnumpy()
        valid_type_pred = np.argmax(valid_type_pred, axis=1)
        valid_quiz_label = np.array(true_quiz[valid_idx].asnumpy())
        valid_quiz_pred = np.array(np.array([1 if x >= pars.t else 0 for x in quiz_pred[valid_idx]]))


        valid_time_pred = mx.nd.exp(mx.nd.array(time_f))
        valid_time_pred = mx.nd.Activation(data=valid_time_pred,act_type='tanh')
        valid_time_pred = mx.nd.Activation(data=valid_time_pred,act_type='relu')
        time_rmse = np.sqrt(metrics.mean_squared_error(np.array(true_time.asnumpy()),
                                                       np.array(valid_time_pred.asnumpy().reshape(-1))))
        time_mae = metrics.mean_absolute_error(np.array(true_time.asnumpy()), np.array(valid_time_pred.asnumpy().reshape(-1)))
        batch_quiz_acc = metrics.accuracy_score(np.array(valid_quiz_pred), np.array(valid_quiz_label))
        # batch_type_acc = metrics.accuracy_score(np.array(np.argmax(type_pred, axis=1)), np.array(true_type.asnumpy()))
        batch_type_acc = metrics.accuracy_score(np.array(valid_type_label),
                                                np.array(valid_type_pred))

        print(f'--------------------------------------- Batch {bi} --------------------------------------\n'
              f'type accuracy:{round(batch_type_acc,3)}, '
              f'result accuracy: {round(batch_quiz_acc,3)}, '
              f'time rmse:{str(round(time_rmse,3))}, '
              f'time mae:{str(round(time_mae,3))}',
              # f'loss:{mx.metric.Loss()}'
              )

        all_paddings+=list(padd.asnumpy())
        # all_time_preds+=list(time_pred.reshape(-1,))
        all_type_preds+=list(valid_type_pred)
        all_quiz_preds+=list(valid_quiz_pred)
        all_quiz_labels+=list(valid_quiz_label)
        all_type_labels+=list(valid_type_label)
        all_time_labels+=list(true_time.asnumpy())
        all_time_preds+=list(valid_time_pred.asnumpy().reshape(-1))
        bi += 1
    epoch_type_acc = metrics.accuracy_score(np.array(all_type_labels), np.array(all_type_preds))
    epoch_quiz_acc = metrics.accuracy_score(np.array(all_quiz_labels), np.array(all_quiz_preds))
    epoch_rmse_time = np.sqrt(metrics.mean_squared_error(np.array(all_time_preds), np.array(all_time_labels)))
    epoch_mae_time = metrics.mean_absolute_error(np.array(all_time_preds), np.array(all_time_labels))

    return epoch_type_acc,epoch_quiz_acc,epoch_rmse_time,epoch_mae_time


def find_best_epoch(net,train_iter,validate_iter,pars,metric = 'loss'):
    metric_map = {'type accuracy':0, 'result accuracy':1,
                  'time_rmse':2,'time_mae':3,'loss':4}
    row_idx = metric_map.get(metric)
    if not os.path.isdir('model'):
        os.makedirs('model')
    if not os.path.isdir(os.path.join('model',pars.save_path)):
        os.makedirs(os.path.join('model',pars.save_path))

    if not os.path.isdir('result'):
        os.makedirs('result')
    if not os.path.isdir(os.path.join('result',pars.save_path)):
        os.makedirs(os.path.join('result',pars.save_path))
    for epoch in range(pars.epoch):
        print(f'epoch {epoch}\n')
        train_iter.reset()
        train_epoch_type_acc,train_epoch_quiz_acc,train_epoch_rmse_time,train_epoch_mae_time\
            =train_test_run(net,train_iter,pars,train_test = 'training',print_results = True)

        print(f'train_epoch_type_acc: {train_epoch_type_acc}\n')
        print(f'train_epoch_quiz_acc: {train_epoch_quiz_acc}\n')
        print(f'train_epoch_rmse_time: {train_epoch_rmse_time}\n')
        print(f'train_epoch_mae_time: {train_epoch_mae_time}\n')
        validate_iter.reset()
        valid_epoch_type_acc,valid_epoch_quiz_acc,valid_epoch_rmse_time,valid_epoch_mae_time\
            = train_test_run(net, validate_iter, pars,train_test='validating', print_results=True)


        print(f'valid_epoch_type_acc: {valid_epoch_type_acc}\n')
        print(f'valid_epoch_quiz_acc: {valid_epoch_quiz_acc}\n')
        print(f'valid_epoch_rmse_time: {valid_epoch_rmse_time}\n')
        print(f'valid_epoch_mae_time: {valid_epoch_mae_time}\n')


    # metric_array = validate_results[row_idx]
    # best_epoch = 1 + metric_array.argmax()
    # net.save_checkpoint(prefix=os.path.join('model', f'{pars.save_path}/seqlen:{pars.sequence_length}'), epoch=best_epoch)



    # train_results = pd.DataFrame(train_results).T
    # train_results.columns = list(metric_map.keys())
    # train_results.to_csv(os.path.join('result',f'{pars.save_path}',f'seqlen:{pars.sequence_length}_training_result_new.csv'))
    #
    # validate_results = pd.DataFrame(validate_results).T
    # validate_results.columns =  list(metric_map.keys())
    # pd.DataFrame(validate_results).to_csv(os.path.join('result',f'{pars.save_path}',f'seqlen:{pars.sequence_length}_validating_result_new.csv'))

    return 1