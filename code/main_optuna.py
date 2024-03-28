import model_debug_new
import model
from utils import *
from train_test import *
import argparse
import logging
import warnings

import optuna
from train_test_split import *
warnings.filterwarnings("ignore")
from sklearn.metrics import *


def objective(trail):
    dataset = 'sample'
    parser = argparse.ArgumentParser(description='MoMENt')
    parser.add_argument('--dataset', type=str, default='sample', help='dataset')
    parser.add_argument('--epoch', type=int, help='number of epoches', default=300)
    parser.add_argument('--save_path', type=str, help='path of the saved model', default='TK4_subset')
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--quiz_embedding_size', type=int, default=10000, help='quiz embedding size')
    parser.add_argument('--result_embedding_size',type = int,default=trail.suggest_categorical('result_emdedding',[16,32,64,128,256,512]))
    parser.add_argument('--time_embedding_size',type = int, default=4)
    parser.add_argument('--num_hidden',type = int,default=trail.suggest_categorical('num_hidden',[16,32,64,128,256,512]),help='hidden layers')
    parser.add_argument('--type_embedding_size',type = int,default=trail.suggest_categorical('type_embedding_size',[1,4,8,16,32,64,128,256]))
    parser.add_argument('--event_f_hidden', type=int, default=16)
    parser.add_argument('--event_type_hidden', type=int, default=16)
    parser.add_argument('--knowledge_hidden',type = int, default=trail.suggest_categorical('knowledge_hidden',[16,32,64,128,256,512]))
    parser.add_argument('--quiz_hidden',type = int, default=trail.suggest_categorical('quiz_hidden',[16,32,64,128,256,512]))
    parser.add_argument('--memory_size',type = int,default=trail.suggest_int('memory_size',1,100),help = 'memory size')
    parser.add_argument('--quiz_has_tags', type=bool, default=True, help='if use quiz tags info')
    parser.add_argument('--dk',type = int, default=trail.suggest_categorical('dk',[16,32,64,128,256,512]),help = 'memory key matrix size')
    parser.add_argument('--dv',type = int, default=trail.suggest_categorical('dv',[16,32,64,128,256,512]),help = 'memory value matrix size')
    parser.add_argument('--event_embedding_size', type=int, help='event embedding size',
                        default=trail.suggest_categorical('event_embedding_size',[1,4,8,16,32,64]))
    parser.add_argument('--init_lr', type=float, default=0.001, help='iniital learning rate')
    parser.add_argument('--t', type=float, default=0.5)
    parser.add_argument('--time_unit', type=int, default=1000)
    parser.add_argument('--tag_onehot', type=bool, default=True)
    parser.add_argument('--quiz_weights', type=list, default=[60, 40])
    parser.add_argument('--save_train_log', type=bool, default=False)
    parser.add_argument('--use_nongraded', type=bool, default=True)
    parser.add_argument('--init_scale', type=float, default=0.1)
    parser.add_argument('--clip_gradient', type=int, default=2500)
    parser.add_argument('--ablation_type', default='full')
    parser.add_argument('--event_embedding_method', default='masked')
    parser.add_argument('--attention_key_weight',default=trail.suggest_categorical('attention_key_weight',[1,4,8,16,32,64,128,256]))
    parser.add_argument('--attention_querry_weight',default=trail.suggest_categorical('attention_querry_weight',[1,4,8,16,32,64,128,256]))
    parser.add_argument('--seq', type=int, default=20, help='seq')
    parser.add_argument('--num_event_types',default=3)
    pars = parser.parse_args()
    if pars.use_nongraded == False:
        parser.add_argument('--class_weights', type=list, default=[1])
    else:
        parser.add_argument('--class_weights', type=list, default=[1, 1, 1])


    pars = parser.parse_args()

    filename_params = f'{dataset}_{pars.ablation_type}_b_{pars.batch_size}_rembed_{pars.result_embedding_size}_h{pars.num_hidden}' \
                      f'_typeembed_{pars.type_embedding_size}_fembed{pars.event_f_hidden}_typeh{pars.event_type_hidden}' \
                      f'_kh_{pars.knowledge_hidden}_qh{pars.quiz_hidden}_m{pars.memory_size}_dk{pars.dk}_dv{pars.dv}_eventembed{pars.event_embedding_size}' \
                      f'_k{pars.attention_key_weight}_q{pars.attention_querry_weight}_{pars.seq}'

    train_iter,valid_iter,test_iter,num_event_types,tags_num = split_data(pars)

    model = model_debug_new.MODEL(pars.batch_size, pars.quiz_embedding_size, num_event_types, pars.seq,
                              pars.num_hidden,pars.event_f_hidden,pars.event_type_hidden
                  ,pars.knowledge_hidden,pars.quiz_hidden,
                  pars.type_embedding_size, pars.time_embedding_size,
                  pars.memory_size, pars.dk, pars.dv, pars.attention_key_weight,pars.attention_querry_weight,
                              binary_result=True, forget_bias=1, quiz_has_tags=pars.quiz_has_tags
                  , tags_num=tags_num, event_embedding_size=pars.event_embedding_size,tag_onehot=pars.tag_onehot,
                  result_embedding_size=pars.result_embedding_size,use_nongraded=pars.use_nongraded,
                              event_embedding_method=pars.event_embedding_method,ablation_type=pars.ablation_type)


    cell= model.KTHawkes_model()
    net = mx.mod.Module(symbol=cell, context=mx.cpu(4), data_names=[x for x in cell.list_inputs() if '_input' in x],
                        label_names=[x for x in cell.list_inputs() if '_label' in x])

    def train_test_batch(net, data, is_train=True):
        train_quiz_pred_all = np.array([])
        train_quiz_true_all = np.array([])
        train_type_pred_all = np.array([])
        train_type_true_all= np.array([])
        train_time_pred_all = np.array([])
        train_time_true_all = np.array([])
        train_type_prob_all = np.array([])
        train_batch_loss = []


        for batch in data:
            net.forward(batch, is_train=is_train)
            preds = [x.asnumpy() for x in net.get_outputs()]
            net.backward()
            norm_clipping(net._exec_group.grad_arrays, pars.clip_gradient)
            net.update()
            quiz_pred = np.argmax(preds[0], axis=1) # batch x seq
            true_quiz = batch.label[-1].asnumpy().reshape(-1, ) # batch x seq



            train_batch_loss.append(preds[-1][0])

            true_time = batch.label[0].asnumpy().reshape(-1, )
            pred_time = preds[2]

            true_type = batch.label[1].asnumpy().reshape(-1, )
            pred_type_prob = preds[1]
            pred_type = np.argmax(pred_type_prob, 1).reshape(-1, )
            train_type_prob_all = np.concatenate([train_type_prob_all, pred_type_prob]) if train_type_prob_all.size else pred_type_prob

            train_type_pred_all = np.concatenate([train_type_pred_all, pred_type]) if train_type_pred_all.size else pred_type
            train_type_true_all = np.concatenate([train_type_true_all, true_type]) if train_type_true_all.size else true_type


            train_quiz_pred_all = np.concatenate([train_quiz_pred_all, quiz_pred.reshape(batch.data[0].shape)],0) if train_quiz_pred_all.size else quiz_pred.reshape(batch.data[0].shape)

            train_quiz_true_all = np.concatenate([train_quiz_true_all, true_quiz.reshape(batch.data[0].shape)]) if train_quiz_true_all.size else true_quiz.reshape(batch.data[0].shape)


            train_time_pred_all = np.concatenate([train_time_pred_all, pred_time]) if train_time_pred_all.size else pred_time

            train_time_true_all = np.concatenate([train_time_true_all, true_time]) if train_time_true_all.size else true_time



        train_quiz_true_all_flatten = train_quiz_true_all.reshape(-1,)
        train_quiz_pred_all_flatten = train_quiz_pred_all.reshape(-1,)
        train_index_all = np.flatnonzero(train_quiz_true_all_flatten != -1)

        epoch_quiz_acc = accuracy_score(train_quiz_true_all_flatten[train_index_all], train_quiz_pred_all_flatten[train_index_all])


        epoch_quiz_auc = roc_auc_score(train_quiz_true_all_flatten[train_index_all], train_quiz_pred_all_flatten[train_index_all])

        epoch_type_acc = accuracy_score(train_type_true_all, train_type_pred_all)
        epoch_type_auc = type_auc(train_type_true_all, train_type_prob_all)

        epoch_rmse = np.sqrt(np.mean(np.square(train_time_true_all - train_time_pred_all)))

        epoch_loss = np.mean(train_batch_loss)
        meta_result = pd.DataFrame([train_quiz_true_all_flatten,train_quiz_pred_all_flatten,train_type_true_all,train_type_pred_all,train_time_true_all,train_time_pred_all]).T
        return epoch_quiz_acc, epoch_quiz_auc, epoch_loss, epoch_rmse, epoch_type_acc, epoch_type_auc,meta_result

    def train_valid_epoch(net, train_data, valid_data, test_data, epoch=5,save_meta = True):
        net.bind(train_data.provide_data, train_data.provide_label)

        net.init_params(initializer=mx.init.Uniform(scale=pars.init_scale))
        net.init_optimizer(optimizer=mx.optimizer.Adam(learning_rate=pars.init_lr, rescale_grad=1 / pars.batch_size
                                                       , clip_gradient=pars.clip_gradient))
        r_train_acc = []
        r_train_auc = []
        r_train_loss = []
        r_valid_acc = []
        r_valid_auc = []
        r_valid_loss = []
        r_test_acc = []
        r_test_auc = []
        r_test_loss = []
        r_test_rmse = []
        r_test_type_acc = []
        r_test_type_auc = []
        for e in range(1, epoch):
            print(e)
            train_data.reset()
            valid_data.reset()
            test_data.reset()

            train_epoch_acc, train_epoch_auc, train_epoch_loss, train_rmse, train_type_acc, train_type_auc, train_meta_result \
                = train_test_batch(net, train_data)

            if np.isnan(train_epoch_loss) == False:
                valid_epoch_acc, valid_epoch_auc, valid_epoch_loss, valid_rmse, valid_type_acc, valid_type_auc, valid_meta_result \
                    = train_test_batch(net, valid_data, False)
                if np.isnan(valid_epoch_loss) == False:
                    test_epoch_acc, test_epoch_auc, test_epoch_loss, test_rmse, test_type_acc, test_type_auc, test_meta_result \
                        = train_test_batch(net, test_data,False)
                    print(e, train_epoch_acc, valid_epoch_acc, test_epoch_acc, test_rmse, test_type_acc, test_type_auc)

                    r_train_acc.append(train_epoch_acc)
                    r_train_auc.append(train_epoch_auc)
                    r_train_loss.append(train_epoch_loss)

                    r_valid_acc.append(valid_epoch_acc)
                    r_valid_auc.append(valid_epoch_auc)
                    r_valid_loss.append(valid_epoch_loss)

                    r_test_acc.append(test_epoch_acc)
                    r_test_auc.append(test_epoch_auc)
                    r_test_loss.append(test_epoch_loss)
                    r_test_rmse.append(test_rmse)
                    r_test_type_acc.append(test_type_acc)
                    r_test_type_auc.append(test_type_auc)

                else:
                    break
            else:
                break
        return r_train_acc, r_train_auc, r_train_loss, r_valid_acc, r_valid_auc, r_valid_loss, r_test_acc, r_test_auc, \
               r_test_loss, r_test_rmse, r_test_type_acc, r_test_type_auc

    train_acc, train_auc, train_loss, valid_acc, valid_auc, valid_loss, test_acc, test_auc, test_loss, r_test_rmse, r_test_type_acc, r_test_type_auc = \
        train_valid_epoch(net, train_iter, valid_iter, test_iter, epoch=pars.epoch,save_meta=True)

    r_df = pd.DataFrame(
        [train_acc, train_auc, train_loss, valid_acc, valid_auc, valid_loss, test_acc, test_auc, test_loss, r_test_rmse,
         r_test_type_acc, r_test_type_auc]).T
    r_df = r_df.dropna()
    r_df.columns = ["train_acc", "train_auc", "train_loss", "valid_acc", "valid_auc", "valid_loss", "test_acc",
                    "test_auc"
        , "test_loss", "r_test_rmse", "r_test_type_acc", "r_test_type_auc"]


    if len(r_df)>0:
        r_df.to_csv(f'../results/{dataset}/{filename_params}.csv', index=False)
        best_epoch = np.argmin(r_df['valid_loss'].tolist())
        test_acc = r_df['test_acc'].tolist()[best_epoch]
        test_auc = r_df['test_auc'].tolist()[best_epoch]
        test_rmse = r_df['r_test_rmse'].tolist()[best_epoch]

        test_type_acc = r_df['r_test_type_acc'].tolist()[best_epoch]
        test_type_auc = r_df['r_test_type_auc'].tolist()[best_epoch]
    else:
        test_acc,test_auc,test_rmse,test_type_acc,test_type_auc = 0,0,0,0,0
    return test_acc,test_auc,test_rmse,test_type_acc,test_type_auc


if __name__ == '__main__':
    study = optuna.create_study(directions=['maximize','maximize','minimize','maximize','maximize'])
    study.optimize(objective, n_trials=150)
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
