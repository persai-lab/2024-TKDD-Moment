import mxnet as mx
import pandas as pd
from utils import tag2bow
def split_data(pars):
    dat = pd.read_pickle(f'{pars.dataset}_data')
    type_array, quiz_array, result_array, time_interval_array, padding_array,tag_array =\
        dat['type_array'],dat['quiz_array'],dat['result_array'],dat['time_interval_array'],dat['padding_array'],dat['tag_array']
    quiz_response_label,event_type_label,event_time_label = dat['quiz_response_label'],dat['event_type_label'],dat['event_time_label']
    tags_num = 0
    type_array = type_array
    quiz_array = quiz_array
    result_array = result_array
    time_interval_array = time_interval_array
    padding_array = padding_array
    if pars.quiz_has_tags:
        try:
            tag_array, tags_num = tag2bow(tag_array)
        except:
            tags_num = len(set(tag_array.reshape(-1,)))

    num_event_types = len([x for x in list(set(type_array.reshape(-1, ))) if x != -1])

    train_size  = int(0.8*len(quiz_array))
    valid_size = int(0.2*len(quiz_array))

    train_data_dict = {
        'quiz_input': quiz_array[:train_size, :pars.seq],
        'result_input': result_array[:train_size, :pars.seq],
        'tags_input': tag_array[:train_size, :pars.seq],
        'class_weights_input':mx.nd.broadcast_mul(mx.nd.ones(shape = (train_size,pars.seq,num_event_types)),
                                                   mx.nd.array(pars.class_weights)),
        'quiz_weights_input': mx.nd.broadcast_mul(mx.nd.ones(shape=(train_size, pars.seq, 2)),
                                                  mx.nd.array(pars.quiz_weights)),
        'time_input': time_interval_array[:train_size, :pars.seq],
        'type_input': type_array[:train_size, :pars.seq],
        'b_input': padding_array[:train_size, :pars.seq]
    }
    train_label_dict =  {
            'quiz_response_label': quiz_response_label[:train_size, :pars.seq],# in the prediction, it reads first then write
            'event_type_label': event_type_label[:train_size, :pars.seq],
            'event_time_label':event_time_label[:train_size, :pars.seq]

        }
    valid_data_dict = {
            'quiz_input': quiz_array[train_size:train_size+valid_size, :pars.seq],
            'result_input': result_array[train_size:train_size+valid_size, :pars.seq],
            'tags_input': tag_array[train_size:train_size+valid_size, :pars.seq],
            'class_weights_input':mx.nd.broadcast_mul(mx.nd.ones(shape = (len(quiz_array),pars.seq,num_event_types)),
                                                       mx.nd.array(pars.class_weights)),
            'quiz_weights_input': mx.nd.broadcast_mul(mx.nd.ones(shape=(len(quiz_array), pars.seq, 2)),
                                                       mx.nd.array(pars.quiz_weights)),
            'time_input': time_interval_array[train_size:train_size+valid_size, :pars.seq],
            'type_input': type_array[train_size:train_size+valid_size, :pars.seq],
            'b_input': padding_array[train_size:train_size+valid_size, :pars.seq]
        }
    valid_label_dict = {
                'quiz_response_label': quiz_response_label[train_size:train_size+valid_size, :pars.seq],

                'event_type_label': event_type_label[train_size:train_size+valid_size, :pars.seq],
                'event_time_label':event_time_label[train_size:train_size+valid_size, :pars.seq]

            }
    test_data_dict = {
            'quiz_input': quiz_array[:, -pars.seq:],
            'result_input': result_array[:,  -pars.seq:],
            'tags_input': tag_array[:,  -pars.seq:],
            'class_weights_input':mx.nd.broadcast_mul(mx.nd.ones(shape = (len(quiz_array),pars.seq,num_event_types)),
                                                       mx.nd.array(pars.class_weights)),
            'quiz_weights_input': mx.nd.broadcast_mul(mx.nd.ones(shape=(len(quiz_array), pars.seq, 2)),
                                                       mx.nd.array(pars.quiz_weights)),
            'time_input': time_interval_array[:, -pars.seq:],
            'type_input': type_array[:, -pars.seq:],
            'b_input': padding_array[:, -pars.seq:]

        }
    test_label_dict = {
                'quiz_response_label': quiz_response_label[:, pars.seq:],
                'event_type_label': event_type_label[:, pars.seq:],
                'event_time_label':event_time_label[:, pars.seq:]

            }
    train_iter = mx.io.NDArrayIter(train_data_dict,train_label_dict,pars.batch_size, shuffle=True)

    valid_iter = mx.io.NDArrayIter(valid_data_dict,valid_label_dict,pars.batch_size, shuffle=True)

    test_iter =  mx.io.NDArrayIter(test_data_dict,test_label_dict,pars.batch_size, shuffle=True)


    return train_iter,valid_iter,test_iter,num_event_types,tags_num