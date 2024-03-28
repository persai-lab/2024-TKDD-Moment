from memory_debug import *
import mxnet as mx


class MODEL(object):


    def __init__(self,batch_size,num_quiz,num_event_types,
                 sequence_length,
                 num_hidden,event_f_hidden,event_type_hidden,knowledge_hidden,quiz_hidden,type_embedding_size,time_embedding_size,
                 memory_size,dk,dv,attention_key_weight,attention_querry_weight,
                 binary_result = True,forget_bias = 1
                 ,quiz_has_tags = False,tags_num = 0,result_embedding_size = 150
                 ,event_embedding_size = 3,tag_onehot = True,use_nongraded = True,ablation_type = 'full'
                 ,event_embedding_method  = None):
    #ablation_type in ['full','quiz_only','quiz+time','quiz+type']
        self.batch_size = batch_size
        self.num_quiz = num_quiz
        self.num_event_types = num_event_types
        self.sequence_length = sequence_length
        self.num_hidden = num_hidden
        self.memory_size = memory_size
        self.dk = dk
        self.dv = dv
        # self.init_Mk = init_Mk
        # self.init_Mv = init_Mv
        self.binary_result = binary_result
        self.quiz_has_tags = quiz_has_tags
        self.tags_num = tags_num
        self.forget_bias =forget_bias

        # variable name should be end with "_weight"
        self._iW = mx.symbol.Variable(name="i2h_weight")
        self._hW = mx.symbol.Variable(name="h2h_weight")
        self._mW = mx.symbol.Variable(name="m2h_weight")
        self._cW = mx.symbol.Variable(name="c2h_weight")
        self._iB = mx.symbol.Variable(name="i2h_bias", init=mx.init.LSTMBias(forget_bias))
        self._mB = mx.symbol.Variable(name="m2h_bias", init=mx.init.LSTMBias(forget_bias))
        self._cB = mx.symbol.Variable(name="c2h_bias", init=mx.init.LSTMBias(forget_bias))
        self._hB = mx.symbol.Variable(name="h2h_bias", init=mx.init.LSTMBias(forget_bias))
        self.type_W = mx.sym.Variable(name='type_weight')
        self.type_b = mx.sym.Variable(name = 'type_bias')
        self.alpha_W = mx.sym.Variable(name = 'alpha_lambda_weight')
        self.beta_W = mx.sym.Variable(name = 'beta_lambda_weight')
        self.b_lambda = mx.sym.Variable(name = 'lambda_bias')
        self.read_contac_W = mx.sym.Variable(name = 'read_concat_weight')
        self.read_contac_b = mx.sym.Variable(name = 'read_concat_bias')
        self.result_fc_W = mx.sym.Variable(name = 'result_fc_weight')
        self.result_fc_b = mx.sym.Variable(name = 'result_fc_bias')
        self.type_pred_W = mx.sym.Variable(name = 'type_pred_weight')
        self.type_pred_b = mx.sym.Variable(name = 'type_pred_bias')
        self.quiz_pred_W = mx.sym.Variable(name = 'quiz_pred_weight')
        self.quiz_pred_b = mx.sym.Variable(name = 'quiz_pred_bias')
        self.time_pred_W = mx.sym.Variable(name = 'time_pred_weight')
        self.time_pred_b = mx.sym.Variable(name = 'time_pred_bias')
        self.type_fc_W = mx.sym.Variable(name= 'type_fc_weight')
        self.type_fc_b = mx.sym.Variable(name= 'type_fc_bias')
        self.event_type_weight = mx.sym.Variable(name= 'event_type_weight')
        self.event_type_bias = mx.sym.Variable(name= 'event_type_bias')
        self.event_embedding_weight = mx.sym.Variable(name='event_embedding_weight')
        self.result_embedding_weight = mx.sym.Variable(name='result_embedding_weight')
        self.result_embedding_size = result_embedding_size
        self.event_embedding_size = event_embedding_size
        self.time_fc_W = mx.sym.Variable(name = 'time_fc_weight')
        self.time_fc_b = mx.sym.Variable(name = 'time_fc_bias')
        self.event_embedding_bias = mx.sym.Variable(name = 'event_embedding_bias')
        self.join_time_weight = mx.sym.Variable(name = 'join_time_weight')
        self.joint_time_bias = mx.sym.Variable(name = 'joint_time_bias')
        self.type_embedding_size = type_embedding_size
        self.time_embedding_size = time_embedding_size
        self.tag_onehot = tag_onehot

        self.event_f_hidden = event_f_hidden
        self.event_type_hidden = event_type_hidden
        self.knowledge_hidden = knowledge_hidden
        self.quiz_hidden = quiz_hidden
        self.time_weight = mx.sym.Variable(name= 'time_weight')
        self.time_bias = mx.sym.Variable(name = 'time_bias')

        self.use_nongraded = use_nongraded
        self.ablation_type = ablation_type
        self.event_embedding_method = event_embedding_method

        self.attention_key_weight = attention_key_weight
        self.attention_querry_weight = attention_querry_weight
        self.base_rate = mx.sym.Variable('base_bias',shape=(self.batch_size,1))

    def log_f(self,d_j, h):
        # base = mx.sym.Variable('base_bias',shape=(self.batch_size,1))
        fch_excite = mx.sym.FullyConnected(data=h, num_hidden=1, weight=self.alpha_W,no_bias=True, name='fch_excite')
        fch_decay = mx.sym.FullyConnected(data=h, num_hidden=1,  weight=self.beta_W,no_bias=True, name='fch_decay')
        self_excite = mx.sym.Activation(data=fch_excite, act_type='tanh', name='self-excitement')
        # decay = -mx.sym.Activation(data=fch_decay, act_type='tanh', name='decay')
        decay = mx.sym.maximum(1e-3,fch_decay)
        intensity = self_excite*mx.sym.exp(decay*(d_j))+self.base_rate
        intensity = mx.sym.maximum(1e-3,intensity)
        f = intensity*mx.sym.exp(-self.b_lambda*d_j + (self_excite - self_excite*mx.sym.exp(-decay*d_j))/decay)
        return mx.sym.log(f),self_excite,decay


    def get_weights(self, args, gates_names=None, group_names=None):
        """ to get the weights from args by the four gates ['i','f','c','o'] """
        gate_names = ['i', 'f', 'c', 'o'] if gates_names is None else gates_names
        args = args.copy()
        h = self.num_hidden
        group_names = ['i2h', 'h2h', 'c2h', 'm2h'] if gates_names is None else group_names
        for group_name in group_names:
            weight = args.pop(f'{group_name}_weight')
            bias = args.pop(f'{group_name}_bias')
            for j, gate in enumerate(gate_names):
                wname = f'{group_name}_{gate}_weight'
                args[wname] = weight[j * h:(j + 1) * h].copy()
                bname = f'{group_name}_{gate}_bias'
                args[bname] = bias[j * h:(j + 1) * h].copy()
        return args


    def Hawkes_updater_cell(self,h,c,b,m,x,counter,dropout = 0.2):
        '''lstm cell updater'''
        name = f'{counter}_'
        h2h = mx.sym.FullyConnected(data=h, num_hidden=self.num_hidden * 4,weight=self._hW,bias=self._hB,name='h2h')
        # h2h = mx.sym.BatchNorm(h2h)

        if dropout>0:
            x = mx.sym.Dropout(data=x,p=dropout)
        i2h = mx.sym.FullyConnected(data=x, num_hidden=self.num_hidden * 4,weight=self._iW,bias=self._iB,name='i2h')
        # i2h = mx.sym.BatchNorm(i2h)

        m2h = mx.sym.FullyConnected(data = m, num_hidden= self.num_hidden * 4,weight=self._mW,bias=self._mB, name='m2h')
        # m2h = mx.sym.BatchNorm(m2h)
        #shape = batch_size * memory_size
        c2h = mx.sym.FullyConnected(data = c, num_hidden=self.num_hidden * 4, weight=self._cW,bias=self._cB,name='c2h')
        # c2h = mx.sym.BatchNorm(c2h)

        #shape = batch_size * num_hidden
        gates = i2h + h2h + c2h \
                 + mx.sym.broadcast_mul(m2h,mx.sym.expand_dims(b,axis = 1))
        slice_gates = mx.sym.SliceChannel(gates, num_outputs=4, name='slice_gate')
        i = mx.sym.Activation(data=slice_gates[0], act_type='sigmoid', name=f'{name}i')
        f = mx.sym.Activation(data=slice_gates[1], act_type='sigmoid', name=f'{name}f')
        ctilda = mx.sym.Activation(data=slice_gates[2], act_type='tanh', name=f'{name}c_tilda')
        o = mx.sym.Activation(data=slice_gates[3], act_type="sigmoid", name=f'{name}o')
        next_c = mx.sym._internal._plus(f * c, i * ctilda, name=f'{name}c')
        next_h = mx.sym._internal._mul(o, mx.symbol.Activation(next_c, act_type="tanh"), name='h')
        return next_h, next_c, o

    def KTHawkes_model(self,init_h =None,init_c = None):
        #input data: create symbolic variables
        time_input = mx.sym.Variable('time_input',shape = (self.batch_size,self.sequence_length))
        type_input = mx.sym.Variable('type_input',shape = (self.batch_size,self.sequence_length))
        indicators_inputs = mx.sym.Variable('b_input', shape=(self.batch_size, self.sequence_length))
        event_time_label = mx.sym.Variable('event_time_label',shape=(self.batch_size,self.sequence_length))
        quiz_input = mx.sym.Variable('quiz_input', shape=(self.batch_size,self.sequence_length))
        result_input = mx.sym.Variable('result_input',shape = (self.batch_size,self.sequence_length))
        # tags_input = mx.sym.Variable('tags_input')

        quiz_embedding = mx.sym.one_hot(quiz_input, self.num_quiz)
        if self.quiz_has_tags:

            if self.tag_onehot:
                tags_input = mx.sym.Variable('tags_input',shape=(self.batch_size,self.sequence_length,self.tags_num))

                tags_embedding = tags_input
            else:
                tags_input = mx.sym.Variable('tags_input',shape=(self.batch_size,self.sequence_length))

                tags_embedding = mx.sym.one_hot(tags_input, self.tags_num)

            quiz_embedding = mx.sym.Concat(quiz_embedding, tags_embedding, dim=2)

        if self.event_embedding_method == 'masked':
            time_embedding = mx.sym.expand_dims(data=time_input, axis=2)
            time_embedding = mx.sym.log(time_embedding)
            time_embedding = mx.sym.Activation(time_embedding,act_type = 'tanh')
            time_mask = mx.sym.FullyConnected(time_embedding,weight = self.time_weight,
                                                   bias = self.time_bias,num_hidden=self.num_quiz+self.tags_num,flatten=False)
            time_mask = mx.sym.sigmoid(time_mask)
            if self.ablation_type=='full':
                quiz_embedding = mx.sym.broadcast_mul(quiz_embedding,time_mask)
            if self.ablation_type == 'quiz+time':
                quiz_embedding = mx.sym.broadcast_mul(quiz_embedding, time_mask)

        elif self.event_embedding_method =='joint':
            time_embedding = mx.sym.expand_dims(time_input,axis=2)
            time_embedding = mx.sym.FullyConnected(time_embedding,weight = self.join_time_weight,
                                                   bias = self.joint_time_bias,num_hidden=self.time_embedding_size,flatten=False)
            time_embedding = mx.sym.softmax(time_embedding,axis=2)
            time_embedding = mx.sym.FullyConnected(time_embedding,weight = self.time_weight,
                                                   no_bias = True,num_hidden=self.num_quiz+self.tags_num,flatten=False)
            if self.ablation_type == 'full':
                quiz_embedding = (quiz_embedding + time_embedding)/2
            if self.ablation_type == 'quiz+time':
                quiz_embedding = (quiz_embedding + time_embedding)/2
        else:
            time_embedding = mx.sym.expand_dims(data=time_input, axis=2)

        # quiz_embedding = mx.sym.concat(time_mask, quiz_embedding, dim=2)
        result_embedding = mx.sym.expand_dims(data=result_input, axis=2)
        result_embedding = mx.sym.Concat(quiz_embedding, result_embedding, dim=2)
        result_embedding = mx.sym.FullyConnected(result_embedding, num_hidden=self.result_embedding_size,
                                                 weight=self.result_embedding_weight, flatten=False)

        event_type_embedding = mx.sym.one_hot(type_input, self.num_event_types)
        event_type_embedding = mx.sym.FullyConnected(event_type_embedding, num_hidden=self.type_embedding_size,
                                                     weight=self.event_type_weight, bias=self.event_type_bias
                                                     , flatten=False)
        event_type_embedding = mx.sym.BatchNorm(event_type_embedding)
        if self.ablation_type =='full':
            event_embedding = mx.sym.concat(time_embedding,event_type_embedding,dim = 2)
        elif self.ablation_type in ['quiz+time','quiz_only']:
            event_embedding = time_embedding
        elif self.ablation_type == 'quiz+type':
            event_embedding = event_type_embedding


        event_embedding = mx.sym.FullyConnected(data = event_embedding,num_hidden=self.event_embedding_size
                                                ,weight=self.event_embedding_weight,bias = self.event_embedding_bias
                                                ,flatten=False)

        assert len(event_embedding.list_outputs()) == 1
        event_embedding_list = list(mx.symbol.split(event_embedding, axis=1, num_outputs=self.sequence_length,
                                   squeeze_axis=1))

        assert len(event_type_embedding.list_outputs()) == 1
        event_type_embedding_list = list(mx.symbol.split(event_type_embedding, axis=1, num_outputs=self.sequence_length,
                                   squeeze_axis=1))

        assert len(indicators_inputs.list_outputs()) == 1
        indicators_inputs_list = list(mx.symbol.split(indicators_inputs, axis=1, num_outputs=self.sequence_length,
                                           squeeze_axis=1))

        assert len(time_input.list_outputs()) == 1
        event_time_interval_list = list(mx.symbol.split(mx.sym.expand_dims(time_input,axis =2), axis=1, num_outputs=self.sequence_length,
                                   squeeze_axis=1))

        assert len(event_time_label.list_outputs()) == 1
        event_time_interval_list = list(mx.symbol.split(mx.sym.expand_dims(event_time_label,axis =2), axis=1, num_outputs=self.sequence_length,
                                   squeeze_axis=1))

        assert len(quiz_embedding.list_outputs()) == 1
        quiz_embedding_list = list(mx.symbol.split(quiz_embedding, axis=1, num_outputs=self.sequence_length,
                                   squeeze_axis=1))

        assert len(result_embedding.list_outputs()) == 1
        result_embedding_list = list(mx.symbol.split(result_embedding, axis=1, num_outputs=self.sequence_length,
                                   squeeze_axis=1))

        if init_h == None:
            init_h = mx.sym.ones(shape=(self.batch_size, self.num_hidden),name = 'h')
        if init_c == None:
            init_c = mx.sym.ones(shape=(self.batch_size, self.num_hidden),name ='c')

        # Mk = mx.sym.Variable(name = 'memory_key_weight',shape=(self.batch_size,self.memory_size,self.dk))
        Mk = mx.sym.Variable(name = 'memory_key_weight',shape=(self.memory_size,self.dk))
        M = mx.sym.Variable(name = 'memory_value_weight',shape=(self.batch_size,self.memory_size,self.dv))
        attn = mx.sym.Variable(name = 'attention_weight',shape =(self.batch_size,1,self.memory_size))
        # attn_list =
        Attn = []
        Mv_list = []
        read_contents = []
        # hawkes_pars = []
        # build memory updater

        # update lstm and memory recurrently
        # build memory updater
        mem = MEMORY(
                    # quiz_embedded_size = self.num_quiz+self.tags_num,
                    #  result_embedded_size = self.num_quiz+self.tags_num + 1,
                     quiz_embedded_size= self.result_embedding_size,
                     result_embedded_size=self.result_embedding_size,
                     hidden_num = self.num_hidden,
                     memory_size= self.memory_size,
                     batch_size= self.batch_size,
                     dk = self.dk,
                     dv = self.dv,
                    attention_key_weight = self.attention_key_weight,
                    attention_querry_weight = self.attention_querry_weight,
                     init_Mk = Mk,
                     init_Mv = M,
                     name = 'Memory')
        # update lstm and memory recurrently
        H = []  # states h
        T = []  # event pdf f*
        Q = []  # read content
        O = []
        F = [] # read content
        Alpha_list = []
        Beta_list = []
        # update_history_M = []
        for i in range(self.sequence_length):
            time_embedding_i = event_embedding_list[i]
            type_embedding_i = event_type_embedding_list[i]

            quiz_embedding_i = quiz_embedding_list[i]
            # quiz_embedding_next =
            time_i = event_time_interval_list[i]
            result_embedding_i = result_embedding_list[i]
            event_embedding_i = event_embedding_list[i]
            b = indicators_inputs_list[i]
            if i == 0:
                h = init_h
                c = init_c
                M = M
                M = M.reshape(shape=(self.batch_size, -1))
                attn,_ = mem.attention(control_input=result_embedding_i, h = init_h)
                Attn.append(attn)
                Mv_list.append(M)


            else:
                # lstm updates
                if self.ablation_type == 'full':
                    h, c, o = MODEL.Hawkes_updater_cell(self, h, c, b, M, event_embedding_i, i)
                elif self.ablation_type == 'quiz+time':
                    h, c, o = MODEL.Hawkes_updater_cell(self, h, c, b, M, time_embedding_i, i)
                elif self.ablation_type == 'quiz+type':
                    h, c, o = MODEL.Hawkes_updater_cell(self, h, c, b, M, type_embedding_i, i)

                if self.ablation_type=='quiz_only':
                    M = mem.updater(control_input=result_embedding_i, h = init_h)
                else:
                    M = mem.updater(control_input=result_embedding_i, h = h)


                attn,_ = mem.attention(control_input=result_embedding_i, h = h)
                # attn = mx.sym.abs(attn)
                Attn.append(attn)
                M = M.reshape(shape=(self.batch_size, -1))
                Mv_list.append(M)

                # update_history_M.append(M)


            quiz_read = mem.read(h, quiz_embedding_i,attn)
            # pred_t = mx.sym.FullyConnected(data = h,num_hidden=1,weight=self.t)
            # y = mx.sym.FullyConnected(data=h, weight=self.type_W, bias=self.type_b, num_hidden=self.num_event_types)
            # y = h
            f,a,beta = MODEL.log_f(self, time_i, h)
            Alpha_list.append(a)
            Beta_list.append(beta)
            # y = h
            H.append(h)
            # O.append(o)
            # H2.append()
            # True_d.append(d_j_list[i])
            T.append(f)
            Q.append(quiz_read)


        H = mx.sym.Concat(*H, num_args=self.sequence_length, dim=0)
        # O = mx.sym.Concat(*O, num_args=self.sequence_length, dim=0)
        Q = mx.sym.Concat(*Q, num_args=self.sequence_length, dim=0)
        T = mx.sym.Concat(*T, num_args=self.sequence_length, dim=0)

        Attn = mx.sym.Concat(*Attn, num_args=self.sequence_length, dim=1)
        # Attn = mx.sym.Reshape(Attn,shape=(self.batch_size,self.memory_size,-1))
        Mv_list = mx.sym.Concat(*Mv_list,dim=1)
        Alpha_list = mx.sym.Concat(*Alpha_list,dim=1)
        Beta_list = mx.sym.Concat(*Beta_list)
        '''event type and time prediction'''


        # event_f_pred = mx.sym.Concat(*T, num_args=self.sequence_length, name='event pdf prediction').reshape(-1, 1)
        event_f_pred = mx.sym.FullyConnected(data=T, weight=self.time_fc_W, bias=self.time_fc_b,
                                             num_hidden=self.event_f_hidden)
        event_f_pred = mx.sym.Activation(data=event_f_pred, act_type='tanh')
        event_f_pred = mx.sym.FullyConnected(data=event_f_pred, weight=self.time_pred_W, bias=self.time_pred_b,
                                             num_hidden=1)
        event_f_pred = mx.sym.Reshape(event_f_pred,shape=(-1,),name = 'event_f_pred')

        # event_f_pred = mx.sym.MAERegressionOutput(
        #     data=event_f_pred,label=mx.sym.Reshape(event_time_label,shape=(-1,),name = 'event_time_label'),
        #     name='event_f_pred')

        event_f_loss = mx.sym.abs(event_f_pred - mx.sym.Reshape(event_time_label,shape=(-1,)),name = 'event_f_pred_loss')


        event_type_pred = mx.sym.FullyConnected(data=H, weight=self.type_pred_W, bias=self.type_pred_b,
                                                num_hidden=self.event_type_hidden)

        event_type_pred = mx.sym.Activation(data=event_type_pred, act_type='sigmoid')



        event_type_pred = mx.sym.FullyConnected(data=event_type_pred, weight=self.type_fc_W, bias=self.type_fc_b,
                                                num_hidden=self.num_event_types)
        event_type_pred = mx.sym.softmax(data=event_type_pred, name='event_type_pred')



        event_type_label = mx.sym.Variable(name='event_type_label', shape=(self.batch_size, self.sequence_length))

        event_type_label = mx.sym.Reshape(event_type_label, shape=(-1,))

        type_mask = mx.sym.broadcast_not_equal(event_type_label, mx.sym.ones_like(event_type_label) * -1)

        type_mask = mx.sym.Reshape(type_mask, shape=(-1,))

        event_type_label = mx.sym.one_hot(event_type_label, self.num_event_types)


        # event_loss = -1*mx.sym.broadcast_mul(event_type_label,mx.sym.log(mx.sym.maximum(1e-10,event_type_pred))) - \
        #                     5*mx.sym.broadcast_mul(1-event_type_label,mx.sym.log(mx.sym.maximum(1e-10,1-event_type_pred)))
        event_loss = -mx.sym.broadcast_mul(event_type_label,mx.sym.log(mx.sym.maximum(1e-10,event_type_pred)))
        class_weights =  mx.symbol.Variable('class_weights_input', shape= (self.batch_size,self.sequence_length,self.num_event_types))
        class_weights = mx.sym.Reshape(class_weights,shape = (self.batch_size*self.sequence_length,self.num_event_types))

        event_loss = mx.sym.broadcast_mul(event_loss,class_weights)

        event_loss = mx.sym.sum(event_loss,1)
        event_loss = mx.sym.mean(event_loss * type_mask)

        time_mask = mx.sym.broadcast_not_equal(event_time_label, mx.sym.ones_like(event_time_label) * -1)
        time_mask = mx.sym.Reshape(time_mask, shape=(-1,))

        event_f_loss = mx.sym.mean(event_f_loss* time_mask)

        event_type_pred = mx.sym.BlockGrad(event_type_pred, name='event_type_pred')
        event_f_pred = mx.sym.BlockGrad(event_f_pred, name='event_f_pred')


        '''quiz prediction'''

        quiz_embedding_all = quiz_embedding.reshape(shape = (-1,self.num_quiz+self.tags_num))

        knowledge_quiz_concat = mx.sym.Concat(Q,quiz_embedding_all,num_args = 2, dim =1,
                                              name = 'knowledge quiz concatenated' )


        knowledge_quiz_embedding = mx.sym.FullyConnected(data = knowledge_quiz_concat,
                                                         weight= self.read_contac_W,
                                                         bias=self.read_contac_b,
                                                         num_hidden = self.knowledge_hidden,
                                                         name = 'knowledge quiz embedding')
        knowledge_quiz_embedding = mx.sym.BatchNorm(knowledge_quiz_embedding)

        knowledge_quiz_embedding = mx.sym.Activation(data = knowledge_quiz_embedding,
                                                     act_type='tanh',
                                                     name = 'knowledge quiz embedding tanh')
        knowledge_quiz_embedding = mx.sym.BatchNorm(knowledge_quiz_embedding)

        quiz_response = mx.sym.FullyConnected(data = knowledge_quiz_embedding,num_hidden=self.quiz_hidden,
                                              weight=self.result_fc_W,
                                              bias=self.result_fc_b,
                                              name = 'quiz prediction')
        quiz_response = mx.sym.BatchNorm(quiz_response)

        quiz_response = mx.sym.Activation(data = quiz_response,act_type='tanh')

        quiz_response_label = mx.sym.Variable(name = 'quiz_response_label',shape = (self.batch_size,self.sequence_length))
        # quiz_response_label = mx.sym.Reshape(quiz_response_label,shape = (-1,))


        quiz_pred = mx.sym.FullyConnected(data = quiz_response,num_hidden=2,
                                          weight=self.quiz_pred_W,
                                          bias=self.quiz_pred_b,
                                          )
        quiz_pred = mx.sym.BatchNorm(quiz_pred)

        quiz_pred = mx.sym.softmax(quiz_pred,name='quiz_pred')

        quiz_response_label = mx.sym.Reshape(quiz_response_label,shape=(-1,))
        mask = mx.sym.broadcast_not_equal(quiz_response_label, mx.sym.ones_like(quiz_response_label) * -1)

        mask = mx.sym.Reshape(mask,shape=(-1,))

        quiz_response_label = mx.sym.one_hot(quiz_response_label,2)

        quiz_weights_input =  mx.symbol.Variable('quiz_weights_input', shape= (self.batch_size,self.sequence_length,2))
        quiz_weights_input = mx.sym.Reshape(quiz_weights_input,shape = (self.batch_size*self.sequence_length,2))

        quiz_loss = -mx.sym.broadcast_mul(quiz_response_label,mx.sym.log(mx.sym.maximum(1e-10,quiz_pred)))

        quiz_loss = mx.sym.broadcast_mul(quiz_loss,quiz_weights_input)

        quiz_loss = mx.sym.sum(quiz_loss,1)
        quiz_loss = mx.sym.mean(quiz_loss * mask)



        quiz_pred = mx.sym.BlockGrad(quiz_pred,name= 'quiz_pred')

        sym = mx.sym.Group([quiz_pred,event_type_pred,
                            event_f_pred,
                            mx.sym.MakeLoss(10*event_loss + 100*quiz_loss + event_f_loss, name='ce_loss')])

        return sym