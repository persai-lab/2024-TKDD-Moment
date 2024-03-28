import mxnet as mx
from mxnet.gluon import nn

class MEMORY_unit(object):
    def __init__(self,quiz_embedded_size,result_embeded_size,hidden_num,batch_size,is_write,memory_size,dk,dv,
                 attention_key_weight,attention_querry_weight,name = 'MemoryUnit'):
        self.batch_size = batch_size
        self.is_write = is_write
        self.memory_size = memory_size
        self.dk = dk
        self.dv = dv
        self.name = name
        self.hidden_num = hidden_num
        self.quiz_embedded_size = quiz_embedded_size
        self.result_embeded_size = result_embeded_size
        self.attention_key_weight = attention_key_weight
        self.attention_querry_weight = attention_querry_weight

        # if self.is_write:
        self.W_c = mx.sym.Variable(name=f"{name}_c_weight")
        self.b_c = mx.sym.Variable(name=f"{name}_c_bias")
        self.W_z = mx.sym.Variable(name=f"{name}_z_weight")
        self.b_z = mx.sym.Variable(name=f"{name}_z_bias")
        #
        # self.W_k = mx.sym.Variable(name = f'{name}_mlpk_weight')
        # self.W_q = mx.sym.Variable(name = f'{name}_mlpq_weight')
        # self.V = mx.sym.Variable(name = f'{name}_mlpv_weight')
        # self.W_v = mx.sym.Variable(name = f'{name}_mlpv_weight')
        # self.W_s = mx.sym.Variable(name = 'attention_embed_weight')
        self.W_q = mx.sym.Variable(f'{name}_query_weight')
        self.W_k = mx.sym.Variable(f'{name}_key_weight')
        self.W_v1 = mx.sym.Variable(f'{name}_value_fc1_weight')
        self.W_v2 = mx.sym.Variable(f'{name}_value_fc2_weight')

        self.W_s = mx.sym.Variable(f'{name}_attention_fc_weight')
        self.W_s2 = mx.sym.Variable(f'{name}_attention_fc2_weight')







    def attention_weight(self,control_input,memory_key,h):
        '''
        :param control_input:
        :param memory_key:
        :param h:
        :return:
                weight: batch x 1x memory_size
                S: batch x 1 x dv
        '''
        K = memory_key # memory_size x dk
        concat_input = mx.sym.Concat(h,control_input) #batch x (hid+E)
        Q = mx.sym.FullyConnected(concat_input,weight=self.W_q,num_hidden=self.dk,flatten=False,no_bias=True) #batch x dk
        Q = mx.sym.expand_dims(Q,axis=1)
        score = mx.sym.dot(Q,mx.sym.transpose(K)) # batch x 1 x memory_size
        weight = mx.sym.softmax(score,axis=2)
        # weight = mx.sym.broadcast_div(weight,mx.sym.sqrt(self.dv))# batch x 1 x memory_size
        # weight = mx.sym.squeeze(weight,axis=1) # batch x memory_size
        V = mx.sym.FullyConnected(concat_input,weight=self.W_v1,num_hidden=self.dv,no_bias=True) #batch x dv
        V = mx.sym.expand_dims(V,axis=2) #batch x 1 x dv
        # V = mx.sym.FullyConnected(V,weight = self.W_v2,num_hidden=self.memory_size,flatten=False,no_bias=True) #batch x memory_size x dv
        V = mx.sym.batch_dot(V,mx.sym.ones(shape = (self.batch_size,1,self.memory_size)))

        V = mx.sym.transpose(V,axes=[0,2,1])
        S = mx.sym.batch_dot(weight,V) #batch x 1 x dv
        S = mx.sym.squeeze(S,axis=1)
        S = mx.sym.expand_dims(S,axis=2)

        Semb = mx.sym.batch_dot(S,mx.sym.ones(shape = (self.batch_size,1,self.memory_size)))
        Semb = mx.sym.transpose(Semb,[0,2,1])
        return weight,Semb




    def update_tensor(self,control_input,memory_key,memory_value,h):
        weight, S = self.attention_weight(control_input, memory_key, h)
        M2h = mx.sym.FullyConnected(data = memory_value, num_hidden= self.dv*2, weight= self.W_c,bias=self.b_c,flatten=False)
        S2h = mx.sym.FullyConnected(data = S, num_hidden= self.dv*2, weight= self.W_z,bias=self.b_z,flatten=False)
        gates = M2h + S2h
        slice_gates = mx.sym.SliceChannel(gates, num_outputs=2, axis = 2,name='slice_gate')

        current_state = mx.sym.Activation(data=slice_gates[0], act_type='tanh', name='M_tilde')
        updated_gate = mx.sym.Activation(data=slice_gates[1], act_type='sigmoid', name='Z')
        updated_memory = mx.sym.broadcast_mul(1 - updated_gate, current_state)
        updated_memory = updated_memory + mx.sym.broadcast_mul(updated_gate, memory_value)
        updated_memory = updated_memory.reshape(shape=(self.batch_size, self.memory_size, self.dv))
        return updated_memory


    def read(self,memory_key,memory_value,h,control_input,read_weight):
        if read_weight is None:
            read_weight,_ = self.attention_weight(control_input,memory_key,h) #batch x N x emb_size
        read_content = mx.sym.Reshape(data=mx.sym.batch_dot(read_weight, memory_value),
                                          shape=(-1, self.dv)) # batch x hidden

        return read_content


class MEMORY(object):
    def __init__(self,quiz_embedded_size,result_embedded_size,
                 hidden_num,memory_size,batch_size,dk,dv,
                 attention_key_weight,attention_querry_weight,init_Mk,
                 init_Mv,name = 'Memory'):
        self.quiz_embedded_size = quiz_embedded_size
        self.result_embeded_size = result_embedded_size
        self.hidden_num = hidden_num
        self.batch_size = batch_size
        self.name = name
        self.memory_size = memory_size
        self.dk = dk
        self.dv = dv
        self.init_Mk = mx.sym.Variable(self.name + ":init_memory_key_weight") \
            if init_Mk is None else init_Mk
        self.init_Mv = mx.sym.Variable(self.name + ":init_memory_key_weight") \
            if init_Mv is None else init_Mv
        self.attention_key_weight = attention_key_weight
        self.attention_querry_weight = attention_querry_weight

        self.Mk = MEMORY_unit(quiz_embedded_size = self.quiz_embedded_size,
                              result_embeded_size = self.result_embeded_size,hidden_num = self.hidden_num,
                              batch_size = self.batch_size,is_write=False,memory_size=self.memory_size,
                              dk = self.dk,dv = self.dv,attention_key_weight = self.attention_key_weight
                              ,attention_querry_weight=self.attention_querry_weight,name = 'Memory_key')
        self.Mv = MEMORY_unit(quiz_embedded_size = self.quiz_embedded_size,
                              result_embeded_size = self.result_embeded_size,hidden_num = hidden_num,
                              batch_size = batch_size,is_write=True,memory_size=self.memory_size,
                              dk = self.dk,dv = self.dv,attention_key_weight = self.attention_key_weight,
                              attention_querry_weight=self.attention_querry_weight,name= 'Memory_value')
        self.memory_key = self.init_Mk
        self.memory_value = self.init_Mv



    def attention(self,control_input,h):
        assert isinstance(control_input, mx.symbol.Symbol)
        weight_matrix, S = self.Mk.attention_weight(control_input = control_input,memory_key=self.memory_key,h = h)
        return weight_matrix,S

    def read(self,h,control_input,weight_matrix):
        read_content = self.Mv.read(self.memory_key,self.memory_value,h,control_input,read_weight = weight_matrix)
        return read_content

    def updater(self,control_input,h):
        self.memory_value = self.Mv.update_tensor(control_input,self.memory_key,self.memory_value,h)
        return self.memory_value