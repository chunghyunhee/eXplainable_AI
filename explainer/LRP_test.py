import numpy as np
from tensorflow import keras
from numpy import newaxis as na

class LRP_LSTM(object):
    def __init__(self, model):
        self.model = model

        names = [weight.name for layer in model.layers for weight in layer.weights]
        weights = model.get_weights()

        # 모델에서 각 layer 가져오기
        np.set_printoptions(suppress=True)
        for name, weight in zip(names, weights):
            if name == 'lstm/lstm_cell_1/kernel:0':
                first_layer = weight
            if name == 'lstm/lstm_cell_1/recurrent_kernel:0':
                second_layer = weight
            if name == 'lstm/lstm_cell_1/bias:0':
                third_layer = weight
            elif name == 'dense/kernel:0':
                output_layer = weight


        print("kernel_0", first_layer.shape)
        print("recurrent_kernel_0", second_layer.shape)
        print("bias_0", third_layer.shape)
        print("output", output_layer.shape)

        # self.Wxh_Left (256, 32)
        # self.Whh_Left (256, 32)
        # self.bxh_Left (256,)
        # self.Why_Left (2, 32)

        self.Wxh = first_layer.T  # shape 4d*e
        self.Whh = second_layer.T  # shape 4d
        self.bxh = third_layer.T  # shape 4d
        self.Why = output_layer.T

        #print(self.Wxh.shape)
        #print(self.Whh.shape)
        #print(self.bxh.shape)
        #print(self.Why.shape)

    def lrp_linear(self, hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor=1.0, debug=False):
        ## with the epsilcon LRP
        sign_out = np.where(hout[na,:]>=0, 1., -1.) # shape (1, M)
        numer    = (w * hin[:,na]) + ( bias_factor * (b[na,:]*1. + eps*sign_out*1.) / bias_nb_units ) # shape (D, M)
        denom    = hout[na,:] + (eps*sign_out*1.)   # shape (1, M)
        message  = (numer/denom) * Rout[na,:]       # shape (D, M)
        Rin      = message.sum(axis=1)              # shape (D,)

        if debug:
            print("local diff: ", Rout.sum() - Rin.sum())
        return Rin

    def get_layer_output(self, layer_name, data):
        intermediate_layer_model = keras.Model(inputs=self.model.input,outputs=self.model.get_layer(layer_name).output)
        return intermediate_layer_model.predict(data)

    def run(self, target_data, target_class):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        x = self.get_layer_output('embedding', target_data).squeeze(axis=1)
        e = x.shape[1]

        ################# forword
        T = target_data.shape[0]
        d = int(32)           # hidden units with int(32)
        C = self.Why.shape[0] # number of classes

        idx    = np.hstack((np.arange(0,d), np.arange(2*d,4*d))).astype(int) # indices of gates i,f,o together
        idx_i, idx_f, idx_c, idx_o = np.arange(0,d), np.arange(d,2*d), np.arange(2*d,3*d), np.arange(3*d,4*d) # indices of gates i,g,f,o separately

        # 최종적으로 구하려는 값은 c에 저장될 값과 h으로 지워질 값
        h  = np.zeros((T,d))
        c  = np.zeros((T,d))

        gates_pre = np.zeros((T, 4*d))  # gates pre-activation
        gates     = np.zeros((T, 4*d))  # gates activation

        for t in range(T):
            #print(self.Wxh.shape, x[t].shape )
            #print(self.Whh.shape, h[t-1].shape )
            #print(self.bxh.shape)
            gates_pre[t]    = np.dot(self.Wxh, x[t]) + np.dot(self.Whh, h[t-1]) + self.bxh

            gates[t,idx]    = sigmoid(gates_pre[t,idx])
            gates[t,idx_c]  = np.tanh(gates_pre[t,idx_c])

            c[t]            = gates[t,idx_f]*c[t-1] + gates[t,idx_i]*gates[t,idx_c]
            h[t]            = gates[t,idx_o]*np.tanh(c[t])

            score = np.dot(self.Why, h[t])

        ################# backwork
        dx     = np.zeros(x.shape)

        dh = np.zeros((T, d))
        dc = np.zeros((T, d))
        dgates_pre = np.zeros((T, 4*d))  # gates pre-activation
        dgates = np.zeros((T, 4*d))  # gates activation

        ds = np.zeros((C))
        ds[target_class] = 1.0
        dy = ds.copy()

        #맨처음을 0으로 시작하면 안됨
        dh[T-1]     = np.dot(self.Why.T, dy)

        for t in reversed(range(T)):
            dgates[t,idx_o]    = dh[t] * np.tanh(c[t])  # do[t]
            dc[t]             += dh[t] * gates[t,idx_o] * (1.-(np.tanh(c[t]))**2) # dc[t]
            dgates[t,idx_f]    = dc[t] * c[t-1]         # df[t]
            dc[t-1]            = dc[t] * gates[t,idx_f] # dc[t-1]
            dgates[t,idx_i]    = dc[t] * gates[t,idx_c] # di[t]
            dgates[t,idx_c]    = dc[t] * gates[t,idx_i] # dg[t]
            dgates_pre[t,idx]  = dgates[t,idx] * gates[t,idx] * (1.0 - gates[t,idx]) # d ifo pre[t]
            dgates_pre[t,idx_c]= dgates[t,idx_c] *  (1.-(gates[t,idx_c])**2) # d c pre[t]
            dh[t-1]            = np.dot(self.Whh.T, dgates_pre[t])
            dx[t]              = np.dot(self.Wxh.T, dgates_pre[t])

        ################# LRP
        eps=1e-3
        bias_factor=1.0
        Rx  = np.zeros(x.shape)
        Rh  = np.zeros((T+1, d))
        Rc  = np.zeros((T+1, d))
        Rg  = np.zeros((T,   d)) # gate g only

        Rout_mask            = np.zeros((C))
        Rout_mask[target_class] = 1.0

        # format shape : lrp_linear(hin, w, b, hout, Rout, bias_nb_units, eps, bias_factor)
        Rh[T-1]  = self.lrp_linear(h[T-1], self.Why.T, np.zeros((C)), score, score*Rout_mask, d, eps, bias_factor, debug=False)

        for t in reversed(range(T)):
            Rc[t]   += Rh[t]
            Rc[t-1]  = self.lrp_linear(gates[t,idx_f]*c[t-1], np.identity(d), np.zeros((d)), c[t], Rc[t], d, eps, bias_factor, debug=False)
            Rg[t]    = self.lrp_linear(gates[t,idx_i]*gates[t,idx_c], np.identity(d), np.zeros((d)), c[t], Rc[t], d, eps, bias_factor, debug=False)
            Rx[t]    = self.lrp_linear(x[t], self.Wxh[idx_c].T, self.bxh[idx_c], gates_pre[t,idx_c], Rg[t], d+e, eps, bias_factor, debug=False)
            Rh[t-1]  = self.lrp_linear(h[t-1], self.Whh[idx_c].T, self.bxh[idx_c], gates_pre[t,idx_c], Rg[t], d+e, eps, bias_factor, debug=False)

        return score, x, dx, Rx, Rh[-1].sum()