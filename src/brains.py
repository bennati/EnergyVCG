from utils import boltzmann
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import uuid

class Wlearner():
    def __init__(self,states,gamma = 0.5,alpha = 0.5):
        self.gamma = gamma
        self.alpha = alpha
        self.states=states
        self.wvalues=pd.DataFrame(data={"w":0},index=[str(s) for s in states])
        # self.w_count=pd.DataFrame(data={"num":0},index=[str(s) for s in states])

    def get_qtable(self):
        ret=self.wvalues.round(3)
        ret.index=self.states
        return ret,None

    # def get_qcount(self):
    #     ret=self.w_count
    #     ret.index=self.states
    #     return ret

    def learn(self,state, next_state, reward,optq_now,optq_future):
        # state=tuple(int(i) for i in state) # convert to int
        # next_state=tuple(int(i) for i in next_state) # convert to int
        # print([state,next_state,optq_now,optq_future])
        w = self.get_decision(state)
        new_w = w + self.alpha * (optq_now - reward - self.gamma * optq_future - w)
        self.wvalues.loc[str(state)] = new_w
        # print("updating wlearner at state "+str(state)+" from "+str(w)+" to "+str(new_w))
        # self.w_count.loc[str(state)]+=1

    def get_decision(self,state):
        # state=tuple(int(i) for i in state) # convert to int
        return float(self.wvalues.loc[str(state)])

class Qlearner():
    def __init__(self,states,actions,gamma = 0.5,alpha = 0.5,tmax=5):
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = np.random.normal(loc=0.5,scale=0.05)
        assert(tmax>1)          # required
        self.temp=np.random.normal(loc=tmax,scale=tmax/2)
        self.act=0
        self.reward=0
        self.actions=actions
        self.states=states
        self.qvalues=pd.DataFrame(data={a:0 for a in self.actions},index=[str(s) for s in states])
        # self.q_count=pd.DataFrame(data={"num":0},index=[str(s) for s in states])

    def train(self,low=0,high=0):
        """
        pretrain the logic to contribute
        """
        self.qvalues[0]=-0.1
        self.qvalues[1]=0.1

    def get_qtable(self):
        ret=self.qvalues.round(3)
        ret.index=self.states
        return ret,None

    # def get_qcount(self):
    #     ret=self.q_count
    #     ret.index=self.states
    #     return ret

    def learn(self,state, next_state, action, reward):
        state=tuple(int(i) for i in state) # convert to int
        next_state=tuple(int(i) for i in next_state) # convert to int
        qsa = self.get_qvalue(state, action)
        new_q = qsa + self.alpha * (reward + self.gamma * self.get_qvalues(next_state).max() - qsa)
        self.qvalues.loc[str(state), action] = new_q
        # print("updating qlearner at state "+str(state)+" from "+str(qsa)+" to "+str(new_q))
        # self.q_count.loc[str(state)]+=1

    def get_decision(self,state):
        state=tuple(int(i) for i in state) # convert to int
        self.temp=max(0.2,self.temp*0.95)
        probs=boltzmann(self.get_qvalues(state),self.temp)
        self.act = np.random.choice(self.actions,p=probs) # choose an action depending on the probabilities
        assert(self.act in self.actions)
        return self.act

    def get_qvalues(self,state):
        state=tuple(int(i) for i in state) # convert to int
        return self.qvalues.loc[str(state)]

    def get_qvalue(self,state,action):
        state=tuple(int(i) for i in state) # convert to int
        return self.qvalues.loc[str(state),action]

class DQlearner():
    def __init__(self,
              states,
              actions,
              n_features=2,
              alpha=0.01,
              gamma=0.0,
              e_greedy=0.9,
              replace_target_iter=300,
              memory_size=500,
              batch_size=1,
              learn_step=1,
              e_greedy_increment=None,
              output_graph=False):
        self.states=states
        # self.q_count=pd.DataFrame(data={"num":0},index=states)
        self.n_actions = len(actions)
        self.act=0
        self.n_features = n_features
        self.lr = alpha
        self.gamma = gamma
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.unique_id=uuid.uuid4()
        # total learning step
        self.learn_step_counter = 0
        self.learn_step=learn_step
        if self.batch_size>1:
        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self.g_1 = tf.Graph()
        self.sess = tf.Session(graph=self.g_1)
        with self.g_1.as_default():
            self._build_net()
            t_params = tf.get_collection('target_net_params')
            e_params = tf.get_collection('eval_net_params')
            self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

            if output_graph:
                # $ tensorboard --logdir=logs
                # tf.train.SummaryWriter soon be deprecated, use following
                tf.summary.FileWriter("logs/", self.sess.graph)

            self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'+str(self.unique_id)):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'+str(self.unique_id)):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'+str(self.unique_id)):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                # apply DropOut to hidden layer
                self.keep_prob = tf.placeholder(tf.float32)  # DROP-OUT here
                drop_out = tf.nn.dropout(l1, self.keep_prob)  # DROP-OUT here
                self.q_eval = tf.matmul(drop_out, w2) + b2

        with tf.variable_scope('loss'+str(self.unique_id)):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'+str(self.unique_id)):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'+str(self.unique_id)):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'+str(self.unique_id)):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'+str(self.unique_id)):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def train(self,states):
        """
        Pretrain the network to contribute
        """
        for s in states:
        for _ in range(100):
                self.learn(s,[0]*len(s),0,0)
                self.learn(s,[0]*len(s),1,1) # contributing is better
        self.cost_his = []

    def get_decision(self,observation,kp=1.0):
        # to have batch dimension when feed into tf placeholder
        # observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.keep_prob:kp,self.s: [observation]})
            self.act = np.argmax(actions_value)
        else:
            self.act = np.random.randint(0, self.n_actions)
        return self.act

    def learn(self,state,next_state,action,reward,kp=1.0):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')
        if self.batch_size>1:
            self.store_transition(state,action,reward,next_state)
        if self.learn_step_counter % self.learn_step==0:
            if self.batch_size>1:
            # sample batch memory from all memory
            if self.memory_counter > self.memory_size:
                sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            else:
                sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]
            s_=batch_memory[:, -self.n_features:]
            s=batch_memory[:, :self.n_features]
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]
            # print("state "+str(self.get_current_state_int()[0])+" has Q "+str(q_target[batch_index,:])+". updating action "+str(self.act)+" with reward "+str(reward))
            """
            For example in this batch I have 2 samples and 3 actions:
            q_eval =
            [[1, 2, 3],
             [4, 5, 6]]

            q_target = q_eval =
            [[1, 2, 3],
             [4, 5, 6]]

            Then change q_target with the real q_target value w.r.t the q_eval's action.
            For example in:
                sample 0, I took action 0, and the max q_target value is -1;
                sample 1, I took action 2, and the max q_target value is -2:
            q_target =
            [[-1, 2, 3],
             [4, 5, -2]]

            So the (q_target - q_eval) becomes:
            [[(-1)-(1), 0, 0],
             [0, 0, (-2)-(6)]]

            We then backpropagate this error w.r.t the corresponding action to network,
            leave other action as error=0 cause we didn't choose it.
            """
        else:
            s_=[next_state]
            s=[state]
            batch_index = 0
            eval_act_index = action
        ## start training
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                    self.keep_prob:kp,
                self.s_: s_,  # fixed params
                self.s: s,  # newest params
            })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: s,
                                                    self.keep_prob:kp,
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        # update counts
        # self.q_count.loc[[tuple(int(s) for s in state)],'num']+=1

    # def plot_cost(self):
    #     import matplotlib.pyplot as plt
    #     plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')
    #     plt.show()

    def get_qtable(self):
        ## slower
        # qtab=[]
        # for idx,d in self.q_count.iterrows():
        #     qvals=self.sess.run(self.q_eval, feed_dict={self.s: [idx]})[0]
        #     d={"index":idx}
        #     for i in range(len(qvals)):
        #         d.update({i:qvals[i]})
        #     qtab.append(d)
        qtab=pd.DataFrame(data=[self.sess.run(self.q_eval, feed_dict={self.keep_prob:1.0,self.s: [idx]})[0] for idx in self.states],index=self.states,dtype=np.float32)
        # print(qtab)
        return qtab,self.cost_his

    # def get_qcount(self):
        # return self.q_count

    def get_qvalues(self,state):
        qvals=self.sess.run(self.q_eval, feed_dict={self.keep_prob:1.0,self.s: [state]})[0]
        return qvals
