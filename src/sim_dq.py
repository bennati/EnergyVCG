from Supervisor import BaseSupervisor
from DecisionLogic import BaseDecisionLogic
from MeasurementGen import *
from utils import *
import itertools
import numpy as np
import math
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim
import uuid

def renormalize(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]

# class MeasurementGenUniformDQ(BaseMeasurementGen):
#     def __init__(self, *args, **kwargs):
#         super().__init__()
#         self.n1=kwargs["n1"]
#         self.n2=kwargs["n2"]
#         assert(self.n1>=0)
#         assert(self.n2>self.n1)

#     def get_measurements(self,population,timestep):
#         """
#         Returns a list of dictionaries containing the measurements: the state of each agent at the current timestep
#         """
#         vals=[np.random.uniform(max(1,self.n1),self.n2) for _ in population]
#         costs=[np.random.uniform(max(1,self.n1),self.n2) for _ in population]
#         # thresh=max(1,int(sum(vals)*np.random.uniform(0,1)))
#         thresh=len(population) #np.random.randint(1,3)
#         assert(thresh<=sum(vals))
#         ret=[{"value":v,"cost":c,"timestep":timestep,"agentID":i,"threshold":thresh} for i,(v,c) in enumerate(zip(vals,costs))]
#         return ret

class DecisionLogicSupervisorDQ(BaseDecisionLogic):
    """
    Returns a constant decision
    """
    def get_decision(self,perceptions):
        decs=[a.get_decision() for a in self.model.schedule.agents]
        idxs=[a.unique_id for a in self.model.schedule.agents]
        tmp1=pd.DataFrame(data={"action":decs,"agentID":idxs})
        tmp=pd.merge(pd.DataFrame(perceptions),tmp1,on=["agentID"])
        #print(decs)
        self.act=[{"contribution":(r[1]["value_raw"] if r[1]["action"] else np.nan),"cost":(r[1]["cost_raw"] if r[1]["action"] else np.nan),"privacy":(1 if r[1]["action"] else 0),"agentID":r[1]["agentID"],"contributed":r[1]["action"],"timestep":r[1]["timestep"],"threshold":r[1]["threshold"]} for r in tmp.iterrows()]
        return self.act

np.random.seed(1)
tf.set_random_seed(1)

# Deep Q Network off-policy
class DecisionLogicDQ(BaseDecisionLogic):
    def __init__(self,model,alpha=0.01,gamma=0.0,training=False):
        super().__init__(model)
        possible_values=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        possible_costs=list(range(max(1,self.model.model.measurement_fct.n1),self.model.model.measurement_fct.n2)) # TODO binarize a continuous range
        self.states=list(itertools.product(possible_values,possible_costs)) # all possible states
        self.actions=[0,1]
        self.act=1
        self.init_(self.states,self.actions,gamma=gamma,alpha=alpha)
        if training:
            self.train()

    def init_(self,
              states,
              actions,
              n_actions=2,
              n_features=2,
              alpha=0.01,
              gamma=0.0,
              e_greedy=0.9,
              replace_target_iter=300,
              memory_size=500,
              batch_size=1,
              e_greedy_increment=None,
              output_graph=False):
        self.q_count=pd.DataFrame(data={"num":0},index=states)
        self.n_actions = len(actions)
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
                self.q_eval = tf.matmul(l1, w2) + b2

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

    def train(self):
        """
        Pretrain the network to contribute
        """
        low=max(1,self.model.model.measurement_fct.n1)
        high=self.model.model.measurement_fct.n2
        for _ in range(100):
            v=np.random.uniform(low,high)
            c=np.random.uniform(low,high)
            self.learn((v,c),[0,0],0,0)
            self.learn((v,c),[0,0],1,1) # contributing is better

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def get_decision(self, perception):
        # to have batch dimension when feed into tf placeholder
        # observation = observation[np.newaxis, :]
        observation=self.get_current_state()
        self.get_decision_(observation)

    def get_decision_(self,observation):
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: [observation]})
            self.act = np.argmax(actions_value)
        else:
            self.act = np.random.randint(0, self.n_actions)
        return self.act

    def feedback(self,perceptions,reward):
        assert(reward["agentID"]==self.model.unique_id)
        current=self.get_current_state()
        self.learn(current,[0,0],self.act,reward["reward"])

    def learn(self,state,next_state,action,reward):
        self.store_transition(state,action,reward,next_state)
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s: batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

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

        # train eval network
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        # update counts
        self.q_count.loc[[tuple(int(s) for s in state)],'num']+=1

    # def plot_cost(self):
    #     import matplotlib.pyplot as plt
    #     plt.plot(np.arange(len(self.cost_his)), self.cost_his)
    #     plt.ylabel('Cost')
    #     plt.xlabel('training steps')
    #     plt.show()

    def get_qtable(self):
        # print(self.losses)
        qtab=[]
        for idx,d in self.q_count.iterrows():
            qvals=self.sess.run(self.q_eval, feed_dict={self.s: [idx]})[0]
            qtab.append({"index":idx,0:qvals[0],1:qvals[1]})
        qtab=pd.DataFrame(qtab).set_index("index")
        # print(qtab)
        return qtab

    def get_qcount(self):
        return self.q_count

    def get_current_state(self):
        return (self.model.current_state["perception"]["value_raw"],self.model.current_state["perception"]["cost_raw"])

    def get_current_state_int(self):
        return (self.model.current_state["perception"]["value"],self.model.current_state["perception"]["cost"])
