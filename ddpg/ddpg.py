""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
import gym_pdsystem
from gym import wrappers
import tflearn
import argparse
import pprint as pp

from replay_buffer import ReplayBuffer

from datetime import datetime


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_shape, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_shape = action_shape
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        #tank_max_loads =  np.array([100., 200, 100., 800., 200.])
        #inputs = tf.multiply(inputs, tank_max_loads**(-1))
        net = tflearn.fully_connected(inputs, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 150)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 50)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        #net = tf.multiply(net, 10.0)

        # a = tf.Print(net, [net], message="This is net: ")
        # b = tf.add(a,a)
        # init_op = tf.initialize_all_variables()
        # sess = tf.InteractiveSession()
        # sess.run(init_op)
        # b.eval()

        #b.eval()
        #tf.Print(net)
        w_init = tflearn.initializations.variance_scaling(factor=2.0, mode='FAN_IN', uniform=False, seed=1, dtype=tf.float32) #(minval=-0.003, maxval=0.003) # try variance_scaling and xavier
        #out = tflearn.fully_connected(
            #net, self.a_dim, activation='softmax', weights_init=w_init)

        outs = []
        #print(self.a_dim)
        for i in range(int((self.a_dim) / (self.s_dim+1))): # - 1 if we consider also staying in the depot
                truck = tflearn.fully_connected(net,self.s_dim+1, activation='softmax', weights_init=w_init)                          
                outs.append(truck)

        out = tf.concat(outs, axis=1)
        #out = tflearn.activations.sigmoid(out)
        #out = tf.multiply(out, self.action_bound)

        #out2 = tflearn.fully_connected(
            #out, self.a_dim, activation='softmax', weights_init=w_init)

        #Filter
        #print(out.shape)
        #out = tf.reshape(out, self.a_shape )
        #print(out)
        #print(tf.reduce_max(out, reduction_indices=[1]))
        #out = (out == tf.reduce_max(out, reduction_indices=[1]))
        #print(out)
        #print(out.shape)
        #out = out.astype(int)
        #tf.reduce_max(x, reduction_indices=[1])
        #out = tf.reshape(out, (-1,self.a_dim) )
        #print(out.shape)

        #print(out)
        # Scale output to -action_bound to action_bound
        #print()
        #scaled_out = tf.multiply(out, self.action_bound)
        #print(scaled_out)
        scaled_out = out

        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 200)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(inputs, 100)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        #net = tflearn.fully_connected(inputs, 50)
        #net = tflearn.layers.normalization.batch_normalization(net)
        #net = tflearn.activations.relu(net)


        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 50)
        t2 = tflearn.fully_connected(action, 50)
        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)


    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()
   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)

    logwriter = tf.summary.FileWriter(logdir, sess.graph)


    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    #tflearn.is_training(True)
    sim_id = int(args['sim_id'])

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()
             ##OUTPUT DATA TO FILE
            if i % 100 == 0:
                file = 'sim{:d}/test-sim{:d}.txt'.format(sim_id,sim_id)
                with open(file,'ab') as f:
                     np.savetxt(f, [s], fmt='%d', delimiter=',')
            if i % 10000 == 0:
         
                save_path = saver.save(sess, 'sim{:d}/model_sim{:d}-episode{:d}.ckpt'.format(sim_id,sim_id,i))    

           
            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))  +  actor_noise() #(1. / (1. + i)) * np.random.randint(1,10**3) # actor_noise()

            s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            


            if(j == (int(args['max_episode_len'])-1) ):
                terminal = True


            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / (float(j+1))
                })

                writer.add_summary(summary_str, i)
                logwriter.add_summary(summary_str, i)

                writer.flush()


                #reward_summary = tf.summary.scalar('Average Reward', ep_reward)
                #qvalue_summary = tf.summary.scalar('Average Max Q-value', ep_ave_max_q / float(j+1))

                #print(i, reward_summary.eval())
                #print(i, reward_summary)

 
                #logwriter.add_summary(reward_summary.eval(), i)
                #logwriter.add_summary(qvalue_summary.eval(), i)


                print('| Reward: {:.6f} | Episode: {:d} | Episode length: {:d} | Qmax: {:.4f}'.format(ep_reward, \
                        i, j, (ep_ave_max_q / float(j+1))))

                with open('sim{:d}/info-sim{:d}.txt'.format(sim_id,sim_id),'ab') as f:
                     np.savetxt(f, [np.array([i,ep_reward, ep_ave_max_q / float(j+1),j])], fmt='%.6f', delimiter=',')    
                break


# ===========================
#   Agent Testing
# ===========================

def test(sess, env, args, actor, critic):



    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sim_id = int(args['sim_id'])

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'], sess.graph)
    #saver = tf.train.import_meta_graph('sim{:d}/model_sim{:d}.ckpt.meta'.format(sim_id,sim_id))

    #saver.restore(sess, 'sim{:d}/model_sim{:d}.ckpt'.format(sim_id,sim_id))

    #actor.eval()
    #critic.eval()

    # Initialize target network weights
    #actor.update_target_network()
    #critic.update_target_network()

    # Initialize replay memory
    #replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)


    for i in range(int(args['max_test_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_test_episode_len'])):

            if args['render_env']:
                env.render()
             ##OUTPUT DATA TO FILE
            if i % 1 == 0:
                with open('sim{:d}-test/testing-sim{:d}.txt'.format(sim_id,sim_id),'ab') as f:
                     np.savetxt(f, [s], fmt='%d', delimiter=',')    

           
            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) # +  actor_noise() #(1. / (1. + i)) * np.random.randint(1,10**3) # actor_noise()

            s2, r, terminal, info = env.step(a[0])

            s = s2
            ep_reward += r

            if(j == (int(args['max_episode_len'])-1) ):
                terminal = True


            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / (float(j+1))
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:.6f} | Episode: {:d} | Episode length: {:d} | Qmax: {:.6f}'.format(ep_reward, \
                        i, j, (ep_ave_max_q / float(j+1))))

                with open('sim{:d}-test/testing-info-sim{:d}.txt'.format(sim_id,sim_id),'ab') as f:
                     np.savetxt(f, [np.array([i,ep_reward, ep_ave_max_q / float(j+1), j])], fmt='%.6f', delimiter=',')    
                break



def main(args):

    training = int(args['train'])
    sim_id = int(args['sim_id'])


    with tf.Session() as sess:
        if training == 1:

           
            env = gym.make(args['env'])
            env._max_episode_steps = int(args['max_episode_len'])

            np.random.seed(int(args['random_seed']))
            tf.set_random_seed(int(args['random_seed']))
            env.seed(int(args['random_seed']))

            state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_shape = (env.action_space.shape[0] ,  env.action_space.shape[1])
            action_dim = action_shape[0] * action_shape[1]
            action_bound = env.action_space.high.reshape(env.action_space.high.shape[0] * env.action_space.high.shape[1] )
            # Ensure action bound is symmetric
            #assert (env.action_space.high == -env.action_space.low)

            actor = ActorNetwork(sess, state_dim, action_dim, action_shape, action_bound,
                                 float(args['actor_lr']), float(args['tau']),
                                 int(args['minibatch_size']))

            critic = CriticNetwork(sess, state_dim, action_dim,
                                   float(args['critic_lr']), float(args['tau']),
                                   float(args['gamma']),
                                   actor.get_num_trainable_vars())
            
            actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

            if args['use_gym_monitor']:
                if not args['render_env']:
                    env = wrappers.Monitor(
                        env, args['monitor_dir'], video_callable=False, force=True)
                else:
                    env = wrappers.Monitor(env, args['monitor_dir'], force=True)

            train(sess, env, args, actor, critic, actor_noise)

        else:
            #tflearn.is_training(False)
            #saver.restore(sess, "/tmp/my_model_final.ckpt")
            env = gym.make(args['env'])
            env._max_episode_steps = int(args['max_test_episode_len'])

            np.random.seed(int(args['random_seed']))
            tf.set_random_seed(int(args['random_seed']))
            env.seed(int(args['random_seed']))

            state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
            action_shape = (env.action_space.shape[0] ,  env.action_space.shape[1])
            action_dim = action_shape[0] * action_shape[1]
            action_bound = env.action_space.high.reshape(env.action_space.high.shape[0] * env.action_space.high.shape[1] )
            # # Ensure action bound is symmetric
            # #assert (env.action_space.high == -env.action_space.low)

            actor = ActorNetwork(sess, state_dim, action_dim, action_shape, action_bound,
                                  float(args['actor_lr']), float(args['tau']),
                                  int(args['minibatch_size']))

            critic = CriticNetwork(sess, state_dim, action_dim,
                                    float(args['critic_lr']), float(args['tau']),
                                    float(args['gamma']),
                                    actor.get_num_trainable_vars())

            #actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
            #tf.reset_default_graph()
            #env = tf.get_variable("env")
            #actor = tf.get_variable("actor")
            #critic = tf.get_variable("critic")

            #saver = tf.train.Saver()

            #print(actor)
            saver = tf.train.import_meta_graph('sim{:d}/model_sim{:d}.ckpt.meta'.format(sim_id,sim_id))

            saver.restore(sess, 'sim{:d}/model_sim{:d}.ckpt'.format(sim_id,sim_id))
           # env = tf.get_variable("env")
           # actor = tf.get_variable("actor")
           # critic = tf.get_variable("critic")
            print("before test")
            #print(actor)


            test(sess, env, args, actor, critic)#, actor, critic, actor_noise)
            print("hello")

        if args['use_gym_monitor']:
            env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    parser.add_argument('--train', help='wheter or not it is a training run (False implies testing)', default=0)


    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=1.0)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--sim-id', help='simulation identifier', default=21)


    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1000000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=100)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    # test parameters
    parser.add_argument('--max-test-episodes', help='max num of episodes to do while testing', default=10)
    parser.add_argument('--max-test-episode-len', help='max length of 1 episode during testing', default=15)



    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
