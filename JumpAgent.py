from collections import deque
import random
import tensorflow as tf
import os

def add_Dimension(state, action, reward, next_state, done):
    states = tf.expand_dims(state, axis=0)
    actions = tf.expand_dims(action, axis=0)
    rewards = tf.expand_dims(reward, axis=0)
    next_states = tf.expand_dims(next_state, axis=0)
    dones = tf.expand_dims(done, axis=0)
    return states, actions, rewards, next_states, dones
def as_Tensors(states, actions, rewards, next_states, dones):
    state = tf.constant(states, dtype=tf.float32)
    action = tf.constant(actions, dtype=tf.float32)
    reward = tf.constant(rewards, dtype=tf.float32)
    next_state = tf.constant(next_states, dtype=tf.float32)
    done = tf.constant(dones, dtype=tf.float32)
    return state, action, reward, next_state, done

def ActionToJump(action):
    # take in action [0, 0] and return True if we're jumping and False if we're not
    if action[0] == 1:
        return True
    elif action[1] == 1:
        return False

def GetReward(done):
    if done is False:
        return 1
    else:
        return -100                                                

class JumpAgent:
    def __init__(self, trainer, max_memory=100000, batch_size=1000, epsilon=0):
        self.n_games = 0
        self.epsilon = epsilon
        self.memory = deque(maxlen=max_memory)
        self.trainer = trainer
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def TrainLongMemory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states, actions, rewards, next_states, dones = as_Tensors(states, actions, rewards, next_states, dones)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def TrainShortMemory(self, state, action, reward, next_state, done):
        state, action, reward, next_state, done = as_Tensors(state, action, reward, next_state, done)
        state, action, reward, next_state, done = add_Dimension(state, action, reward, next_state, done)
        self.trainer.train_step(state, action, reward, next_state, done)

    def GetAction(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0,0]
        prediction = [0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,1)
            final_move[move] = 1
        else:
            stateTensor = tf.constant([state], dtype=tf.float32)
            # stateTensor = tf.expand_dims(stateTensor, axis=0)
            prediction = self.trainer.model.predict(stateTensor)[0]
            move = tf.argmax(prediction).numpy()
            print(prediction)
            print(move)
            final_move[move] = 1
        return final_move, prediction[0], prediction[1]

    def SaveModel(self, directory="./model", filename="model.pth"):
        if not os.path.exists(directory):
            os.mkdir(directory)
        filename = os.path.join(directory, filename)
        self.trainer.model.save(filename)