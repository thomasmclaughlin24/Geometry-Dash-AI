import numpy
import win32com.client as w3c
import time
import tensorflow as tf
from Utils import *
import GameOverTrainer
from JumpTrainer import JumpTrainer, LinearQNet
from JumpAgent import JumpAgent, ActionToJump, GetReward
import matplotlib.pyplot as plt
import matplotlib.animation as animation

width = 1920
height = 1080
shell = w3c.Dispatch("Wscript.Shell")
success = shell.AppActivate("Geometry Dash")
images = []
got = GameOverTrainer.GameOverTrainer()
got.Train()
done = False
old_images = []
old_image_stack = []
jump_reward = 0
no_jump_reward = 0

time.sleep(3)

def Generate_Model():
    model = LinearQNet((5120,), 128, 2)
    return model

def ProcessReward(action):
    global old_image_stack
    old_images.pop(0)
    old_image_stack = np.ravel(numpy.vstack(old_images)).reshape(5120)
    reward = GetReward(done)
    agent.TrainShortMemory(old_image_stack, action, reward, image_stack, done)
    agent.remember(old_image_stack, action, reward, image_stack, done)

# def Update()

if(success):
    agent = JumpAgent(JumpTrainer(Generate_Model()))
    old_images = np.zeros((4,1024)).tolist()
    image = None
    images = np.zeros((4,1024)).tolist()
    action = None
    actions = []
    plt.ion()
    figure, axis = plt.subplots()
    figure.set_figwidth(4)
    figure.set_figheight(4)
    bar_graph = axis.bar((0,1), (jump_reward, no_jump_reward))
    while True:
        image = GetImage()
        image = image.reshape(1, -1)
        gameover = tf.argmax(got.model.predict(image, verbose=0)[0]).numpy()
        image = image[0]
        images.append(image)
        if gameover == 1:
            shell.SendKeys(" ")
            done = True
            agent.n_games += 1
        else:
            done = False
        if len(images) > 5:
            old_images.append(images[0])
            images.pop(0)
        old_image_stack = np.ravel(numpy.vstack(old_images))
        image_stack = np.ravel(numpy.vstack(images)).reshape(5120)
        if len(old_images) > 5:
            ProcessReward(actions[0])
            actions.pop(0)
            if done:
                for action in actions:
                    old_images.append(images[-1])
                    ProcessReward(action)
                agent.TrainLongMemory()
        action, jump_reward, no_jump_reward = agent.GetAction(image_stack)
        actions.append(action)
        jump = ActionToJump(action)
        if jump:
            shell.SendKeys(" ")

        # time.sleep(1/60)