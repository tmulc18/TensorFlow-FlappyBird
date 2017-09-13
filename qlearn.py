#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

import json

import tensorflow as tf


from ops import conv2D, dense

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 3200. # timesteps to observe before training
EXPLORE = 3000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.1 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
CONTINUE_TRAIN = False

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

def tf_buildmodel(x,reuse=None):
    x = tf.nn.relu(conv2D(x,shape=[8,8,32],name='1',stride=[1,4,4,1],reuse=reuse))
    x = tf.nn.relu(conv2D(x,shape=[64,4,4],name='2',stride=[1,2,2,1],reuse=reuse))
    x = tf.nn.relu(conv2D(x,shape=[64,3,3],name='3',stride=[1,1,1,1],reuse=reuse))
    x = tf.reshape(x,[-1,300])
    x = dense(x,512,name='1',reuse=reuse)
    x = dense(x,2,name='prj',activation=None,reuse=reuse)
    return x


def trainNetwork(model,args):
    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()
    # TODO: implement queue in TF

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = skimage.color.rgb2gray(x_t)
    x_t = skimage.transform.resize(x_t,(80,80))
    x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    #print (s_t.shape)

    #In Keras, need to reshape
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4
    # tf_s_t = tf.placeholder(dtype=tf.float32,shape=(None, s_t.shape[0], s_t.shape[1], s_t.shape[2])) #$$$$
    

    if args['mode'] == 'Run':
        OBSERVE = 999999999    #We keep observe, never train
        epsilon = FINAL_EPSILON
        print ("Now we load weight")
        model.load_weights("model2.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)
        print ("Weight load successfully")    
    else:                       #We go to training mode
        OBSERVE = OBSERVATION
        epsilon = INITIAL_EPSILON

    #(s,a)
    image_input = tf.placeholder(dtype=tf.float32,shape=(None, 80, 80, 4)) #1*80*80*4
    Q_model = tf_buildmodel(image_input,reuse=None) #$$$

    global_step = tf.Variable(0.,trainable=False)
    tf_targets = tf.placeholder(dtype=tf.float32,shape=(BATCH,ACTIONS))
    # TODO: zero out the loss for the action that isn't picked? 
    tf_actions = tf.placeholder(dtype=tf.float32,shape=(BATCH,ACTIONS))
    tf_loss = tf.reduce_mean(tf.squared_difference(Q_model*tf_actions, tf_targets))
    tf_opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(tf_loss,global_step=global_step)

    #summaries
    Q_sa_max = tf.reduce_mean(tf.reduce_max(Q_model,axis=1))

    Q_summary = tf.summary.scalar('Q',Q_sa_max)
    loss_summary = tf.summary.scalar('Loss',tf_loss)
    merged = tf.summary.merge_all()

    #writer for summary
    writer = tf.summary.FileWriter('logdir/',tf.get_default_graph())

    #init ops
    init = tf.global_variables_initializer()

    #save model
    saver = tf.train.Saver()

    #session
    sess = tf.Session()
    if args['mode'] == 'Run' or CONTINUE_TRAIN == True:
        saver.restore(sess,'logdir/model')
    sess.run(init)


    t = 0
    while (True):
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0
        a_t = np.zeros([ACTIONS])
        #choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = sess.run([Q_model],feed_dict = {image_input : s_t}) ##input a stack of 4 images, get the prediction
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[action_index] = 1

        #We reduced the epsilon gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        #run the selected action and observed next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        # TODO implement below x_t1 transformations in TF
        #tf.placeholder()
        #tf.expand_dims(,axis=0)
        #tf.image.rgb_to_grayscale()
        #tf.image.resize_images() #maybe crop here instead
        ##tf.image.resize_image_with_crop_or_pad()

        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        #only train if done observing
        if t > OBSERVE:
            #sample a minibatch to train on
            minibatch = random.sample(D, BATCH)

            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

            #Now we do the experience replay
            #process entire minibatch at once for (s',a')
            state_t1 = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
            minibatch = np.array(minibatch)
            state_t1 = np.vstack(minibatch[:,3])

            Q_sa_prime = sess.run(Q_model,feed_dict = {image_input : state_t1})
            action_array_t = np.zeros((BATCH,ACTIONS))

            #fill in the rest of the targets for the chosen action at time t (s,a)
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]
                action_t = minibatch[i][1]   #This is action index
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                terminal = minibatch[i][4]
                # if terminated, only equals reward

                inputs[i:i + 1] = state_t    #I saved down s_t

                #targets[i] = model.predict(state_t)  # Hitting each buttom probability
                #Q_sa = model.predict(state_t1)

                # TODO: Why is targets being assigned here?
                # targets[i],Q_sa = sess.run([Q_model,Q_model],
                #                     feed_dict = {image_input : state_t,image_input : state_t1}) #$$$

                if terminal:
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa_prime[i])

                #set action
                action_array_t[i,action_t] = 1

            # targets2 = normalize(targets)
            #loss += model.train_on_batch(inputs, targets)
            sess.run(tf_opt,feed_dict = {image_input : inputs, tf_targets : targets, tf_actions : action_array_t})

            if t % 500 == 0:
                summary,gs = sess.run([merged,global_step],feed_dict = {image_input : inputs, tf_targets : targets, tf_actions : action_array_t})
                writer.add_summary(summary,gs)
            #loss += sess.run(tf_loss,feed_dict = {image_input : inputs, tf_targets : targets})


        s_t = s_t1
        t = t + 1

        # save progress every 10000 iterations
        if t % 1000 == 0:
            print("Now we save model")
            saver.save(sess, 'logdir/model') 

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("************************")

def playGame(args):
    model = None #$$$
    trainNetwork(model,args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)
    main()
