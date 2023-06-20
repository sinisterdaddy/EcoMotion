from Environment import Environment
from DoubleDQN import QNetwork
import numpy as np
import tensorflow as tf
import pandas as pd
import time as tm
from haversine import haversine
# import random
from random import sample
import math

import numpy as np


class ExperienceReplayBuffer:
    def __init__(self, size=50000):
        self.buffer = []
        self.buffer_save = []
        self.buffer_size = size
        self.currents1 = []
        self.currents2 = []
        self.actions = []
        self.rewards = []
        self.nexts1 = []
        self.nexts2 = []
        self.ds = []

    def append(self, exp):
        buffer_len = len(self.buffer)
        exp_len = len(exp)

        if buffer_len + exp_len >= self.buffer_size:
            self.buffer[0:buffer_len + exp_len - self.buffer_size] = []

        buffer_save_len = len(self.buffer_save)
        if buffer_save_len + exp_len >= self.buffer_size:
            self.buffer_save[0:buffer_save_len +
                             exp_len - self.buffer_size] = []

        currents1_len = len(self.currents1)
        if currents1_len + exp_len >= self.buffer_size:
            self.currents1[0:currents1_len + exp_len - self.buffer_size] = []

        currents2_len = len(self.currents2)
        if currents2_len + exp_len >= self.buffer_size:
            self.currents2[0:currents2_len + exp_len - self.buffer_size] = []

        actions_len = len(self.actions)
        if actions_len + exp_len >= self.buffer_size:
            self.actions[0:actions_len + exp_len - self.buffer_size] = []

        rewards_len = len(self.rewards)
        if rewards_len + exp_len >= self.buffer_size:
            self.rewards[0:rewards_len + exp_len - self.buffer_size] = []

        nexts1_len = len(self.nexts1)
        if nexts1_len + exp_len >= self.buffer_size:
            self.nexts1[0:nexts1_len + exp_len - self.buffer_size] = []

        nexts2_len = len(self.nexts2)
        if nexts2_len + exp_len >= self.buffer_size:
            self.nexts2[0:nexts2_len + exp_len - self.buffer_size] = []

        ds_len = len(self.ds)
        if ds_len + exp_len >= self.buffer_size:
            self.ds[0:ds_len + exp_len - self.buffer_size] = []

        self.buffer.extend(exp)
        self.buffer_save.append(exp)
        self.currents1.append(exp[0][0][0])
        self.currents2.append(exp[0][0][1])
        self.actions.append(exp[0][1])
        self.rewards.append(exp[0][2])
        self.nexts1.append(exp[0][3][0])
        self.nexts2.append(exp[0][3][1])
        self.ds.append(exp[0][4])

    def batch(self, num):
        return np.array(sample(self.buffer, num)).reshape((num, 5))


def update_net(trainable_var, sess):
    num = len(trainable_var)
    container = []
    for id, var in enumerate(trainable_var[:num // 2]):
        container.append(trainable_var[id + num // 2].assign(var))
    for contain in container:
        sess.run(contain)


print("------------for tensorflow --------------")
tf.compat.v1.reset_default_graph()

with tf.compat.v1.variable_scope('Qnet'):
    Qnet = QNetwork(state_size=2, action_size=4)

with tf.compat.v1.variable_scope('Targetnet'):
    Targetnet = QNetwork(state_size=2, action_size=4)

print("------------for env & google info--------------")
env = Environment('40.468254,-86.980963', '40.445283,-86.948429')
# The info of the Google Maps route
step_reward, charge_num, SOC, time = env.origine_map_reward()
env.battery_charge()
step_length = 1000  # meters
env.length = 1000 / step_length
print("stride length:", env.length)
learning_rate = 0.0001
# sleep = False

# Use the direction list to iterate over all 8 directions
# for direction in env.directions:
#     # Perform actions or calculations for each direction
#     env.move(direction)
#     env.explore()

print("------------for map --------------")
s = env.start_position
saver = tf.train.Saver(max_to_keep=50)
print("map bound:", env.map_bound)
north = env.map_bound['north']
east = env.map_bound['east']
west = env.map_bound['west']
south = env.map_bound['south']
upper_left = (north, west)
upper_right = (north, east)
lower_left = (south, west)
lower_right = (south, east)
map_height = haversine(upper_left, lower_left)
map_width = haversine(upper_left, upper_right) if haversine(upper_left, upper_right) > haversine(
    lower_left, lower_right) else haversine(lower_left, lower_right)
# print("map height stride:", (north - south) / (map_height * env.length))
# print("map height (km):", map_height)
# print("env.stride height (km):", env.envheightkm)
# print("map width (km):", map_width)
# the number of grid points horizontally
wide_grid_num = map_width / (step_length / 1000)
# the number of grid points vertically
height_grid_num = map_height / (step_length / 1000)
total_point = int(math.ceil(wide_grid_num) * math.ceil(height_grid_num))
print("total grid points:", total_point)
max_train_step = 4 * math.ceil(wide_grid_num) * math.ceil(height_grid_num)
print("Max training steps:", max_train_step)
pre_train_step = max_train_step * 5
print("Pre-train step:", pre_train_step)
s_list = list(s)
print("start position:", s_list)
print("end position:", env.end_position)
replay_buffer = ExperienceReplayBuffer()
init = tf.global_variables_initializer()
trainable_vars = tf.trainable_variables()
print("trainable_vars:", len(trainable_vars))

print("------------Parameters--------------")
path = "./ev/model"
pre_train = pre_train_step  # don't update and train the model within these steps
train_num = 300   # total episode num
max_step = max_train_step
update_freq = 5   # frequency of copying weights from Qnet to Targetnet
batch_num = 32
gamma = 0.9  # discount factor
high_prob = 1
low_prob = 0.1
slope = (high_prob - low_prob) / 20000
# Load Model
pathload = "./ev/Result/47_proceed46/model"
load_model = False
model_num = 27
sleep = False

if load_model:
    high_prob = 0.1

# Initialize constants
tt = 0

# Load the replay buffer
if load_model:
    with open("./ev/buffercurrents1.txt", "r") as f:
        buffercurrents1 = f.readlines()
        buffercurrents1 = [x.strip() for x in buffercurrents1]
    with open("./ev/buffercurrents2.txt", "r") as f:
        buffercurrents2 = f.readlines()
        buffercurrents2 = [x.strip() for x in buffercurrents2]
    with open("./ev/bufferactions.txt", "r") as f:
        bufferactions = f.readlines()
        bufferactions = [x.strip() for x in bufferactions]
    with open("./ev/buffernexts1.txt", "r") as f:
        buffernexts1 = f.readlines()
        buffernexts1 = [x.strip() for x in buffernexts1]
    with open("./ev/buffernexts2.txt", "r") as f:
        buffernexts2 = f.readlines()
        buffernexts2 = [x.strip() for x in buffernexts2]
    with open("./ev/bufferrewards.txt", "r") as f:
        bufferrewards = f.readlines()
        bufferrewards = [x.strip() for x in bufferrewards]
    with open("./ev/bufferds.txt", "r") as f:
        bufferds = f.readlines()
        bufferds = [x.strip() for x in bufferds]

    for current1, current2, action, reward, next1, next2, ds in zip(buffercurrents1, buffercurrents2, bufferactions, bufferrewards, buffernexts1, buffernexts2, bufferds):
        boo = (ds == 'True')
        replay_buffer.append(np.reshape(np.array([[float(current1), float(current2)], int(
            action), float(reward), [float(next1), float(next2)], boo]), [1, 5]))


# Print success message for buffer loading
print("Loading buffer success............................")

# Print the content of the replay buffer
print(replay_buffer.buffer)

# Save training parameters to a CSV file
parameters = [env.map_bound, map_height, map_width, step_length, wide_grid_num, height_grid_num,
              total_point, max_train_step, pre_train_step, learning_rate, step_reward, charge_num, SOC, time]
df_parameters = pd.DataFrame([parameters])
df_parameters.to_csv("./ev/train_parameters.csv", header=["Google map_boundary", "map_height(km)", "map_width(km)", "step_length(m)", "wide_grid_num","height_grid_num", "total_point", "max_train_step in episode", "pre_train_step", "learning_rate", "Google r", "Google charge num", "Google SOC", "Google time"])


# Print sleep message
print("Sleeping for 5 minutes...")
tm.sleep(10)  # Sleep for 10 seconds

# Print start training message
print("Starting training...")

with tf.Session() as sess:
    sess.run(init)
    battery = []
    total_step = 0
    episode_num = 1
    reward_history = []
    e = high_prob

    if load_model:
        print("Loading Model....")
        saver.restore(sess, pathload + "/model-" + str(model_num) + ".ckpt")
        print("Model restored.")

    for episode in range(train_num):  # Number of episodes
        s = env.start_position
        s_list = list(s)
        env.battery_charge()
        soc, charge_num = env.battery_condition()
        print("Current Episode: ", episode_num)
        print("SOC, charge_number: ", soc, charge_num)
        print("Current position: ", env.current_position)
        episode_num += 1
        in_ep_step = 0
        step_buffer = []
        episode_reward = 0
        testt = []  # Try
        random_action = 0
        network_action = 0
        avg_loss = 0
        overq_num = 0  # OVER_QUERY_LIMIT
        unreachable_step_history = []
        loss_history = []
        overq_num_roll = 0

    while in_ep_step <= max_step:  # Max step in one episode
        test = 0  # Try
        is_train = 0  # Try
        is_update = 0  # Try
        Q_value = 0
        in_ep_loss = 0
        update_num = 0

        if np.random.rand(1) < e or (total_step < pre_train and not load_model):
            action = np.random.randint(0, 4)
            test = 1  # Try
        else:
            action = sess.run(Qnet.predict, feed_dict={Qnet.input: [s_list]})[0]
            Q_value = sess.run(Qnet.action, feed_dict={
                            Qnet.input: [s_list]})  # For data analysis
            test = 2  # Try

        if test == 1:
            random_action += 1

        if test == 2:
            network_action += 1

        s1, r, d, charge_num, SOC = env.step(action)
        s = list(s1)
        episode_reward += r

        if env.status_dir_check != 'OVER_QUERY_LIMIT':
            step_buffer.append([s_list, action, r, s, d])
            replay_buffer.append(np.reshape(
                np.array([s_list, action, r, s, d]), [1, 5]))
            in_ep_step += 1
            total_step += 1
            s_list = s

        if env.status_dir_check == 'OVER_QUERY_LIMIT':
            overq_num += 1
            overq_num_roll += 1

        if (total_step > pre_train and env.status_dir_check != 'OVER_QUERY_LIMIT' and len(replay_buffer.buffer) > batch_num) or (load_model and len(replay_buffer.buffer) > batch_num):
            if e > low_prob:
                e -= slope

            ex_batch = replay_buffer.batch(batch_num)
            Qnet_pre = sess.run(Qnet.predict, feed_dict={
                                Qnet.input: np.vstack(ex_batch[:, 3])})
            Targetnet_action = sess.run(Targetnet.action, feed_dict={
                                        Targetnet.input: np.vstack(ex_batch[:, 3])})
            mul = 1 - ex_batch[:, 4]
            y = ex_batch[:, 2] + mul * gamma * \
                Targetnet_action[range(batch_num), Qnet_pre]
            loss = sess.run(Qnet.loss, feed_dict={Qnet.input: np.vstack(
                ex_batch[:, 0]), Qnet.target_y: y, Qnet.a: ex_batch[:, 1]})
            in_ep_loss += loss
            if total_step % 10 == 0:                
                loss_history.append(loss)

            _ = sess.run(Qnet.update, feed_dict={Qnet.input: np.vstack(ex_batch[:, 0]), Qnet.target_y: y, Qnet.a: ex_batch[:, 1]})
            istrain = 1

            if total_step % update_freq == 0:
                update_net(trainable_vars, sess)
                isupdate = 1
                update_num += 1

            if d == True:
                print("True exame")
                print(abs(env.next_position[0] - env.end_position[0]))
                print("stride_height: ", env.stride_height)
                print(abs(env.next_position[1] - env.end_position[1]))
                print("stride_wide: ", env.stride_wide)
                print("-----###")
                print("Success")
                if in_ep_step > max_step:  # we don't want too many steps
                    step_buffer = []
                if in_ep_step > 0:
                    avg_loss = in_ep_loss / in_ep_step
                # We don't penalize transition for real_reward (compare with Google route)
                real_reward = episode_reward + 0.1 * \
                    (in_ep_step - env.unreach_position_num) - 1
                real_r_nofail = real_reward + env.unreach_position_num
                print("Step to reach end: ", in_ep_step)
                battery.append([charge_num, SOC])  # SOC is the current one
                time = env.time
                history = [episode + 1, in_ep_step, time, episode_reward, real_reward, real_r_nofail, charge_num, SOC,env.unreach_position_num, d, random_action / (random_action + network_action), avg_loss, overq_num, loss_history, step_buffer]

                if episode == 0:
                    
                    df = pd.DataFrame([history])
                    df.to_csv("./ev/result.csv", header=["episode", "step", "time", "reward", "reward_notrain", "reward_nofail", "charge_num" "SOC", "unreach_position", "Reach", "Random_a", "Avg_loss", "overQuery_num", "Loss history", "Step history"])
                    tt += 1
                elif episode > 0:                    
                    with open('./ev/result.csv', 'a') as f:
                        df = pd.DataFrame([history])
                        df.to_csv(f, header=False)
                        print("Total time: ", time)
                        print("Number of failed steps: ", env.unreach_position_num)

                        env.time = 0  # Reset time
                        env.unreach_position_num = 0  # Reset the number of unreachable positions

                if in_ep_step < total_point / 2:
                    print("Google route info >>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                    print("Real reward, charge_num, SOC, time: ",
                        step_reward, charge_num, SOC, time)
                    print("Our route step is less than", str(
                        total_point / 2), "  >>>>>>>>>>>>>>>>>>>")
                    print("Summary: ", real_r_nofail, charge_num, SOC, time)

                break
            
            if d == False and in_ep_step == max_step:
                # step_buffer = []
                if in_ep_step > 0:
                    avg_loss = in_ep_loss / in_ep_step   # average loss in one episode
                # print("testt [test, action, istrain, isupdate]: ", testt)  # try
                print("Failed")
                time = env.time
                # We don't penalize transition for real_reward (compare with google route) and we don't count the goal
                real_reward = episode_reward + 0.1 * \
                    (in_ep_step - env.unreach_position_num)
                real_r_nofail = real_reward + env.unreach_position_num
                history = [episode + 1, in_ep_step, time, episode_reward, real_reward, real_r_nofail, charge_num, SOC,
                        env.unreach_position_num, d, random_action / (random_action + network_action), avg_loss, overq_num, loss_history, step_buffer]
                if episode == 0:
                    df = pd.DataFrame([history])
                    df.to_csv("./ev/result.csv", header=["episode", "step", "time", "reward", "reward_notrain", "reward_nofail", "charge_num",
                            "SOC", "unreach_position", "Reach", "Random_a", "Avg_loss", "overQuery_num", "Loss history", "Step history"])
                elif episode > 0:
                    with open('./ev/result.csv', 'a') as f:
                        df = pd.DataFrame([history])
                        df.to_csv(f, header=False)
                # env.current_position = env.start_position  # reset the start position to origin
                # s = env.start_position  # reset the start position to origin
                env.time = 0  # reset time
                # env.charge_num = 0  # reset the charging number
                env.unreach_position_num = 0
                break
                # total_step = total_step + 1
            # total_step = total_step + 1
            if overQ_num_roll > 50:
                overQ_num_roll = 0
                print("Sleeping within episode for 60 min")
                tm.sleep(3600)
            
        if d == True and in_ep_step < 60 and episode > 10 and episode % 1 == 0 or (load_model == True and d == True):
            j = episode + 1
            save_path = saver.save(sess, path+"/model-"+str(j)+".ckpt")
            print("Saved model with step less than 60")

        print("Last position: ", env.current_position)
        print("Destination: ", env.end_position)
        # print("env.next_position", env.next_position)
        # print("env.stride height(km): ", env.envheightkm)
        print("env.stridebound a, b", env.stridebounda, env.strideboundb)
        env.current_position = env.start_position  # reset the start position to origin
        # print("current position: ", env.current_position)
        s = env.start_position  # reset the start position to origin
        s_list = list(s)
        ss, nn = env.battery_condition()
        print("SOC, charge_number: ", ss, nn)
        env.charge_num = 0  # reset the charging number
        # total_step = total_step + 1
        reward_history.append(episode_reward)
        env.battery_charge()
        
        ###################### save the repaly buffer ############################

        with open("./ev/buffercurrents1.txt", "w") as ff:    # save the replay buffer
            for s in replay_buffer.currents1:
                ff.write(str(s) + "\n")
        with open("./ev/buffercurrents2.txt", "w") as mm:    # save the replay buffer
            for s in replay_buffer.currents2:
                mm.write(str(s) + "\n")
        with open("./ev/bufferactions.txt", "w") as gg:    # save the replay buffer
            for s in replay_buffer.actions:
                gg.write(str(s) + "\n")
        with open("./ev/bufferrewards.txt", "w") as hh:    # save the replay buffer
            for s in replay_buffer.rewards:
                hh.write(str(s) + "\n")
        with open("./ev/buffernexts1.txt", "w") as ii:    # save the replay buffer
            for s in replay_buffer.nexts1:
                ii.write(str(s) + "\n")
        with open("./ev/buffernexts2.txt", "w") as kk:    # save the replay buffer
            for s in replay_buffer.nexts2:
                kk.write(str(s) + "\n")
        with open("./ev/bufferds.txt", "w") as jj:    # save the replay buffer
            for s in replay_buffer.ds:
                jj.write(str(s) + "\n")
        
         ###################### Save the replay buffer ############################
        replay_buffer.save("./ev/replay_buffer.txt")
        ###################### Save the replay buffer ############################

        if (total_step > 450 and total_step % 120 == 0) or sleep:
            print("Sleeping now for 20 min")
            time.sleep(1200)
            print("-------------------------------------------------------------------------------")

        print("______________Episode End___________________")
print("------------------End----------------")
