from tensorforce.environments import Environment
import tensorforce as tensorforce
import numpy as np
from solver import Solver
import os
import shutil

class NetworkEnv(Environment):
    """
        Action space:
        NN can control the epsilon value at one given tube
        as a value between 0-10

        Observation space:
        First attempt - NN can see the radius of all tubes
        i.e. an array of n number of tubes

        Reward:
        +1 for any action that keeps the flux through our selected tube
        between q_min and q_max which is hardcoded somewhere

        Episode End:
        Episode ends if
        1) flux in selected tube is < 0.5*q_min or 1.5*q_max
        2) episode length is greater than 5000

    """

    def __init__(self, max_solver_step, number_steps_execution, q_off):
        self.number_steps_execution = number_steps_execution
        self.solver_step = 0
        self.max_solver_step = max_solver_step
        self.episode_number = 0
        self.q_off = q_off

        print('init done!')


        self.make_folder()
        if os.path.exists('episodes/'):
            shutil.rmtree('episodes/')
        self.solver = Solver()
        self.start_class()




    def make_folder(self):
        if not os.path.exists('episodes/%g/'%self.episode_number):
            os.makedirs('episodes/%g/'%self.episode_number)
        self.folder = 'episodes/%g/'%self.episode_number


    def start_class(self):
        print('starting class')
        self.solver_step = 0
        self.number_of_tubes, self.init_radius, self.q_init, self.close_tube, self.optimize_tube = self.solver.set_up()
        self.radius = self.init_radius
        self.observation_value = self.q_init
        self.q_min = 14.303     # Chosen number to fit the test, should be provided as an argument
        self.q_max = 15.8088    # Chosen number to fit the test, should be provided as an argument
        print('initial flow = ', self.q_init)
        print('finished starting class')

    def states(self):
        print('states')
        return dict(type='float', shape=(self.number_of_tubes,))


    def actions(self):
        """
            Each tube can have a feedback factor, epsilon, 
            between -100 and 100. 2 is the ideal for the test.
        """

        return dict(type='float',
                    shape=(self.number_of_tubes,),
                    min_value=-100,
                    max_value=100)

    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)

    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    # Optional additional steps to close environment
    def render(self):
        pass


    def reset(self):
        print('reset, finished episode number = ', self.episode_number)
        if self.episode_number > 0:
            outfile = open('episodes/data.txt', 'a')
            outfile.write('%f\t %f\n' %(self.total_flow, self.total_reward))
            outfile.close()
        self.total_flow = 0
        self.total_reward = 0
        self.episode_number += 1
        self.rounds = 0
        self.min_flow_constant = 1
        if (self.episode_number % 25) == 0 or self.episode_number == 1:
            self.make_folder()

        state = self.init_radius
        state = np.reshape(state, (self.number_of_tubes,))
        self.state = state

        self.start_class()
        return state


    def compute_reward(self):
        # set up 2nd degree polynomial, func, to give reward
        q_mid = 0.5*(self.q_max + self.q_min)
        prefac = (10 - 1.1) / ((self.q_min - self.q_max)*(q_mid**2 - self.q_min**2)/(self.q_max**2 - self.q_min**2) + (q_mid - self.q_min) )
        prefac2 = (self.q_min - self.q_max) / (self.q_max**2 - self.q_min**2)
        func = prefac * prefac2 * (self.observation_value**2 - self.q_min**2) + prefac * (self.observation_value - self.q_min) + 1.1

        if self.observation_value > self.q_min and self.observation_value < self.q_max:
            """
                reward is a log function divided by the maximum change in epsilon at a single tube between
                timesteps. I.e. small changes in epsilon is highly rewarded
            """
            reward = np.log(func) / max(abs(self.old_pass_function - self.present_pass_function))[0]#
        else:
            reward = -1

        return reward


    def write_states(self, pass_actions):
        if not os.path.exists(self.folder+'flow.txt'):
            write_states_file = open(self.folder+"flow.txt", "w")
            write_states_file.write(' '.join("%g\t"%i  for i in range(len(self.flow)))+'\n')
            write_states_file.close()
        if not os.path.exists(self.folder+'state.txt'):
            write_states_file = open(self.folder+"state.txt", "w")
            write_states_file.write(' '.join("%g\t"%i  for i in range(len(pass_actions)))+'\n')
            write_states_file.close()
        write_states_file = open(self.folder+"flow.txt", "a")
        write_states_file.write(' '.join("%f\t"%float(i)  for i in self.flow)+'\n')
        write_states_file.close()
        write_states_file = open(self.folder+"state.txt", "a")
        write_states_file.write(' '.join("%f\t"%float(i)  for i in pass_actions)+'\n')
        write_states_file.close()       


    def execute(self, actions):

        print('##########################')

        pass_function = np.reshape(actions, (len(actions), 1))
        pass_actions = np.zeros(shape=(self.number_of_tubes, 1))

        pass_actions[:] = pass_function[:]
        self.reward_action = pass_function

        if self.rounds > 1:
            # the chosen epsilon is used to calculate a gradient for the epsilon used in the numerics
            pass_actions[:] = self.old_actions[:] - 0.05 * (self.old_actions[:] - pass_actions[:])

        self.old_actions = pass_actions


        if self.solver_step == 0:
            self.old_pass_function = pass_function
            self.present_pass_function = pass_function
        else:
            self.present_pass_function = pass_function

        for _ in range(self.number_steps_execution):

            # solve eq and retreive state and
            self.solver_step += 1

            self.radius, self.length, self.q_measure, self.flow = self.solver.evolve(pass_actions)


        self.rounds += 1
        

        if (self.episode_number % 25) == 0 or self.episode_number == 1:

            self.write_states(pass_actions)
            
            if (self.rounds % 25) == 0 or self.rounds == 1:

                self.solver.plot(self.folder, pass_actions, 0)
                self.solver.plot(self.folder, pass_actions, 1)

                
        calc_list = np.delete(self.flow, [1,5,7,11,13,17,19,23])

        self.observation_value = sum(calc_list)
        

        reward = self.compute_reward()

        if self.solver_step != 0:
            self.old_pass_function = pass_function

        self.total_flow += sum(calc_list)#self.q_measure
        self.total_reward += reward

        print('reward = ', reward)
        print('##########################')


        next_state = np.reshape(self.radius, (self.number_of_tubes, ))


        if self.solver_step >= self.max_solver_step:
            done = True
        elif reward < -1E10:
            done = True
        else:
            done = False


        return next_state, done, reward
