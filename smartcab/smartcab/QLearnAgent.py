import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import os

cur_dir = os.path.dirname(__file__)
path = os.path.abspath(cur_dir)
path = os.path.join(path, '../sim-results')
fullpath = os.path.join(path, "q_table.txt")
file = open(fullpath, 'a')

class QLearn(Agent):
    def __init__(self, epsilon=0.1, alpha=0.2, gamma=0.9): # values taken from https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
        self.Q = {} # initialize Q matrix
        self.epsilon = epsilon 
        self.alpha = alpha # learning rate
        self.gamma = gamma # discount factor
        self.valid_actions = Environment.valid_actions

        parameters= "epsilon = " + str(epsilon) + ", alpha = " + str(alpha) + ", gamma = " + str(gamma) + "\n"
        
        #file.write("\n" + parameters)

    def getQ(self, state, action):
        key = (state, action)
        return self.Q.get(key, None)

    def learnQ(self, state, action, reward, max_val):
        old_Q = self.Q.get((state, action), None)

        if old_Q is None:
            self.Q[(state, action)] = 0
            # initializing with higher constant to make optimistic agent: all possible actions will yield excellent rewards
            # incentivise exploratory-leaning agent 
        else:
            self.Q[(state, action)] = old_Q + self.alpha * (max_val - old_Q)

    def chooseBest(self, state):
        if random.random() < self.epsilon: # random exploration case
            action = random.choice(self.valid_actions)
        else:
            q = [self.getQ(state, x) for x in self.valid_actions]
            max_q = max(q)
            count = q.count(max_q)

            # in a case where there's multiple best actions
            if count > 1:
                best_actions = [i for i in range(len(self.valid_actions)) if q[i] == max_q]
                idx = random.choice(best_actions)
            else:
                idx = q.index(max_q)

            action = self.valid_actions[idx]

        return action

    def learn(self, state, action, state2, reward):
        argmax_q = max([self.getQ(state2, a) for a in self.valid_actions])
        if argmax_q is None:
            argmax_q = 0.0
        self.learnQ(state, action, reward, reward + self.gamma*argmax_q)
        file.write("state = {}, action = {}, reward = {}\n".format(state, action, reward))


class QLearningAgent(Agent):
    """An agent that learns to drive in the smartcab world using Q-Learning."""

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

    def params(self, epsilon, alpha, gamma):
        self.QLearn = QLearn(epsilon, alpha, gamma);

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Define state using parameters
        self.state = (self.next_waypoint, inputs['light'], inputs['oncoming'], inputs['left'])
 
        # Select action according to your policy
        action = self.QLearn.chooseBest(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        new_inputs = self.env.sense(self)
        next_state = (self.next_waypoint, new_inputs['light'], new_inputs['oncoming'], new_inputs['left'])

        # TODO: Learn policy based on state, action, reward
        self.QLearn.learn(self.state, action, next_state, reward)

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)


def run(epsilon, alpha, gamma):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(QLearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials
    gridSearch = a.params(epsilon, alpha, gamma)

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    file.write("Success Rate:{}%\n".format(e.successRate))

if __name__ == '__main__':
    epsilon = [0, 0.01, 0.05, 0.1, 0.2]
    alpha = [0.1, 0.01, 0.05, 0.1, 0.2]
    gamma = [0, 0.2, 0.4, 0.6, 0.8]

    """for eps in epsilon:
        for a in alpha:
            for g in gamma:
                run(epsilon=eps, alpha=a, gamma=g)"""

    run(epsilon=0.1, alpha=0.1, gamma=0.2)
