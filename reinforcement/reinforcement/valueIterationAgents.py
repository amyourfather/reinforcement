# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        allstates = self.mdp.getStates()
        #possibleaction = self.mdp.getPossibleActions(self)
        #mdp.getTransitionStatesAndProbs(state, action)
        #mdp.getReward(state, action, nextState)
        #mdp.isTerminal(state)
        print(allstates)
        #print(1)
        for i in range(0,self.iterations):
            newvalue = util.Counter()
            for state in allstates:
                action = self.getAction(state)
                if action != None:
                    newvalue[state] = self.getQValue(state, action)
                #print(action)
            self.values = newvalue
        "*** YOUR CODE HERE ***"


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #Q = sigma(s')(T(s,a,s')[R(s,a,s') + rV(s')])
        Q = 0
        allstateandprob = self.mdp.getTransitionStatesAndProbs(state, action)

        for stateandprob in allstateandprob:
            tempstate = stateandprob[0]
            prob = stateandprob[1]
            reward = self.mdp.getReward(state, action, tempstate)
            value = self.getValue(tempstate)
            Q += prob * (reward + self.discount * value)

        return Q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):#if is terminal state, return None
            return None
        else:
            allaction = self.mdp.getPossibleActions(state)
            maxvalue = self.getQValue(state, allaction[0])
            maxaction = allaction[0]
            for action in allaction:
                tempvalue = self.getQValue(state, action)
                if maxvalue < tempvalue:
                    maxvalue = tempvalue
                    maxaction = action
        return maxaction
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        """
        allstates = self.mdp.getStates()
        #possibleaction = self.mdp.getPossibleActions(self)
        #mdp.getTransitionStatesAndProbs(state, action)
        #mdp.getReward(state, action, nextState)
        #mdp.isTerminal(state)
        #print(allstates)
        #print(1)
        x = 0
        while x < self.iterations:
            newvalue = util.Counter()
            for state in allstates:
                action = self.getAction(state)# change it
                if action != None:
                    newvalue[state] = self.getQValue(state, action)
                x += 1
                if x >= self.iterations:
                    return
                #print(action)
            self.values = newvalue
            #print("value is :", self.values)
        """
        temp = 0
        allstates = self.mdp.getStates()
        while temp < self.iterations:
          for state in allstates:
            newvalue = util.Counter()
            for action in self.mdp.getPossibleActions(state):
              newvalue[action] = self.getQValue(state, action) #put all new value to newvalue with action
            self.values[state] = newvalue[newvalue.argMax()] #only update the max one
            temp += 1 #every time update then counter ++
            if temp >= self.iterations: #when enough update return
              return

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdpstate = self.mdp.getStates()
        #Compute predecessors of all states.
        predecessors = {} #use dictionary to contain predecessors
        for state in mdpstate: #predecessors has same amount as mdp states
            predecessors[state] = set()

        #Initialize an empty priority queue.
        pqueue = util.PriorityQueue()

        for s in mdpstate: #For each non-terminal state s, do
            #1 find highest q value
            q = util.Counter()
            for action in self.mdp.getPossibleActions(s):
                q[action] = self.getQValue(s, action)
                allstateandprob = self.mdp.getTransitionStatesAndProbs(s, action)
                for stateandprob in allstateandprob:
                    tempstate = stateandprob[0]
                    prob = stateandprob[1]
                    if prob != 0:# if = 0 then we skip this state
                        predecessors[tempstate].add(s)#add pre state to predecessors
            #check non terminal or not
            if not self.mdp.isTerminal(s):
                #find highest q
                max_q = q[q.argMax()]
                diff = abs(self.values[s] - max_q) #difference between the current value of s in self.values and the highest Q-value across all possible actions from s
                pqueue.update(s, -diff) #s into the priority queue with priority -diff
        #For iteration in 0, 1, 2, ..., self.iterations - 1, do
        for i in range(self.iterations):
            if pqueue.isEmpty():
                return
            pops = pqueue.pop()
            #Update the value of s (if it is not a terminal state) in self.values
            if not self.mdp.isTerminal(pops):
                q = util.Counter()
                for action in self.mdp.getPossibleActions(pops):
                    q[action] = self.getQValue(pops, action)
                self.values[pops] = q[q.argMax()]
            for p in predecessors[pops]:
                q_p = util.Counter()
                for paction in self.mdp.getPossibleActions(p):
                    q_p[paction] = self.computeQValueFromValues(p, paction)
                maxqp = q_p[q_p.argMax()]
                diff = abs(self.values[p] - maxqp)
                if diff > self.theta:
                    pqueue.update(p, -diff)
