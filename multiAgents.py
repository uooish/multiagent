# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        distance = []
        foodList = currentGameState.getFood().asList()
        pacmanPos = list(successorGameState.getPacmanPosition())

        # print 'successorGameState\n',successorGameState
        # print 'newPos',newPos
        # print 'newFood',newFood
        # print 'ghostState',newGhostStates[0].getPosition()

        currentScore = 0
        ghostDistences = []
        for g in newGhostStates:
            ghostDistences += [manhattanDistance(g.getPosition(), newPos)]
        foodList = newFood.asList()
        foodDistences = []
        for f in foodList:
            foodDistences += [manhattanDistance(newPos, f)]
        if len(foodDistences) > 0 and min(foodDistences) > 0:
            foodScore = 2.25 / (min(foodDistences) * min(foodDistences))
        else:
            foodScore = 0
        if min(ghostDistences) > 3.0:
            ghostScore = 3
        elif min(ghostDistences) < 1.1:
            ghostScore = -100
        else:
            ghostScore = min(ghostDistences)

        currentScore += foodScore + ghostScore + successorGameState.getScore()
        return currentScore


        # print 'foodDistence', foodDistences
        # print 'ghostDistance', ghostDistences
        # print 'foodkist',foodList
        # print'score', currentScore


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

#value function return (action, value) tuple, getAction function just need return action

        result = self.value(gameState, 0)
        return result[0]

    def value(self, gameState, depth):
        numAgents = gameState.getNumAgents()
        curAgent = depth % numAgents
        actions = gameState.getLegalActions(curAgent)


        if (depth == self.depth * numAgents) or len(actions) == 0:
            return (None, self.evaluationFunction(gameState))
        depth += 1
        if (curAgent == 0):
            bestAction = (None, float("-inf"))
            for action in actions:
                newValue = self.value(gameState.generateSuccessor(curAgent, action), depth)
                if newValue[1] > bestAction[1]:
                    bestAction = (action,newValue[1])
            return bestAction

        else:
            bestAction = (None, float("inf"))
            for action in actions:
                newValue = self.value(gameState.generateSuccessor(curAgent, action), depth)
                if newValue[1] < bestAction[1]:
                    bestAction = (action, newValue[1])
            return bestAction



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        result = self.value(gameState, 0, float("-inf"), float("inf"))
        return result[0]

    def value(self, gameState, depth, alpha, beta):
        numAgents = gameState.getNumAgents()
        curAgent = depth % numAgents
        actions = gameState.getLegalActions(curAgent)


        if (depth == self.depth * numAgents) or len(actions) == 0:
            return (None, self.evaluationFunction(gameState))

        depth += 1
        if (curAgent == 0):
            bestAction = (None, float("-inf"))
            for action in actions:
                newValue = self.value(gameState.generateSuccessor(curAgent, action), depth, alpha, beta)
                if newValue[1] > bestAction[1]:
                    bestAction = (action, newValue[1])
                if bestAction[1] > beta:
                    return bestAction
                else:
                    alpha = max(alpha, bestAction[1])
            return bestAction

        else:
            bestAction = (None, float("inf"))
            for action in actions:
                newValue = self.value(gameState.generateSuccessor(curAgent, action), depth, alpha, beta)
                if newValue[1] < bestAction[1]:
                    bestAction = (action, newValue[1])
                if bestAction[1] < alpha:
                    return bestAction
                else:
                    beta = min(beta, bestAction[1])

            return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(gameState, depth, agent):
            numAgents = gameState.getNumAgents()
            curAgent = depth % numAgents
            actions = gameState.getLegalActions(agent)
            if gameState.isLose() or gameState.isWin() or (depth == 0):
                return self.evaluationFunction(gameState), None

            nextAgent = (agent + 1) % numAgents
            depth -= 1
            if agent == 0:
                bestValue = (float("-inf"),None)
                for action in actions:
                    newValue = expectimax(gameState.generateSuccessor(agent, action), depth, nextAgent)
                    if newValue[0] > bestValue[0]:
                        bestValue = (newValue[0], action)
                return bestValue

            else:
                values = []
                bestValue = (0.0, None)
                for action in actions:
                    newValue = expectimax(gameState.generateSuccessor(agent, action), depth, nextAgent)
                    values.append(newValue[0])
                bestValue = (sum(values) / len(values), action)
                return bestValue

        numAgents = gameState.getNumAgents()
        result = expectimax(gameState, self.depth * numAgents, self.index)
        return result[1]




def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    currentPosition = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    currentScore = 0
    ghostDistences = []
    for g in GhostStates:
        ghostDistences += [manhattanDistance(g.getPosition(), currentPosition)]
    foodList = newFood.asList()
    foodDistences = []
    for f in foodList:
        foodDistences += [manhattanDistance(currentPosition, f)]
    if len(foodDistences) > 0 and min(foodDistences) > 0:
        foodScore = 2.25 / (min(foodDistences) * min(foodDistences))
    else:
        foodScore = 0
    if min(ghostDistences) > 4.0:
        ghostScore = 3
    elif min(ghostDistences) < 1.1:
        ghostScore = -100
    else:
        ghostScore = min(ghostDistences)

    currentScore += foodScore + ghostScore + currentGameState.getScore()

    return currentScore

# Abbreviation
better = betterEvaluationFunction

