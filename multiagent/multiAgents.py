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
from pacman import SCARED_TIME

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
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        currentFood = currentGameState.getFood()
        currentCapsules = currentGameState.getCapsules()

        "*** YOUR CODE HERE ***"
        '''
            things to consider
            1)
            ghost positions - how close
            if scared_time > distance_to_ghost:
                closer is better
            else:
                stay away

            2)
            capsules

            3)
            food positions


            4) 
            score

        '''

        GHOSTMAX = 20
        THREAT_RANGE = 6
        CAPSULEMAX = 25
        FOODMAX = 10

        pointScore = successorGameState.getScore()

        ghostScore = 0
        closestGhostDistance = MinDistance(newPos, [g.getPosition() for g in newGhostStates])
        for ghostState in newGhostStates:
            ghost_distance = manhattanDistance(newPos, ghostState.getPosition())
            scared_time = ghostState.scaredTimer
            
            if scared_time > ghost_distance:
                ghostScore += 2*(GHOSTMAX - ghost_distance)
            
            elif ghost_distance == 1:
                ghostScore += -3*GHOSTMAX

            elif ghost_distance < THREAT_RANGE:
                ghostScore += min(0, -GHOSTMAX -ghost_distance)


        capsuleScore = 0
        if closestGhostDistance < SCARED_TIME:
            #print "getting capsule vibes"
            capsule_distance = MinDistance(newPos, currentCapsules)
            capsuleScore = max(0, CAPSULEMAX - capsule_distance)


        foodScore = 0
        foodMinDistance = float('inf')
        for food in currentGameState.getFood().asList():
            food_distance = manhattanDistance(newPos, food)

            if food_distance == 0:
                foodScore = FOODMAX
                break
            elif food_distance < foodMinDistance:
                foodScore   =  5 - food_distance
                foodMinDistance = food_distance



        #print "pointScore:", pointScore, " ghostScore:", ghostScore, " foodScore:", foodScore, " capsuleScore:", capsuleScore

        score = pointScore + ghostScore + capsuleScore + foodScore 


        return score

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
        
        PACMAN = 0

        def MaxFunction(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(PACMAN)
            score = float('-inf')
            for action in actions:
                score = max(score, MinFunction(gameState.generateSuccessor(PACMAN, action), 1, depth))

            return score

        def MinFunction(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            score = float('inf')

            if agentIndex + 1 < gameState.getNumAgents():
                for action in actions:
                    score = min(score, MinFunction(gameState.generateSuccessor(agentIndex, action), agentIndex+1, depth))
            else:
                for action in actions:
                    score = min(score, MaxFunction(gameState.generateSuccessor(agentIndex, action), depth-1))

            return score

        #MinMax(state)
        #    return argmax(action): MinValue(state->action)
        
        minmax_action = None
        score = float('-inf')

        actions = gameState.getLegalActions(PACMAN)
        for action in actions:
            actionScore = MinFunction(gameState.generateSuccessor(PACMAN, action), 1, self.depth)
            if actionScore > score:
                score = actionScore
                minmax_action = action

        return minmax_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        PACMAN = 0

        def MaxFunction(gameState, alpha, beta, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(PACMAN)
            score = float('-inf')
            for action in actions:
                score = max(score, MinFunction(gameState.generateSuccessor(PACMAN, action), alpha, beta, 1, depth))
                if score > beta:
                    return score
                alpha = max(alpha, score)

            return score

        def MinFunction(gameState, alpha, beta, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            actions = gameState.getLegalActions(agentIndex)
            score = float('inf')

            if agentIndex + 1 < gameState.getNumAgents():
                for action in actions:
                    score = min(score, MinFunction(gameState.generateSuccessor(agentIndex, action), alpha, beta, agentIndex+1, depth))
                    if score < alpha:
                        return score
                    beta = min(beta, score)
            else:
                for action in actions:
                    score = min(score, MaxFunction(gameState.generateSuccessor(agentIndex, action), alpha, beta, depth-1))
                    if score < alpha:
                        return score
                    beta = min(beta, score)
            return score


        alphabeta_action = None
        score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        actions = gameState.getLegalActions(PACMAN)
        for action in actions:
            actionScore = MinFunction(gameState.generateSuccessor(PACMAN, action), alpha, beta, 1, self.depth)
            if actionScore > score:
                score = actionScore
                alphabeta_action = action

            if score >= beta:
                return alphabeta_action
            alpha = max(alpha, score)

        return alphabeta_action

        util.raiseNotDefined()

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

        PACMAN = 0

        def ExpectiMiniMaxFunction(gameState, agentIndex, depth):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)

            nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()
            if nextAgentIndex == 0:
                depth -= 1

            actions = gameState.getLegalActions(agentIndex)

            if agentIndex == PACMAN:
                # return argmax action: Expectimax(s->a)
                score = float('-inf')
                for action in actions:
                    score = max(score, ExpectiMiniMaxFunction(gameState.generateSuccessor(agentIndex, action), nextAgentIndex, depth ))

            else:
                # return sum[ P(r) * Expectimax(s->r) ]
                score = 0
                probability = 1.0 / len(actions)
                for action in actions:
                    actionScore = ExpectiMiniMaxFunction(gameState.generateSuccessor(agentIndex, action), nextAgentIndex, depth)
                    score += probability * actionScore

            return score
        #Expectimax(state)
        #    return argmax(action): MinValue(state->action)
        

        DEBUG_actions = {}

        expectimax_action = None
        score = float('-inf')

        actions = gameState.getLegalActions(PACMAN)
        for action in actions:
            actionScore = ExpectiMiniMaxFunction(gameState.generateSuccessor(PACMAN, action), 1, self.depth)
            DEBUG_actions[action] = actionScore
            if actionScore > score:
                score = actionScore
                expectimax_action = action

        for action, score in DEBUG_actions.iteritems():
            print action, score, ", ",
        print ""

        return expectimax_action

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """

    "*** YOUR CODE HERE ***"
    def EvaluateAction(currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newCapsules = successorGameState.getCapsules()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        currentFood = currentGameState.getFood()
        currentCapsules = currentGameState.getCapsules()

        MAX = 100
        MIN = -100

        GHOSTMAX = 1.2
        THREAT_RANGE = 8
        GHOST_GAMMA = 0.9
        CAPSULEMAX = 4.5
        FOODMAX = 1.5

        if successorGameState.isWin():
            return MAX
        if successorGameState.isLose():
            return MIN

        pointScore = successorGameState.getScore()

        ghostScore = 0
        closestGhostDistance = MinDistance(newPos, [g.getPosition() for g in newGhostStates])
        for ghostState in newGhostStates:
            ghost_distance = manhattanDistance(newPos, ghostState.getPosition())
            #ghost_distance = DistanceWithGamma(newPos, ghostState.getPosition(), GHOST_GAMMA)

            scared_time = ghostState.scaredTimer
            
            if scared_time > ghost_distance:
                ghostScore += 2*(GHOSTMAX - ghost_distance)
            
            elif ghost_distance == 1:
                ghostScore += -3*GHOSTMAX

            elif ghost_distance < THREAT_RANGE:
                ghostScore += min(0, -GHOSTMAX -ghost_distance)


        capsuleScore = 0
        if closestGhostDistance < SCARED_TIME:
            #print "getting capsule vibes"
            capsule_distance = MinDistance(newPos, currentCapsules)
            capsuleScore = max(0, CAPSULEMAX - capsule_distance)


        foodScore = 0
        foodMinDistance = float('inf')
        for food in currentGameState.getFood().asList():
            food_distance = manhattanDistance(newPos, food)

            if food_distance == 0:
                foodScore = FOODMAX
                break
            elif food_distance < foodMinDistance:
                foodScore   =  5 - food_distance
                foodMinDistance = food_distance

        score = pointScore + ghostScore + capsuleScore + foodScore 

        if action == Directions.STOP:
            score -= pointScore

        #print action, ":: pointScore:", pointScore, " ghostScore:", ghostScore, " foodScore:", foodScore, " capsuleScore:", capsuleScore, "total:", score

        return score


    "*** Function begins here ***"
    
    if currentGameState.isWin():
        return random.randrange(50, 100)
    if currentGameState.isLose():
        return float('-inf')

    PACMAN = 0

    pacmanPos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsules = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()


    # Consider the available actions from this state
    # The best state is the one that produces the best successor
    bestNextScore = 0
    actions = currentGameState.getLegalActions(PACMAN)
    for action in actions:
        bestNextScore = max(bestNextScore, EvaluateAction(currentGameState, action))

    bestNextScore*=0.001

    # food incentive
    # need an extra incentive to get rid of the food
    foodMax = float(len(food.data) * len(food.data[0]))
    foodRemainingPenalty = float(-foodMax) / (len(food.asList()) + 1)
    #foodRemainingPenalty = len(food.asList()) / float(foodMax) 
    #foodRemainingPenalty = (foodMax - len(food.asList()) )/ foodMax
    foodRemainingPenalty *= -1

    foodDistanceMax = MaxDistance(pacmanPos, food.asList())
    foodDistanceScore = 0
    for dot in food.asList():
        foodDistanceScore += manhattanDistance(pacmanPos, dot)
    foodDistanceScore = foodDistanceScore / (max(1, foodDistanceMax * len(food.asList())))


    currentScore = currentGameState.getScore()
    currentScore *= 0.01

    #score = currentScore + foodRemainingPenalty + foodDistanceScore + bestNextScore
    #print pacmanPos, "points:", currentScore, " foodpenalty:", foodRemainingPenalty, " bestaction:", bestNextScore, " total", score

    #import pdb;pdb.set_trace()

    #foodRemaining = len(food.asList())
    scaredGhostPositions = [ghost.getPosition() for ghost in ghostStates if ghost.scaredTimer > 0]

    roundTripScore = FoodToFoodTripCost(pacmanPos, food.asList() + capsules + scaredGhostPositions)
    roundTripScore *= -0.01




    score = roundTripScore + foodRemainingPenalty + bestNextScore
    score += random.random()*0.01 if score == 0 else random.random()/(score*score)


    #print pacmanPos, "foodremaining:", foodRemainingPenalty, "bestnext", bestNextScore, "score:", score
    #import pdb;pdb.set_trace()


    return score
# Abbreviation
better = betterEvaluationFunction



'''Some extra functions for convenience
'''

def MinDistance(a, listofb, distanceFunction=manhattanDistance, returnTuple=False):
    closest = None
    minDistance = float('inf')
    for b in listofb:
        distance = distanceFunction(a, b)
        if distance < minDistance:
            minDistance = distance
            closest = b

    if returnTuple:
        return closest, minDistance
    else:
        return minDistance

def MaxDistance(a, listofb, distanceFunction=manhattanDistance, returnTuple=False):
    farthest = None
    maxDistance = 0
    for b in listofb:
        distance = distanceFunction(a, b)
        if distance > maxDistance:
            maxDistance = distance
            farthest = b

    if returnTuple:
        return farthest, maxDistance
    else:
        return maxDistance


def DistanceWithGamma(a, b, gamma):
    real_distance = manhattanDistance(a,b)
    falloff_distance = 0
    for d in range(int(real_distance)):
        falloff_distance += gamma
        gamma *= gamma

    return falloff_distance


def FoodToFoodTripCost(position, foodPositions):
    """This is my final heuristic function
    
    Starting at pacman's position (position), this 
    function carves a path through all food tiles, 
    choosing the closest unvisited tile every time.

    For each step, the manhattam distance is added
    to a running total, (hcost).

    In order to keep the heuristic cost from getting
    too high, the length of the largest move (freeMove)
    will be subtracted from (hcost) at the end.
    """
    subgoals = list(foodPositions)
    hcost = 0 # running total cost
    freeMove = 0 # i will allow the largest move for free...

    #... unless there is only one food tile
    if len(subgoals) < 2:
        return util.manhattanDistance(position, subgoals[0])

    while subgoals:
        closest, distance = MinDistance(position, subgoals, returnTuple=True)
        hcost += distance
        position = closest
        subgoals.remove(closest)
        if freeMove < distance:
            freeMove = distance

    #hcost -= freeMove
    return hcost