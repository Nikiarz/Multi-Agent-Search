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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"
        #after getting the base code check for closer food  and fewer dots left ke behtare 
        # give bonus age resid be ghaza

        score = successorGameState.getScore() 
        foodListSucc = newFood.asList()
        foodListCurr = currentGameState.getFood().asList()
        if foodListSucc: #food
            minFoodDist = min(manhattanDistance(newPos, f) for f in foodListSucc)
            score += 4.0 / (minFoodDist + 1.0)
        score -= 1.0 * len(foodListSucc)
        if len(foodListSucc) < len(foodListCurr): 
            score += 0.8
        #capsul
        capsSucc = successorGameState.getCapsules()
        capsCurr = currentGameState.getCapsules()
        if len(capsSucc)<len(capsCurr):
            score +=15.0
        #ghost
        #closer to scared ghost behtare vali nearer live ghost bad taresh mikone
        for  gState, scared in zip(newGhostStates, newScaredTimes):
            gPos =  gState.getPosition()
            d = manhattanDistance(newPos, gPos)
            if scared >0:
                score += 6.0 / (d + 1.0)
            else:
                if d == 0:
                    return float('-inf')
                if d <= 1:
                    score -= 25.0
                score -= 3.0 / d
        if action == Directions.STOP:
            score -= 5.0
        return score
    
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"


        #Count Pacman + all ghosts (used to know when to wrap back to Pacman)
        #age move nabod treat as terminal and evaluate
        #depth stays teh same until wewrap back to pacman
        #you alternate max at Pacman nodes and min at ghost nodes, bad ezafe kon depth only after all agents move,
        #stop at win/lose or the depth limit to evaluate, and at the root you pick the action whose subtree has the highest minimax value.

        numAgents= gameState.getNumAgents()
        def minimaxValue(state: GameState, agentIndex: int, depth: int):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            actions = state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            nextAgent = agentIndex + 1
            nextDepth = depth
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1
            
            if agentIndex ==0:
                best = -float('inf') #initiate to max pas miay migi negative infinity 
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a)
                    best = max(best, minimaxValue(succ, nextAgent, nextDepth))
                return best
            else: 
                best = float('inf')
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a)
                    best = min(best, minimaxValue(succ, nextAgent, nextDepth))
            return best
        
#baraye root call choose action ke highst minimaxo mide

        bestScore = -float('inf')
        bestAction = Directions.STOP
        #Evaluate that branch: next agent is Ghost 1, depth starts at 0 
        for a in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0,a)
            score = minimaxValue(succ, 1, 0)
            if score > bestScore:
                bestScore, bestAction = score, a
        return bestAction            



        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    #add 2 bounds alpha ke the best score pacman can garantee so far va 
    #beta ke mishe the best score the ghost can garantee 
    #age emtiaz ghoste mosavi ya kamtar as pac beshe i can prune

    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState): 
        
        #you’re telling Python (and the reader) “the parameter called gameState should be an instance of the GameState class.
        #this is the variable name, the actual object being passed in when Pacman
        #calls getAction. Inside your function, you always work with gameState.
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        def alphabeta(state: GameState, agentIndex: int, depth: int, alpha: float, beta:float):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            
            actions= state.getLegalActions(agentIndex)
            if not actions:
                return self.evaluationFunction(state)
            #bad az in bayad next agent / depth bookeeping
            nextAgent = agentIndex + 1
            nextDepth = depth 
            if nextAgent == numAgents:
                nextAgent = 0
                nextDepth = depth + 1
            if agentIndex == 0:
                value = -float('inf')
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a) 
                    value = max(value, alphabeta(succ, nextAgent, nextDepth, alpha, beta))
                    if value > beta :
                        return value
                    alpha = max(alpha, value)
                return value
            else:
                value = float('inf')
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a)
                    value = min(value, alphabeta(succ, nextAgent, nextDepth, alpha, beta))
                    if value < alpha:
                        return value
                    beta = min(beta, value)
                return value
        

        alpha, beta = -float('inf'), float('inf')
        bestScore = -float('inf')
        bestAction = Directions.STOP

        for a in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, a)
            score = alphabeta(succ, 1, 0, alpha, beta)
            if score > bestScore:
                bestScore, bestAction = score, a
            if bestScore > beta:
                break
            alpha = max(alpha, bestScore)

        return bestAction

    # for a in gameState.getLegalActions(0):
    #     succ = gameState.generateSuccessor(0, a)
    #     score = alphabeta(succ, 1, 0, alpha, beta)
    #     if score > bestScore:
    #         bestScore, bestAction = score, a
    #     if bestScore > beta:              
    #         break
    #     alpha = max(alpha, bestScore)

    # return bestAction   
        





        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
#Pacman (agent 0) is still a MAX node (choose the action with the highest value).
#Ghosts (agent ≥1) are not MIN anymore


    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        
        #go track the best score found so far and track the aciton that gives that best score
        # use gamestate the instance nort the Gamestate which is theclass
        # always use gameState (lowercase) inside getAction
        #0 = Pacman, ≥1 = ghosts


        numAgents = gameState.getNumAgents()
        def expectimax(state: GameState, agentIndex: int, depth: int):
            if state.isWin() or state.isLose() or depth ==self.depth:
                return self.evaluationFunction(state)
            actions = state.getLegalActions(agentIndex)
            if not actions:
                # no legal moves (corner case)
                return self.evaluationFunction(state)
            
            nAgent = agentIndex +1
            nDepth=depth
            #pacman max beshe 
            if nAgent == numAgents:
                nAgent=0
                nDepth=depth+1

            if agentIndex == 0:
                value = -float('inf')
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a)
                    value = max(value, expectimax(succ, nAgent, nDepth))
                return value
            #ghost chance 
            #uniform avg
            else:
                #ghost pick actions uniformly at random expected value

                total = 0
                prob = 1 /len(actions)
                for a in actions:
                    succ = state.generateSuccessor(agentIndex, a)
                    total += prob * expectimax(succ, nAgent, nDepth)
                return total
            
        bestS = -float('inf')
        bestA = Directions.STOP
        # vaghti loop over mikoni pacman mitrone az current state dorost kone
        for a in gameState.getLegalActions(0):
            succ = gameState.generateSuccessor(0, a)
            score = expectimax(succ, 1, 0)
            if score >bestS:
                bestS, bestA = score , a 
        return bestA





        




        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <To guide Pacman I added rewards for being closer to food and capsules,
subed points for having more food/capsules left, reduced the score for being near active ghosts
and rewarded chasing scared ghosts when they are edible. This balances safety with faster
food collection.>
    """
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood().asList()
    # food = currentGameState.getFood()
    capsules= currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates() 
    scaredTimes = [g.scaredTimer for g in ghostStates]
    score= currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    #.asList(), iterating the Grid later crashed

    if food:
        dists = [manhattanDistance(pos, f)for f in food]
        minFoodD = min (dists)
        avgFoodD = sum(dists)/len(dists)
        score +=5 / ( minFoodD + 1 )
        score += 1 / (avgFoodD + 1)     
        score -= 2 * len(food)
    
    else:# age poli namond bonus bede
        score += 100

    if capsules:
        capDists = [manhattanDistance(pos,c) for c in capsules]
        score += 2.0 / (min(capDists) + 1.0)    # nudge toward nearest capsule
    score -= 4.0 * len(capsules) 

    for g, scared in zip(ghostStates, scaredTimes):
        gpos = g.getPosition()
        d = manhattanDistance(pos, gpos)  

        if scared >0:
            score += 6 / (d + 1)
            if d <= scared:
                score = score +10
        else:
            if d==0:
                return float('-inf')
            if d <=1:
                score -=30
            score -= 6/d
    return score      








    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
