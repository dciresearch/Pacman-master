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
from model import fetch_model#, commonModel
from featureBasedGameState import FeatureBasedGameState
from math import sqrt, log
from pacman import GameState

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

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
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

        return successorGameState.getScore()+1

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

    def is_terminal(self, depth, gameState):
        return gameState.isLose() or gameState.isWin() or depth == self.depth


def getLegalActionsNoStop(index, gameState):
    possibleActions = gameState.getLegalActions(index)
    if Directions.STOP in possibleActions:
        possibleActions.remove(Directions.STOP)
    return possibleActions

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def minimax(self, agent, depth, gameState):
        """
        Create recursive function for minimax evaluation

        Implementation details:
        - agent = index of game agent where 0 is for pac-man and >=1 for ghosts
        - depth = and increasing variable that tracks current depth of search tree

        - use getLegalActionsNoStop(agent, gameState) to get possible actions for agent
        - use self.is_terminal(depth, gameState) to break recursion
          and return self.evaluationFunction(gameState)
        - use gameState.getNumAgents() to check whether every agent has made its move
        - use gameState.generateSuccessor(agent, action) to get next gameState
        """
        "*** YOUR CODE HERE ***"
        pass


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState.

          Here are some method calls that might be useful when implementing minimax.

          is_terminal(depth, gameState):
            Checks if search should be terminated due to
            depth limit or winning/losing condition

          getLegalActionsNoStop(agentIndex, gameState):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.minimax(0, 0, gameState.generateSuccessor(0, action)) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphabeta(self, agent, depth, gameState, alpha, beta):
        """
        Using the same functions and methods implement Alpha-Beta version of minimax recursion
        """
        "*** YOUR CODE HERE ***"
        pass
        
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        alpha = -999999
        beta = 999999
        action_scores = [self.alphabeta(0, 0, gameState.generateSuccessor(0, action), alpha, beta) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax(self, agent, depth, gameState):
        """
        The same thing as minimax except:
        - For Pac-Man (agent = 0) should return maximum of leaf evaluation scores
        - For ghosts (agent >= 1) should return the expected value of leaf evaluation scores
        - assume that action probability is uniform
        """
        "*** YOUR CODE HERE ***"
        pass
        
    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        possibleActions = getLegalActionsNoStop(0, gameState)
        action_scores = [self.expectimax(0, 0, gameState.generateSuccessor(0, action)) for action
                         in possibleActions]
        max_action = max(action_scores)
        max_indices = [index for index in range(len(action_scores)) if action_scores[index] == max_action]
        chosenIndex = random.choice(max_indices)
        return possibleActions[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: This evaluation function -
      1) Rewards not dying (of course!) - this logic is captured by the game score itself
      2) Gives a high reward for eating up food pellets
      3) Gives a small reward for being closer to the food

    Useful functions and methods:
    - currentGameState.getPacmanPosition()
    - currentGameState.getFood().asList()
    - currentGameState.getScore()
    """
    # Imports
    from random import randint
    from util import manhattanDistance as dist

    "*** YOUR CODE HERE ***"
    pass


class MCTSAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn = 'betterEvaluationFunction', numTraining = '0', isReal = False):
        self.currentGame = 0
        self.numberOfTrainingGames = int(numTraining)
        self.model = fetch_model(self.numberOfTrainingGames > 0)
        print(numTraining)
        # Some configurable parameters go here - try changing these to tune your results

        # This is the probability with which we will guide the pacman to make a good move during every state.
        # I have introduced this because, in some layouts, the pacman never wins during simulations, and hence, can
        # never find good moves it can exploit.
        # Think of this as an additional "exploitation factor". Kocsis does its own exploitation too.
        # This parameter is, of course, not used during real games.
        self.guidance = 0.3

        # The exploitation-exploration factor used by Kocsis - higher value = higher exploration
        self.c = sqrt(2) + 0.5

    def registerInitialState(self, state):
        """Used to update epoch variable for training"""
        self.currentGame += 1

    def getUCTValue(self, wins, num_games, num_games_with_parent):
        """"
        Implement UCT formula
        - use self.c to control exploration
        """
        "*** YOUR CODE HERE ***"
        pass


    def expand(self, MCTS_state, model):
        """
        A function that expands selected MCTS_state node
        into set of (MCTS_state, action) pairs 
        and calculates getUCTValue score for each action

        Implementation details:
        - MCTS_state.getLegalActions() contains a list of possible agent actions
        - if exist load the game/win values from model[(MCTS_state, action)]
          using attributes nSimulations, nWins
        - must return List[Tuple[action, UCT_score]]
        """
        legalActions = MCTS_state.getLegalActions()
        "*** YOUR CODE HERE ***"
        pass

    
    
    def inference_expand(self, MCTS_state, model):
        """
        This function is real-time counterpart of expand.
        
        UCT formula is useful for training MCTS models as it allows
        utilization of unexplored nodes. However during inference 
        we are limited to only one simulation and exploration of
        rare situations is likely to lead to guranteed loss.

        You should derive an action selection strategy that would 
        guide to winning move by utilizing accumulated knowledge in model.
        
        Implementation details:
        - You may just call the existing self.expand method
        - Otherwise just use some combination of nWins nSimulations avgReward
          attributes of model[(MCTS_state, action)]
        - MCTS_state.getLegalActions() contains a list of possible agent actions
        """
        "*** YOUR CODE HERE ***"
        pass

    def oracleExpand(self, MCTS_state):
        return MCTS_state.moveToClosestFood
    
    def oracleAvailable(self, MCTS_state):
        return not MCTS_state.isGhostBlockingPathToClosestFood

    def getAction(self, state):
        # type: (GameState) -> str
        MCTS_state = FeatureBasedGameState(state)
        if self.currentGame <= self.numberOfTrainingGames:
            # For better convergence use Oracle action-selection strategy once in a while
            if random.random() < self.guidance and self.oracleAvailable(MCTS_state):
                return self.oracleExpand(MCTS_state)
            else:
                action_scores = self.expand(MCTS_state, self.model)
        else:
            action_scores = self.inference_expand(MCTS_state, self.model)
        actionToReturn = max(action_scores, key=lambda x: x[1])[0]
        return actionToReturn
        


def backup(model, game):
    # type: (Model, Game) -> None
    """
    This function updates the MCTS model after each game

    Implementation details:
    - model fields can be accessed by model[(state, action)]
    - FeatureBasedGameState(gameState) is a MCTS-specific version of game state
    that is represented as a tuple of (best_move_direction:str, is_move_blocked_by_ghost:bool)
    (i.e. treat it as state in state-action key pair for model)
    - Use model.updateEntry(state, action, nWins, nSimulations, avgReward) to update model data
    - Use nWins, nSimulations, avgReward attributes of model[(state, action)] to get current values
    - Whether the game was won or not can be checked by game.state.isWin()
    """

    pairStateAction = [(FeatureBasedGameState(gameState), action) for gameState, (agent_idx, action)
                      in zip(game.stateHistory, game.moveHistory) if agent_idx == 0]
    score = game.state.getScore()
    # NOTE it is up to you to filter repeating state-action pairs
    "*** YOUR CODE HERE ***"
    pass

