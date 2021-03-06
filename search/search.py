# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import heapq # required for uppdateIgnoreActions()

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    stack = [([], problem.getStartState())]
    visited = set()

    while stack:
        actions, node = stack.pop()
        if problem.isGoalState(node):
            return actions
        
        visited.add(node)
        for position, action, cost in problem.getSuccessors(node):
            if position not in visited:
                stack.append((actions + [action], position))

    print "didn't find path"
    return False

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    queue = [([], problem.getStartState())]
    visited = set()

    while queue:
        actions, node = queue.pop()
        if problem.isGoalState(node):
            return actions

        visited.add(node)
        for position, action, cost in problem.getSuccessors(node):
            if (
                    position not in visited and
                    position not in [snd for fst, snd in queue]
               ):
                queue.insert(0, (actions + [action], position))

    print "didn't find path"
    return False


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    pqueue = util.PriorityQueue()
    pqueue.push(([], problem.getStartState()), 0)
    visited = set(problem.getStartState())
 
    while pqueue:
        actions, node = pqueue.pop()
        if problem.isGoalState(node):
            return actions

        visited.add(node)
        for position, action, cost in problem.getSuccessors(node):
            if position not in visited:
                
                new_actions = actions + [action]
                new_cost = problem.getCostOfActions(actions) + cost
                pqueue.updateIgnoreActions((new_actions, position), new_cost)

    print "didn't find path"
    return False
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    pqueue = util.PriorityQueue()
    pqueue.push(([], problem.getStartState()), 0)
    visited = set(problem.getStartState())

    while pqueue:
        actions, node = pqueue.pop()
        if problem.isGoalState(node):
            return actions

        visited.add(node)
        for position, action, cost in problem.getSuccessors(node):
            if position not in visited:
                
                new_actions = actions + [action]
                new_cost = problem.getCostOfActions(actions) + cost + heuristic(position, problem)
                pqueue.updateIgnoreActions((new_actions, position), new_cost)

    print "didn't find path"
    return False



'''
I added this function for convenience!

The normal update() for PriorityQueue was awkward to use with 
my ([action], position) tuple

It would add a new element for like nodes with different paths

This function expects a tuple (actions, item) and only uses
the second value during comparison.
    
'''
def updateIgnoreActions(self, (actions, item), priority):
    # The same as update() for PriotityQueue EXCEPT:
    #   item must be a tuple: (list of actions, item)
    # [actions] will be ignored when comparing items

    for index, (p, c, (a, i)) in enumerate(self.heap):
        if i == item:
            if p <= priority:
                break
            del self.heap[index]
            self.heap.append((priority, c, (actions, item)))
            heapq.heapify(self.heap)
            break
    else:
        self.push((actions,item), priority)

util.PriorityQueue.updateIgnoreActions = updateIgnoreActions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
