# SLAMAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import slam
import random

class NullGraphics:
    "Placeholder for graphics"
    def initialize(self, state, isBlue = False):
        pass
    def update(self, state):
        pass
    def pause(self):
        pass
    def draw(self, state):
        pass
    def updateDistributions(self, dist):
        pass
    def finish(self):
        pass

class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """
    def initializeUniformly(self, gameState):
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = slam.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class SLAMAgent:
    "An agent that tracks and displays its beliefs about wall positions and its own position."

    def __init__( self, index = 0, inference = "SLAMParticleFilter", ghostAgents = None, observeEnable = True, elapseTimeEnable = True):
        self.inferenceType = util.lookup(inference, globals())     
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.prevAction = None
        
    def tellGameInfo(self, startPos, width, height, wallPrior, legalPositions):
        self.inferenceModules = [self.inferenceType(startPos, width, height, wallPrior, legalPositions)]
        self.inferenceModule = self.inferenceModules[0]

    def registerInitialState(self, gameState):
        import __main__
        self.display = __main__._display
        self.firstMove = True

    def observationFunction(self, gameState):
        """
        Observes noisy distance to walls in each of four directions.
        
        Also remembers what action Pacman just attempted to take,
        regardless of if that action was actually successful.
        
        Returns the gameState for bookkeeping purposes, though this is
        not known to the inference module.
        """        
        return gameState.getNoisyRangeMeasurements(), self.prevAction, gameState

    def getAction(self, observation):
        """
        "Updates beliefs, then chooses an action."
        """
        
        beliefs = []
        noisyRangeMeasurements, prevAction, gameState = observation
        if self.observeEnable:
            self.inferenceModule.observe(prevAction, noisyRangeMeasurements)
        beliefs.append(self.inferenceModule.getWallBeliefDistribution())
        beliefs.append(self.inferenceModule.getPositionBeliefDistribution())
        self.display.updateDistributions(beliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a SLAMAgent just stops.  This should be overridden."
        return Directions.STOP

class SLAMKeyboardAgent(SLAMAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about wall positions and its own position."

    def __init__(self, index = 0, inference = "SLAMParticleFilter", ghostAgents = None):
        KeyboardAgent.__init__(self, index)
        SLAMAgent.__init__(self, index, inference, ghostAgents)

    def getAction(self, gameState):
        return SLAMAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        action = KeyboardAgent.getAction(self, gameState)
        self.prevAction = action
        return action

from distanceCalculator import Distancer

class AutoSLAMAgent(SLAMAgent):
    "An agent that moves automatically at random."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        SLAMAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        legal = [a for a in gameState.getLegalPacmanActions()]
        legal.remove(Directions.STOP)
        action = random.choice(legal)
        self.prevAction = action
        return action
            
class PatrolSLAMAgent(SLAMAgent):
    "An agent that tends to move forward and avoids backtracking when possible."
    
    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        SLAMAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        legal = [a for a in gameState.getLegalPacmanActions()]
        legal.remove(Directions.STOP)
        if self.prevAction in legal:
            return self.prevAction
        for a in legal:
            if self.prevAction not in Directions.REVERSE or a != Directions.REVERSE[self.prevAction]:
                self.prevAction = a
                return a
        self.prevAction = Directions.REVERSE[self.prevAction]
        return self.prevAction
        
