# inference.py
# ------------
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


import itertools
import util
import random
import game
import slam, pdb, math

from game import Directions, Grid, Actions

class InferenceModule:
    """
    An inference module tracks a belief distribution over walls and
    Pacman's location.
    This is an abstract class, which you should not modify.
    """

    ############################################
    # Useful methods for all inference modules #
    ############################################

    def __init__(self):
        pass

    ######################################
    # Methods that need to be overridden #
    ######################################

    def initialize(self):
        "Sets the belief state to some initial configuration."
        pass

    def observe(self, observation):
        """
        Updates beliefs based on the given distance observation and gameState.
        This combines together observe and elapseTime from the previous particle
        filtering project. You may, of course, make your own elapseTime helper
        function if you wish.
        """
        pass

    def getWallBeliefDistribution(self):
        """
        Returns the agent's current belief state about the distribution
        of walls conditioned on all evidence so far.
        """
        pass
    
    def getPositionDistribution(self):
        """
        Returns the agent's current belief state about the distribution
        of its own position conditioned on all evidence so far.
        """
        pass


def getTrueRangeMeasurement( position, hypothesizedMap):
        x, y = position
        currentX = x
        currentY = y
        while not hypothesizedMap[currentX][currentY]:
            currentY += 1
        N = util.manhattanDistance((x, y), (currentX, currentY))
        currentX = x
        currentY = y
        while not hypothesizedMap[currentX][currentY]:
            currentX += 1
        E = util.manhattanDistance((x, y), (currentX, currentY))
        currentX = x
        currentY = y
        while not hypothesizedMap[currentX][currentY]:
            currentY -= 1
        S = util.manhattanDistance((x, y), (currentX, currentY))
        currentX = x
        currentY = y
        while not hypothesizedMap[currentX][currentY]:
            currentX -= 1
        W = util.manhattanDistance((x, y), (currentX, currentY))
        return (N, E, S, W)

class SLAMParticleFilter(InferenceModule):
    """
    Particle filtering inference module for use in SLAM.
    """
    
    "*** YOU MAY ADD WHATEVER HELPER METHODS YOU WANT TO THIS CLASS ***"
    
    def __init__(self, startPos, layoutWidth, layoutHeight, wallPrior, legalPositions, numParticles=1000):
        "*** YOU OVERWRITE THIS METHOD HOWEVER YOU WANT ***"
        self.startPos = startPos
        self.legalPositions = legalPositions
        self.numParticles = numParticles
        self.width = layoutWidth
        self.height = layoutHeight
        self.wallPrior = wallPrior


        self.priorWallBeliefDistribution = self.generatePriorMap
        self.particles = [(self.startPos, self.generatePriorMap) for _ in xrange(self.numParticles)]
        self.weights = util.Counter()  # weights of 500 particles
        self.allDirections = (Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP)

        self.posBeleives = None
        self.prob_ONE = 0.9999
        self.prob_ZERO = 0.0001

    def onEdge(self, x, y):
        return (x == 0) or (x == self.width - 1) or (y == 0) or (y == self.height - 1)

    def inverse(self, num):
        return num/(1.0-num)

    @property
    def generatePriorMap(self):
        wallBeliefDistribution = util.Counter() 
        for pos in self.legalPositions:
            wallBeliefDistribution[pos] = self.inverse(0.999) if self.onEdge(*pos) else self.inverse(self.wallPrior)
        return wallBeliefDistribution

    def vector_sum(self, v1, v2):
        return (int(v1[0]+v2[0]), int(v1[1]+v2[1]))

    def initialize(self):
        "*** YOU OVERWRITE THIS METHOD HOWEVER YOU WANT ***"
        self.weights[self.startPos] = 1.0

    def isWall(self, prob):
        return prob > 0.5

    def reweightParticles(self, ranges):

        self.weights = util.Counter()
        for pos, map in self.particles:
            gridMap = Grid(self.width, self.height)
            for position in map:
                prob = map[position] / ( map[position] + 1.0)
                if self.onEdge(*position) or self.isWall(prob):
                        gridMap[position[0]][position[1]] = True
            trueRanges = getTrueRangeMeasurement(pos, gridMap)
            newWeight = 1.0
            for r, trueRange in zip(ranges, trueRanges):
                newWeight *= slam.getObservationDistribution(r)[trueRange] if slam.getObservationDistribution(r)[trueRange] else self.prob_ZERO
            self.weights[pos] += newWeight
        self.weights.normalize()

    def observe(self, prevAction, ranges):
        "*** YOU OVERWRITE THIS METHOD HOWEVER YOU WANT ***"

        def nextPacamPosDistribution( prevAction, particlePosition, particleMap):
            newDistribution = util.Counter()
            unintendedPositions = []
            for action in self.allDirections:
                successorPosition = self.vector_sum(Actions.directionToVector(action), particlePosition)
                if action != prevAction:  unintendedPositions.append(successorPosition)
                else:
                    wallProb = particleMap[successorPosition] / (particleMap[successorPosition] + 1)
                    newDistribution[successorPosition] = 0.9 * (1.0 - wallProb)
            for pos in unintendedPositions:
                wallProb = particleMap[successorPosition] / (particleMap[successorPosition] + 1)
                newDistribution[pos] = 0.1 / len(unintendedPositions) * (1.0 - wallProb)
            newDistribution.normalize()
            return newDistribution

        if prevAction != None:
            self.particles = [(util.sampleFromCounter(nextPacamPosDistribution(prevAction, oldPos, map)), map) for (oldPos, map) in self.particles]
        newParticles = []
        for particlePosition, particleMap in self.particles:
            newParticleMap = util.Counter()
            for pos in self.legalPositions:
                # take PacmanPosition as given
                newParticleMap[pos] = particleMap[pos]
                newProb = self.emissionModel(particlePosition, ranges, pos)
                if newProb != None:
                    prob = newParticleMap[pos] * (newProb / (1 - newProb) / self.priorWallBeliefDistribution[pos])
                    newParticleMap[pos] = 9999 if prob > 9999 else (self.prob_ZERO if prob == 0 else prob)
            newParticles.append((particlePosition, newParticleMap))
        self.particles = newParticles
        self.reweightParticles(ranges)
        mapAtPos = lambda particlePosition: [particle[1] for particle in self.particles if particle[0] == particlePosition][0]
        newParticlePositions = (util.sampleFromCounter(self.weights) for _ in range(self.numParticles))
        self.particles = [(part, mapAtPos(part)) for part in newParticlePositions]
        # update position belief
        self.posBeleives = util.Counter()
        for pos, _ in self.particles: self.posBeleives[pos] = 1.0

    def emissionModel(self, pacmanPosition, rangeMeasurements, wallPosition):
        x, y = pacmanPosition
        wallPosition0, wallPosition1 = wallPosition
        distN, distE, distS, distW = rangeMeasurements
        sameHorizontalLine = lambda pos1, pos2: pos1[1] == pos2[1]  # same y
        sameVerticalLine = lambda pos1, pos2: pos1[0] == pos2[0]  # same x
        if self.onEdge(*wallPosition):
            return self.prob_ONE
        if pacmanPosition == wallPosition: return self.prob_ZERO
        elif sameVerticalLine(wallPosition, pacmanPosition):
            if ((wallPosition1 - y < distN) and (wallPosition1 > y)) or ((y - wallPosition1 < distS) and (y - wallPosition1 > 0)): return self.prob_ZERO
            elif (wallPosition1 - y == distN) or (y - wallPosition1 == distS): return self.prob_ONE
        elif sameHorizontalLine(wallPosition, pacmanPosition):
            if ((wallPosition0 - x < distE) and (wallPosition0 - x > 0)) or ((x - wallPosition0 < distW) and (x - wallPosition0 > 0)): return self.prob_ZERO
            elif (wallPosition0 - x == distE) or (x - wallPosition0 == distW): return self.prob_ONE


    def getWallBeliefDistribution(self):
        "*** YOU OVERWRITE THIS METHOD HOWEVER YOU WANT ***"
        wallDist = util.Counter()
        for particlePosition, particleMap in self.particles:
            for position, storedProb in particleMap.items():
                wallDist[position] += self.weights[particlePosition] * storedProb / (storedProb + 1.0)
        for key in wallDist:
            wallDist[key] = wallDist[key] / self.numParticles
        return wallDist

    def getPositionBeliefDistribution(self):
        "*** YOU OVERWRITE THIS METHOD HOWEVER YOU WANT ***"
        beliefs = util.Counter()
        for pos, _ in self.particles:
            beliefs[pos] += 1.0
        beliefs.normalize()
        return beliefs
