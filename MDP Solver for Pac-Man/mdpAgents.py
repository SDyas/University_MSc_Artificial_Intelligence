# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
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


from pacman import Directions
from game import Agent
import api
import copy


def getPerpendiculars(key):
    if key == "up" or key == "down":
        return ["left", "right"]
    if key == "left" or key == "right":
        return ["up", "down"]


class MDPAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        # PACMAN
        self.pacLoc = None
        self.valueMap = None
        self.pacMap = None

        # FOOD
        self.foodLocations = None
        self.FOOD_VALUE = 1
        self.CAPSULE_VALUE = 1

        # GHOSTS
        self.ghostDict = None
        self.ghostBufferVisited = None
        self.GHOST_BUFFER_SIZE = 2  # don't need to change this, it's done automatically
        self.GHOST_VALUE = -1
        self.PILLED_GHOST_VALUE = 10
        self.MIN_PILL_TIME = 2

        # MAP
        self.xMax = None
        self.yMax = None
        self.mapType = "mediumClassic"  # don't need to change this, it's done automatically
        self.printMap = False

        # VALUE ITERATION
        self.DISCOUNT_FACTOR = 0.04
        self.GAMMA = 0.2  # don't need to change this, it's done automatically
        self.NUM_ITERATIONS = 11  # don't need to change this, it's done automatically

    #
    # Value Iteration
    #

    def valueIteration(self, pacMap):
        # apply the bellman update simultaneously (in parallel, NOT SEQUENTIAL) to all values
        # iterate until convergence on optimal values
        def assignValue(currX, currY):
            directions = {
                "left": [0, -1, 0],
                "up": [0, 0, 1],
                "right": [0, 1, 0],
                "down": [0, 0, -1]
            }

            # get the probabilities for each direction, its perpendiculars, and remaining stationary
            for chosenDir, value in directions.items():
                perpendiculars = getPerpendiculars(chosenDir)
                newDirections = [chosenDir, perpendiculars[0], perpendiculars[1]]

                # update probabilities
                pChosenDir, pPer1, pPer2, pStationary = 0.8, 0.1, 0.1, 0

                for key in range(len(newDirections)):
                    currDir = directions[newDirections[key]]
                    changeX = currDir[1]
                    changeY = currDir[2]

                    # check for wall to update pStationary
                    if pacMap[currX + changeX][currY + changeY] == "W":
                        if key == 0:
                            pChosenDir = 0
                            pStationary += 0.8
                        else:
                            if key == 1:
                                pPer1 = 0
                            else:
                                pPer2 = 0
                            pStationary += 0.1

                # calculate values
                currValue = 0
                for key in range(len(newDirections)):
                    currDir = directions[newDirections[key]]
                    changeX = currDir[1]
                    changeY = currDir[2]

                    # if not wall, update current value
                    if pacMap[currX + changeX][currY + changeY] != "W":
                        if key == 0:
                            prob = pChosenDir
                        elif key == 1:
                            prob = pPer1
                        else:
                            prob = pPer2
                        currValue += prob * pacMap[currX + changeX][currY + changeY]

                # add value for remaining stationary
                if pStationary:
                    currValue += pStationary * pacMap[currX][currY]

                # update value for current direction
                directions[chosenDir][0] += currValue

            # find max value and set to position in copyMap
            maxValue = -1
            for v in directions.values():
                maxValue = max(v[0], maxValue)

            tileValue = -self.DISCOUNT_FACTOR + self.GAMMA + maxValue
            copyMap[currX][currY] = tileValue

        # main loop
        iteration = 0
        while iteration < self.NUM_ITERATIONS:
            copyMap = copy.deepcopy(pacMap)
            for i in range(len(pacMap) - 1):
                for j in range(len(pacMap[0]) - 1):
                    if pacMap[i][j] != "W" and pacMap[i][j] != self.GHOST_VALUE:
                        assignValue(i, j)
            pacMap = copy.deepcopy(copyMap)
            iteration += 1

        # assign result of value iteration to value map
        self.valueMap = pacMap

        # print to observe each movement's value map
        if self.printMap:
            for row in pacMap:
                print(row)
            print("///////////////")

    #
    # Map Creation
    #

    # set variables for smallGrid map
    def setMapType(self, xMax, yMax):
        if xMax == 6 and yMax == 6:
            self.mapType = "smallGrid"
            self.GAMMA = 0
            self.NUM_ITERATIONS = 13
            self.GHOST_BUFFER_SIZE = 1

    # set the value for given position in pacMap
    def setGridTile(self, pacMap, xPos, yPos, value):
        if value == self.GHOST_VALUE:  # if ghost, set buffer
            self.ghostBufferVisited = set()
            self.setGhostBuffer(pacMap, xPos, yPos, self.GHOST_BUFFER_SIZE)
        pacMap[xPos][yPos] = value

    # dfs buffer around active ghost
    def setGhostBuffer(self, pacMap, xPos, yPos, bufferRemaining):
        # conditions to not set buffer at current position
        if (xPos >= len(pacMap)  # out of bounds x
                or yPos >= len(pacMap[0])  # out of bounds y
                or pacMap[xPos][yPos] == "W"  # wall
                or (xPos, yPos) in self.ghostBufferVisited  # already visited
                or bufferRemaining < 0):  # end of buffer
            return

        # set position to (ghost) value
        pacMap[xPos][yPos] = self.GHOST_VALUE

        # dfs in four compass directions
        self.ghostBufferVisited.add((xPos, yPos))
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        for cX, cY in directions:
            self.setGhostBuffer(pacMap, xPos + cX, yPos + cY, bufferRemaining - 1)
        self.ghostBufferVisited.remove((xPos, yPos))

    # value of ghost will be positive if enough pill time and not near spawn
    def setGhostValue(self, pos):
        spawnPositions = {(8, 5), (9, 5), (10, 5), (11, 5),
                          (8, 6), (9, 6), (10, 6), (11, 6),
                          (8, 7), (9, 7), (10, 7), (11, 7)}

        pilledTime = self.ghostDict[pos]
        if pos not in spawnPositions and pilledTime > self.MIN_PILL_TIME:
            return self.PILLED_GHOST_VALUE
        else:
            return self.GHOST_VALUE

    # for smallGrid map
    def setSmallMapGhostValue(self, xPos, yPos, itemValue):
        if (xPos, yPos) in self.ghostDict:
            return self.setGhostValue((xPos, yPos))
        else:
            return itemValue

    # for mediumClassic map
    def setMediumMapGhostValue(self, xPos, yPos, itemValue):
        if itemValue == self.GHOST_VALUE:
            return self.setGhostValue((xPos, yPos))
        else:
            return itemValue

    # create map with initial values
    def createMap(self, state):
        # set Pacman location
        self.pacLoc = api.whereAmI(state)

        # create empty map
        pacMapRows, pacMap = [], []
        for i in range(self.xMax + 1):
            pacMapRows.append(0)
        for j in range(self.yMax + 1):
            pacMap.append(list(pacMapRows))

        # create correctly formatted dictionary for ghost positions and pill times
        ghostState = api.ghostStatesWithTimes(state)
        ghostPositions = [p for p, t in ghostState]
        ghostPillTimes = [t for p, t in ghostState]
        self.ghostDict = {(int(p[0]), int(p[1])): t for p, t in zip(ghostPositions, ghostPillTimes)}

        # populate map with item values
        mapData = [
            {"positions": [self.pacLoc], "value": 0},
            {"positions": api.food(state), "value": self.FOOD_VALUE},
            {"positions": api.capsules(state), "value": self.CAPSULE_VALUE},
            {"positions": self.ghostDict.keys(), "value": self.GHOST_VALUE},
            {"positions": api.walls(state), "value": "W"}
        ]

        for item in mapData:
            for position in item["positions"]:
                xPos, yPos = int(position[0]), int(position[1])
                if self.mapType == "smallGrid":
                    item["value"] = self.setSmallMapGhostValue(xPos, yPos, item["value"])
                elif self.mapType == "mediumClassic":
                    item["value"] = self.setMediumMapGhostValue(xPos, yPos, item["value"])
                self.setGridTile(pacMap, xPos, yPos, item["value"])

        self.pacMap = pacMap

    #
    # Pacman Movement
    #

    # choose direction based on highest adjacent value
    def chooseDirection(self):
        directions = {"NORTH": [0, 1], "SOUTH": [0, -1], "EAST": [1, 0], "WEST": [-1, 0]}
        maxValue, direction = 0, None

        # for each compass direction
        for k, v in directions.items():
            changeX = self.pacLoc[0] + v[0]
            changeY = self.pacLoc[1] + v[1]
            newLocValue = self.valueMap[changeX][changeY]

            # choose the direction with the greatest value
            if newLocValue >= maxValue and newLocValue != "W":
                maxValue = newLocValue
                direction = k

        # return best direction
        if direction == "NORTH":
            return Directions.NORTH
        elif direction == "SOUTH":
            return Directions.SOUTH
        elif direction == "EAST":
            return Directions.EAST
        elif direction == "WEST":
            return Directions.WEST
        else:
            return Directions.STOP

    #
    # Main logic
    #

    # called once at start of game
    def registerInitialState(self, state):
        # get size of map
        corners = api.corners(state)
        xMax, yMax = 0, 0
        for corner in corners:
            xMax = max(yMax, corner[0])
            yMax = max(xMax, corner[1])

        # set map type
        self.setMapType(xMax, yMax)
        self.xMax, self.yMax = xMax, yMax

    # called every movement
    def getAction(self, state):
        self.createMap(state)
        self.valueIteration(self.pacMap)
        nextMove = self.chooseDirection()

        return api.makeMove(nextMove, api.legalActions(state))
