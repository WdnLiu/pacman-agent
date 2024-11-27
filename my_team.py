# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food and strategically returns to its side
    to deposit points after collecting dots.
    """

    def get_features(self, game_state, action):
        
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        new_state = successor.get_agent_state(self.index)
        new_pos = new_state.get_position()
        # Food collected by the agent
        carried_food = new_state.num_carrying
        features['successor_score'] = -len(food_list)
        features['carrying_food'] = carried_food        
        curr_state = game_state.get_agent_state(self.index)
        # Identify if there are enemies nearby
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        scared_ghosts = [g for g in ghosts if g.scared_timer > 0]
        # Favor states where enemies are scared
        features['scared_enemies'] = len(scared_ghosts)
        enemy_pacman = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        enemy_pacman_nearby = [a for a in enemy_pacman if self.get_maze_distance(new_pos, a.get_position()) <= 10]

        if not new_state.is_pacman and len(enemy_pacman_nearby) > 0:
            if enemy_pacman_nearby:
                # Chase the nearest enemy Pac-Man
                pacman_distances = [
                    self.get_maze_distance(new_pos, a.get_position()) for a in enemy_pacman_nearby
                ]
                features['chase_enemy_pacman'] = -min(pacman_distances)                
                return features  # Skip other computations to focus on chasing


        # Compute distance to the nearest food
        if len(food_list) > 0:
            food_distances = [self.get_maze_distance(new_pos, food) for food in food_list]
            min_distance_to_food = min(food_distances)
            features['distance_to_food'] = min_distance_to_food
        else:
            features['distance_to_food'] = 9999

        # Count food within a close range (e.g., 3 spaces)
        nearby_food = [food for food in food_list if self.get_maze_distance(new_pos, food) <= 3]
        features['nearby_food_count'] = len(nearby_food)
        
        # If carrying food, prioritize returning to the home side
        if carried_food > 0 and len(nearby_food) < 1:
            home_positions = self.get_home_positions(game_state)
            min_home_distance = min([self.get_maze_distance(new_pos, home) for home in home_positions])
            features['distance_to_home'] = min_home_distance
        else:
            features['distance_to_home'] = 0  # No need to prioritize home if no food is carried

        # Discourage stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # Discourage reversing direction
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        # Compute distance to the nearest ghost if there are any
        if len(ghosts) > 0 and len(scared_ghosts) == 0:
            ghost_distances = [self.get_maze_distance(new_pos, ghost.get_position()) for ghost in ghosts]
            if min(ghost_distances) < 3:
                features['distance_to_ghost'] = -100
            else:
                features['distance_to_ghost'] = min(ghost_distances)
        else:
            features['distance_to_ghost'] = 9999  # No ghosts, no need to worry

        return features

    def get_weights(self, game_state, action):
        """
        Assigns weights to the offensive features for evaluation.
        """
        return {
            'successor_score': 50,          # Favor eating food (successor score)
            'distance_to_food': -1,         # Favor actions closer to food
            'distance_to_home': -2,         # Strongly favor returning home when carrying food
            'distance_to_ghost': 2,         # Avoid ghosts unless they are scared
            'stop': -100,                   # Strongly discourage stopping
            'scared_enemies': 500,          # Strongly favor states with scared enemies
            'reverse': -2,                  # Discourage reversing
            'chase_enemy_pacman': 200       # Prioritize chasing enemy Pac-Men on home side
        }

    def get_home_positions(self, game_state):
        """
        Returns the list of valid positions on the agent's home side.
        """
        width, height = game_state.data.layout.width, game_state.data.layout.height
        mid_x = width // 2 - 1 if self.red else width // 2  # Adjust for red/blue sides
        home_positions = []

        for y in range(height):
            if not game_state.data.layout.is_wall((mid_x, y)):
                home_positions.append((mid_x, y))

        return home_positions
      
class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that focuses on defending its territory by hunting invaders (enemy Pac-Men)
    and patrolling when no invaders are visible.
    """

    def get_features(self, game_state, action):
        """
        Computes features specific to a defensive strategy.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        new_state = successor.get_agent_state(self.index)
        new_pos = new_state.get_position()

        # On defense: checks if the agent is a Pac-Man or not
        features['on_defense'] = 1
        if new_state.is_pacman:
            features['on_defense'] = 0

        # Track visible invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Compute distance to invaders if any exist
        if len(invaders) > 0:
            invader_distances = [self.get_maze_distance(new_pos, invader.get_position()) for invader in invaders]
            features['invader_distance'] = min(invader_distances)
        else:
            # Patrol key points when no invaders are visible
            patrol_points = self.get_patrol_points(game_state)
            min_patrol_distance = min([self.get_maze_distance(new_pos, point) for point in patrol_points])
            features['patrol_distance'] = min_patrol_distance

        # Discourage stopping
        if action == Directions.STOP:
            features['stop'] = 1

        # Discourage reversing direction
        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        """
        Assigns weights to the defensive features for evaluation.
        """
        return {
            'num_invaders': -1000,      # Strongly prioritize targeting invaders
            'on_defense': 100,          # Encourage staying on defense
            'invader_distance': -10,    # Get closer to visible invaders
            'patrol_distance': -5,      # Patrol when no invaders are visible
            'stop': -100,               # Discourage stopping
            'reverse': -2               # Mildly discourage reversing
        }

    def get_patrol_points(self, game_state):
        """
        Determines key patrol points on the defensive side.
        These are typically entry points where invaders are likely to appear.
        """
        width, height = game_state.data.layout.width, game_state.data.layout.height
        mid_x = width // 2 - 1 if self.red else width // 2  # Adjust for red/blue sides
        patrol_points = []

        for y in range(height):
            if not game_state.data.layout.is_wall((mid_x, y)):
                patrol_points.append((mid_x, y))

        return patrol_points
