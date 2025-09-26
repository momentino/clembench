## Working Grid environment

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from typing import Dict, Tuple
import ast
import os

import logging

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("mapworld.environment")

class MapWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, render_mode: str = "human", size: int = 5, map_metadata: Dict = None,
                 agent_pos: str = None, target_pos: str = None):
        """
        Initialize mapworld as a Gymnasium environment.

        Args:
            render_mode: Type of rendering mode to use - human (renders the observation as a plot), rgb_array (returns an array without any plots)
            size: Grid size of the mapworld environment
            map_metadata: metadata from the ADEMap class for a graph
            agent_pos: Agent start room on the mapworld environment, assigns a random outdoor room if agent_pos is None
            target_pos: Target room on the mapworld environment (Think Escaperoom Base version),
                        assigns a random outdoor room if target_pos is None
        """

        self.size = size  # The size of the square grid
        self.window_size = 500  # The size of the PyGame window
        self.map_metadata = map_metadata

        self.start_pos = agent_pos if agent_pos is not None else np.array(ast.literal_eval(self.map_metadata["start_node"]))
        self.target_pos = target_pos if target_pos is not None else np.array(ast.literal_eval(self.map_metadata["target_node"]))
        self._agent_location = self.start_pos
        self._target_location = self.target_pos

        # Counters
        self.visited = set()
        self.visited.add(tuple(self.start_pos))
        self.reached_target = False

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2
        # Ref - https://gymnasium.farama.org/introduction/create_custom_env/
        # Ref - https://gymnasium.farama.org/introduction/basic_usage/#action-and-observation-spaces
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 5 actions, corresponding to "east", "north", "west", "south", "explore", and "escape"
        self.action_space = spaces.Discrete(6)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "down" etc.
        """

        self._action_to_direction = {
            0: np.array([1, 0]),  # East
            1: np.array([0, 1]),  # South
            2: np.array([-1, 0]), # West
            3: np.array([0, -1]), # North
            4: np.array([0, 0]),  # Explore
            5: np.array([1, 1])   # Escape 
        }

        self._action_to_move = {
            0: "east",
            1: "south",
            2: "west",
            3: "north",
            4: "<explore>",
            5: "<escape>"
        }

        self._move_to_action = {
            "east": 0,
            "south": 1,
            "west": 2,
            "north": 3,
            "<explore>": 4,
            "<escape>": 5
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        self._agent_location = self.start_pos
        self._target_location = self.target_pos

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        
        # Map the action (element of {0,1,2,3,4}) to the direction we walk in
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        if tuple(self._agent_location) == tuple(self.target_pos):
            self.reached_target = True

        if tuple(self._agent_location) not in self.visited:
            self.visited.add(tuple(self._agent_location))

        # An episode is done if the guide agent has generated the <escape> token
        terminated = 0
        reward = 0
        if self._action_to_move[action] == "<escape>":
            terminated = 1
            reward = 1
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _draw_rect(self, canvas, color, pos, pix_square_size, room_ratio, label):

        # Define rectangle dimensions
        pygame.draw.rect(
            canvas,
            color,
            pygame.Rect(
                (pos*pix_square_size + ((1-room_ratio)/2)*pix_square_size), # (left, top)
                (room_ratio*pix_square_size, room_ratio*pix_square_size), # (width, height)
            ),
        )


        text_pos = pos*pix_square_size + pix_square_size/2
        text_pos = [text_pos[0], text_pos[1] - pix_square_size/2 + 10]
        text_surf = self.font.render(str(label), True, (0, 0, 0))
        # center it in the cell
        text_rect = text_surf.get_rect(center=text_pos)
        canvas.blit(text_surf, text_rect)


    def _draw_line(self, canvas, color, edge, pix_square_size, room_ratio):
        start_pos = np.array(edge[0])*pix_square_size + pix_square_size/2
        end_pos = np.array(edge[1])*pix_square_size + pix_square_size/2

        if edge[0][0] < edge[1][0]:
            # Horizontal edges
            start_pos[0] = start_pos[0] + room_ratio/2
            end_pos[0] = end_pos[0] - room_ratio/2
        elif edge[0][0] > edge[1][0]:
            start_pos[0] = start_pos[0] - room_ratio / 2
            end_pos[0] = end_pos[0] + room_ratio / 2
        elif edge[0][1] < edge[1][1]:
            # Vertical Edges
            start_pos[1] = start_pos[1] + room_ratio/2
            end_pos[1] = end_pos[1] - room_ratio/2
        else:
            start_pos[1] = start_pos[1] - room_ratio / 2
            end_pos[1] = end_pos[1] + room_ratio / 2

        pygame.draw.line(canvas, color, start_pos, end_pos)

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.font.init()
            self.font = pygame.font.SysFont("Arial", 10)
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = int(
            self.window_size / self.size
        )  # The size of a single grid square in pixels
        room_ratio = 0.6 # Ratio of a visible room pixel wrt pix_square_size

        # Draw edges
        for edge in self.map_metadata["unnamed_edges"]:
            self._draw_line(canvas, (0, 0, 0), edge, pix_square_size, room_ratio)

        # Draw Pieces
        for node in self.map_metadata["unnamed_nodes"]:
            self._draw_rect(canvas, (255,0,0), np.array(node), pix_square_size,
                            room_ratio, self.map_metadata["node_to_category"][str(tuple(node))])

        # Now we draw the agent
        self.robot_img = pygame.image.load(os.path.join("engine", "resources", "robot.png")).convert_alpha()
        # scale it to cell size
        self.robot_img = pygame.transform.smoothscale(
            self.robot_img, (room_ratio*pix_square_size, room_ratio*pix_square_size)
        )

        # draw the robot
        canvas.blit(self.robot_img, self._agent_location*pix_square_size + 0.2*pix_square_size)

        self.window.blit(canvas, (0, 0))
        pygame.display.flip()

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    @staticmethod
    def _get_direction(start_pos: Tuple, next_pos: Tuple) -> str:

        """
        Get the direction of next move
        Args:
            start_pos: current node of the agent inside mapworld
            next_pos: next possible node of the agent inside mapworld

        Returns:
            direction: direction of the next move as a string item
        """
        if next_pos[0] == start_pos[0] and next_pos[1] == start_pos[1] + 1:
            return "south"
        elif next_pos[0] == start_pos[0] and next_pos[1] == start_pos[1] - 1:
            return "north"
        elif next_pos[1] == start_pos[1] and next_pos[0] == start_pos[0] + 1:
            return "east"
        elif next_pos[1] == start_pos[1] and next_pos[0] == start_pos[0] - 1:
            return "west"
        else:
            raise ValueError("Invalid move! Check the node positions!")

    def get_next_moves(self):
        agent_node = self._agent_location
        moves = []
        edges = self.map_metadata['unnamed_edges']

        for edge in edges:
            start_pos = None
            next_pos = None
            # Check edges from current agent position
            # TODO: Save metadata containing edge from n1 to n2 and n2 to n1, instead of only one of em
            edge1 = ast.literal_eval(edge[0])
            edge2 = ast.literal_eval(edge[1])

            if np.array_equal(edge1, agent_node):
                start_pos = edge1
                next_pos = edge2
            elif np.array_equal(edge2, agent_node):
                start_pos = edge2
                next_pos = edge1

            if start_pos:
                direction = self._get_direction(start_pos, next_pos)
                # room = self.map_metadata['node_to_category'][str(tuple(next_pos))]
                moves.append(direction)

        return str(moves)


    def _check_room(self) -> str:
        """
        Checks if the current room of agent is ambiguous, target room, or other
        Returns:
            "ambiguous" if ambiguous room,
            "target" if target room,
            "other" if other room
        """
        current_node = self._agent_location

        room_name = self.map_metadata["node_to_category"][str(tuple(current_node))]
        if room_name.endswith("1") or room_name.endswith("2") or room_name.endswith("3"):
            room_name = room_name[:-2].strip()
            target_name = self.map_metadata["node_to_category"][str(tuple(self.target_pos))]
            target_name = target_name[:-2].strip()
            if target_name == room_name:
                return "ambiguous"
            else:
                return "other"
        else:
            return "other"


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == '__main__':
    # n, m = 4, 4
    # rooms = 8
    # # np.random.seed(44)
    # ademap = ADEMap(m, n, rooms)
    # graph_a = ademap.create_cycle_graph()
    # graph_a = ademap.assign_types(graph_a, ambiguity=[1], use_outdoor_categories=False)
    # graph_a = ademap.assign_images(graph_a)
    # metadata = ademap.metadata(graph_a, "indoor", "indoor")
    # print(metadata)
    metadata = {
                    "game_id": 0,
                    "graph_id": "44m43p42n32p33w34b35k45h",
                    "m": 6,
                    "n": 6,
                    "named_nodes": [
                        "Music studio",
                        "Playroom 1",
                        "Nursery",
                        "Playroom 2",
                        "Workroom",
                        "Breakroom",
                        "Kindergarden classroom",
                        "Hunting lodge indoor"
                    ],
                    "unnamed_nodes": [
                        [
                            4,
                            4
                        ],
                        [
                            4,
                            3
                        ],
                        [
                            4,
                            2
                        ],
                        [
                            3,
                            2
                        ],
                        [
                            3,
                            3
                        ],
                        [
                            3,
                            4
                        ],
                        [
                            3,
                            5
                        ],
                        [
                            4,
                            5
                        ]
                    ],
                    "named_edges": [
                        [
                            "Music studio",
                            "Playroom 1"
                        ],
                        [
                            "Music studio",
                            "Hunting lodge indoor"
                        ],
                        [
                            "Playroom 1",
                            "Nursery"
                        ],
                        [
                            "Nursery",
                            "Playroom 2"
                        ],
                        [
                            "Playroom 2",
                            "Workroom"
                        ],
                        [
                            "Workroom",
                            "Breakroom"
                        ],
                        [
                            "Breakroom",
                            "Kindergarden classroom"
                        ],
                        [
                            "Kindergarden classroom",
                            "Hunting lodge indoor"
                        ]
                    ],
                    "unnamed_edges": [
                        [
                            [
                                4,
                                4
                            ],
                            [
                                4,
                                3
                            ]
                        ],
                        [
                            [
                                4,
                                4
                            ],
                            [
                                4,
                                5
                            ]
                        ],
                        [
                            [
                                4,
                                3
                            ],
                            [
                                4,
                                2
                            ]
                        ],
                        [
                            [
                                4,
                                2
                            ],
                            [
                                3,
                                2
                            ]
                        ],
                        [
                            [
                                3,
                                2
                            ],
                            [
                                3,
                                3
                            ]
                        ],
                        [
                            [
                                3,
                                3
                            ],
                            [
                                3,
                                4
                            ]
                        ],
                        [
                            [
                                3,
                                4
                            ],
                            [
                                3,
                                5
                            ]
                        ],
                        [
                            [
                                3,
                                5
                            ],
                            [
                                4,
                                5
                            ]
                        ]
                    ],
                    "category_to_node": {
                        "Music studio": [
                            4,
                            4
                        ],
                        "Playroom 1": [
                            4,
                            3
                        ],
                        "Nursery": [
                            4,
                            2
                        ],
                        "Playroom 2": [
                            3,
                            2
                        ],
                        "Workroom": [
                            3,
                            3
                        ],
                        "Breakroom": [
                            3,
                            4
                        ],
                        "Kindergarden classroom": [
                            3,
                            5
                        ],
                        "Hunting lodge indoor": [
                            4,
                            5
                        ]
                    },
                    "node_to_image": {
                        "(4, 4)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/cultural/music_studio/ADE_train_00012281.jpg",
                        "(4, 3)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/playroom/ADE_train_00015396.jpg",
                        "(4, 2)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/nursery/ADE_train_00013898.jpg",
                        "(3, 2)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/playroom/ADE_train_00015404.jpg",
                        "(3, 3)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/work_place/workroom/ADE_train_00020099.jpg",
                        "(3, 4)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/shopping_and_dining/breakroom/ADE_train_00004520.jpg",
                        "(3, 5)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/cultural/kindergarden_classroom/ADE_train_00010112.jpg",
                        "(4, 5)": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/hunting_lodge__indoor/ADE_train_00009730.jpg"
                    },
                    "category_to_image": {
                        "Music studio": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/cultural/music_studio/ADE_train_00012281.jpg",
                        "Playroom 1": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/playroom/ADE_train_00015396.jpg",
                        "Nursery": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/nursery/ADE_train_00013898.jpg",
                        "Playroom 2": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/playroom/ADE_train_00015404.jpg",
                        "Workroom": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/work_place/workroom/ADE_train_00020099.jpg",
                        "Breakroom": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/shopping_and_dining/breakroom/ADE_train_00004520.jpg",
                        "Kindergarden classroom": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/cultural/kindergarden_classroom/ADE_train_00010112.jpg",
                        "Hunting lodge indoor": "https://www.ling.uni-potsdam.de/clembench/adk/images/ADE/training/home_or_hotel/hunting_lodge__indoor/ADE_train_00009730.jpg"
                    },
                    "node_to_category": {
                        "(4, 4)": "Music studio",
                        "(4, 3)": "Playroom 1",
                        "(4, 2)": "Nursery",
                        "(3, 2)": "Playroom 2",
                        "(3, 3)": "Workroom",
                        "(3, 4)": "Breakroom",
                        "(3, 5)": "Kindergarden classroom",
                        "(4, 5)": "Hunting lodge indoor"
                    },
                    "start_node": "(3, 4)",
                    "target_node": "(4, 3)",
                    "explorer_prompt": "You are stuck in a mapworld environment containing several rooms. \nYour task is to explore this world and reach an escape room.\nYou are always given an image of the current room you are in. \nI have an image of the Escape Room, this is an initial description of my image - $INIT_DESCRIPTION.\n\nBased on my description and the image given, you now have three options\n\nFirst Option - If you think that the description of my image matches the image you have, i.e.\nyou are in the escape room then respond with one word - ESCAPE.\n\nSecond Option - If you think you need more details from my image - respond in the following format\nQUESTION: Ask details about the image I have to verify if we have the same image or not\n\nThird Option - If you think you are in a different room than what I have, i.e we are seeing different images,\nthen you can make a move in following Directions - $DIRECTIONS. Respond in the following format\nMOVE: direction - Here direction can be one of {north, south, east, west}\n",
                    "guide_prompt": "I need your help, I am stuck in a mapworld environment. \nYour task is to help me reach an escape room. I do not know what the escape room looks like. \nBut fortunately, you have an image of the escape room with you. I will explore each room here and ask you a few QUESTIONS\nto verify if we have the same image or not, thus verifying if I am in the Escape Room or not.\n\nFirst start by describing the image that you have, Respond in the following format\nDESCRIPTION: your description of the image that you have\n\nThen if I ask a QUESTION, the respond appropriately based on your image in the following format\nANSWER: your Answer\n",
                    "explorer_reprompt": "\nNow you made a move to this room, and you can either ESCAPE, ask me a QUESTION or MOVE in one of these directions \n$MOVES\n"
                }
    env = MapWorldEnv(render_mode="human", size=6, map_metadata=metadata)


    env.reset()
    env.render()

    moves = [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4]

    for i in moves:
        env.render()
        env.step(i)
    env.close()