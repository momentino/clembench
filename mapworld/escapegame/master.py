"""
Game Master for EscapeGame
"""
# TODO : Add max number of images - If it reaches limit remove images from beginning
# TODO : Add this value as a variable.

import ast
from typing import List, Dict, Union
import logging
import os
import json
import re

from clemcore.clemgame import Player, GameMaster, GameBenchmark, DialogueGameMaster, GameScorer, GameSpec
from clemcore.backends import Model
from clemcore.clemgame.errors import ParseError
import numpy as np

from mapworld.engine.environment import MapWorldEnv
from mapworld.engine.utils import get_next_node
from mapworld.escapegame.scorer import EscapeRoomScorer, is_efficient_move, get_neighbors_str

logger = logging.getLogger(__name__)
stdout_logger = logging.getLogger("escapegame.master")

# Disable logs from Multimodal backend
# logging.getLogger("huggingface.multimodal.api").disabled = True

# Language Config file - Handles the text in multiple languages (default - English)
lang_config_path = os.path.join(os.path.dirname(__file__), "resources", "language_config_en.json")
with open(lang_config_path) as f:
    LANG_CFG = json.load(f)

class Seeker(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self.response: str = ""
        self.tag: str = "Seeker"

    def _custom_response(self, context: Dict) -> str:
        random_move = np.random.choice(["north", "south", "east", "west", "escape"])
        if random_move == "escape":
            return "ESCAPE"
        else:
            return f"MOVE: {random_move}"


class Guide(Player):
    def __init__(self, model: Model):
        super().__init__(model)
        self.response: str = ""
        self.tag: str = "Guide"

    def _custom_response(self, context: Dict) -> str:
        return "DESCRIPTION: This is a sample description"


class EscapeRoom(DialogueGameMaster):

    def __init__(self, game_spec: GameSpec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)

        # Scorers
        self.aborted = False
        self.fail = False  # Set when Seeker returns any invalid move
        self.success = False  # Set when Seeker returns - ESCAPE and seeker location == target location
        self.reprompt_fail = False  # Set when Seeker returns a move to an invalid room

        # Pass Turn
        self.pass_turn = True

    def _on_setup(self, **game_instance):

        # Initialize Game Instance
        self.game_instance = game_instance
        self.m = game_instance["m"]
        self.game_map = MapWorldEnv(render_mode="rgb_array", size=self.m, map_metadata=self.game_instance)

        # Prompts
        self.seeker_base_prompt: str = self.game_instance["seeker_prompt"]
        self.seeker_base_reprompt: str = self.game_instance["seeker_reprompt"]
        self.seeker_base_failed_reprompt: str = self.game_instance["seeker_failed_reprompt"]
        self.guide_prompt: str = self.game_instance["guide_prompt"]

        # Initialize Players
        # Player 1 (Seeker) is in the mapworld
        # Player 2 (Guide) is outside the world
        self.seeker = Seeker(self.player_models[0])
        self.guide = Guide(self.player_models[1])

        # Setup for Seeker/Player1
        self.seeker_pos = self.game_instance["start_node"]
        self.seeker_image = self.game_instance["node_to_image"][self.seeker_pos]
        # Keep the nodes and edges as str in master (straightforward mapping) but pass as Tuples to the mapworld engine

        self.max_seeker_retries = 1  # At max, Let the seeker make 1 wrong move continuously from the same room
        self.current_seeker_try = 0  # reset try after every seeker move (to another room)
        self.total_seeker_moves = 0  # log all seeker moves valid+invalid here.
        # Check against a max value for aborting

        # Name of the room category - bedroom, for example
        self.seeker_room = self.game_instance["node_to_category"][self.seeker_pos]
        self.initial_description_tag = LANG_CFG["initial_description_tag"]
        self.directions_tag = LANG_CFG["directions_tag"]

        self.seeker_target = self.game_instance["target_node"]

        # Setup for Guide/Player2
        self.escape_pos = self.game_instance["target_node"]
        self.escape_room = self.game_instance["node_to_category"][self.escape_pos]
        self.guide_image = self.game_instance["node_to_image"][self.escape_pos]

        # Add players
        # NOTE: Player calls will be made in the order below
        self.add_player(self.guide)
        self.add_player(self.seeker)

        # Question Flag to keep track of - when seeker asks the questions, and if guide responds with answer
        self.question_flag = 0

    def _on_before_game(self):
        """
        Pass initial message - first player (Guide), first turn
        """

        # Add initial prompt to Seeker in (Seeker's) history
        self.set_context_for(self.guide, self.guide_prompt, image=[self.guide_image])
        
        stdout_logger.info(f"First message for Guide: {self.guide_prompt}")
        stdout_logger.info(f"First Room image path for Guide: {self.guide_image}")

    def _does_game_proceed(self) -> bool:
        """
        Fail cases for each turn, use init_flags/scorers etc...
        """
        if self.aborted or self.current_round == 25 or self.success or self.fail:
            return False
        else:
            return True

    def _should_pass_turn(self):
        return self.pass_turn

    @staticmethod
    def clean_agent_response(response: str) -> str:
        """
        Remove leading and trailing markdown wrappers from response
        """
        response = response.strip()
        response = response.replace("```json", "")
        response = response.replace("```", "")
        if response.endswith("."):
            response = response[:-1]
        return response.lower()

    def _parse_response(self, player, utterance: str) -> bool:
        """
        Check Correct format/ tag etc... in each Player's response
        Args:
            player: Player object - Seeker or Guide type
            utterance: str - response from Player
        Returns:
            True if response format is valid, False otherwise
        """
        stdout_logger.info(f"Generated player response: {utterance}")
        utterance = self.clean_agent_response(utterance)
        stdout_logger.info(f"Cleaned Player response {player.tag}: {utterance}")

        if not utterance:
            self.aborted = True
            stdout_logger.info(f"Aborting the Game. Could not parse response from Player.")
            stdout_logger.info(f"Invalid utterance: {utterance}")
            self.log_to_self("invalid value", "abort game: seeker")
            raise ParseError(f"No utterance generated by Player - {player.tag}")

        if type(player) == Seeker:
            """
            Seeker should respond only in one of the following format
            1) MOVE: North
            2) ESCAPE
            3) QUESTION: 
            Check for each tag

            Abort - If seeker responds in invalid format, or invalid keys
            """
            valid_tags = ["move", "escape", "question"]
            valid_directions = ["north", "east", "south", "west"]
            utterance = utterance.lower()
            splits = utterance.split(":")

            tag = splits[0]
            invalid_move = False
            self.question_flag = 0

            if tag not in valid_tags:
                self.aborted = True
                stdout_logger.info(f"Aborting the Game. Seeker generated invalid tag {tag}")
                stdout_logger.info(f"Invalid utterance: {utterance}")
                self.log_to_self("invalid value", "abort game: seeker")
                raise ParseError(f"Invalid utterance: {utterance}. Player response should start with either 'Move:' , 'Question:' or 'Escape' tags")

            if tag == "move":
                self.total_seeker_moves += 1
                stdout_logger.info(f"Current seeker move: {self.total_seeker_moves}")
                if self.total_seeker_moves >= 14:
                    self.fail = True
                    self.log_to_self("turns exceeded", "failed game: seeker")

                stdout_logger.info(f"Move made from location - {self.game_map._agent_location}")
                move = splits[1]
                move = move.lower().strip()
                self.pass_turn = False

                if move not in valid_directions:
                    self.aborted = True
                    stdout_logger.info(f"Aborting the Game. Seeker generated invalid move {move}")
                    stdout_logger.info(f"Invalid utterance: {utterance}")
                    self.log_to_self("invalid value", "abort game: seeker")
                    raise ParseError(f"Invalid move: {move}. Move should be one of {valid_directions}")

                next_node = get_next_node(tuple(self.game_map._agent_location), move)
                next_node = tuple(next_node)

                stdout_logger.info(f"Move: {move}")
                stdout_logger.info(f"Next node: {next_node}")
                # FIXME: add str/tuple typecheck
                current_node = str(tuple(self.game_map._agent_location))
                next_node_str = str(next_node)
                edge = [current_node, next_node_str]
                reverse_edge = [next_node_str, current_node]

                if edge not in self.game_map.map_metadata["unnamed_edges"] and reverse_edge not in \
                        self.game_map.map_metadata["unnamed_edges"]:
                    stdout_logger.info(f"Invalid move from {current_node} to {next_node_str}")
                    self.log_to_self("move", "invalid")
                    self.reprompt_fail = True
                    self.current_seeker_try += 1
                    invalid_move = True
                    if self.current_seeker_try == self.max_seeker_retries:
                        self.fail = True
                        self.log_to_self("turns exceeded", "failed game: seeker")
                else:
                    stdout_logger.info(f"Valid move: {move}")
                    # self.log_to_self("move", "valid")
                    self.current_seeker_try = 0  # Reset seeker tries

                edges = self.game_map.map_metadata["unnamed_edges"]
                tuple_edges = []
                for edge in edges:
                    tuple_edges.append((tuple(ast.literal_eval(edge[0])), tuple(ast.literal_eval(edge[1]))))

                return utterance

            # Episodic Success case
            elif tag == "escape":
                stdout_logger.info(f"Agent Location - {str(tuple(self.game_map._agent_location))}")
                stdout_logger.info(f"Target Location - {self.game_instance['target_node']}")
                if str(tuple(self.game_map._agent_location)) == self.game_instance["target_node"]:
                    stdout_logger.info(f"Escape room {self.escape_room} - Reached, Seeker successfully escaped!")
                    self.log_to_self("escape", "success")
                    self.success = True
                else:
                    stdout_logger.info(f"Seeker tried to Escape from a wrong room!")
                    self.log_to_self("escape", "failed")
                    self.fail = True


                return utterance

            # tag == "question"
            else:
                self.question_flag = 1
                self.pass_turn = True
                stdout_logger.info(f"Seeker asked Question - {utterance}")
                self.log_to_self("question", "seeker")


                return utterance

        else:
            """
            Guide should respond only in one of the following format
            1) DESCRIPTION: 
            2) ANSWER:
            """
            utterance = utterance.lower()
            splits = utterance.split(":")
            tag = splits[0]
            valid_tags = ["description", "answer"]
            self.pass_turn = True
            
            if "description:" in utterance and "answer:" in utterance:
                self.aborted = True
                stdout_logger.info(
                    f"Invalid Response for Guide: Expected DESCRIPTION/ANSWER tag, got both - {utterance}")
                self.log_to_self("invalid value", "abort game: guide")  # Violated request count
                raise ParseError(f"Response for Guide: Got both Description and Answer tags")

            if tag not in valid_tags:
                self.aborted = True
                stdout_logger.info(f"Invalid Response for Guide: Expected DESCRIPTION/ANSWER tag, got {splits[0]}")
                self.log_to_self("invalid value", "abort game: guide")  # Violated request count
                raise ParseError(f"The response of Guide should start with either 'Description:' or 'Answer:' tags")

            if tag == "description":
                if self.question_flag == 1:
                    self.fail = True
                    self.log_to_self("description", "wrong response")
                    stdout_logger.info(f"Description by Guide, but Seeker asked a Question: {utterance}")
                else:
                    stdout_logger.info(f"Description by Guide: {utterance}")
                    self.log_to_self("description", "guide")
                return utterance
            else:
                if not self.question_flag == 1:
                    self.fail = True
                    self.log_to_self("answer", "wrong response")
                    stdout_logger.info(f"Answer by Guide, but Seeker didn't asked a Question: {utterance}")
                else:
                    stdout_logger.info(f"Answer by Guide: {utterance}")
                    self.log_to_self("answer", "guide")
                return utterance

    def compute_turn_score(self):
        pass

    def compute_episode_score(self):
        pass

    def _advance_game(self, player: Union[Seeker, Guide], parsed_response: str):
        """
        Send Seeker's response to Guide and vice versa

        Args:
            player: Player object - Seeker or Guide type
            parsed_response: str - response from current Player
        """

        # First seeker turn is done, the response from seeker always goes into guide, unchanged
        # The guide response never goes into the Seeker, rather the reprompt of seeker is fixed
        # and the next possible moves are interpreted based on the guide's response
        stdout_logger.info(f"Current Round index: {self.current_round}. Current player: {player.tag}")
        # parsed response is without tags:
        utterance = self.clean_agent_response(parsed_response)
        stdout_logger.info(f"Utterance: {utterance}")

        if type(player) == Guide:
            if self.current_round == 0:  # First prompt to Seeker from Guide.
                moves = self.game_map.get_next_moves()
                self.seeker_prompt = self.seeker_base_prompt.replace(self.initial_description_tag, utterance)
                self.seeker_prompt = self.seeker_prompt.replace(self.directions_tag, moves)
                stdout_logger.info(f"First prompt for Seeker: {self.seeker_prompt}")
                stdout_logger.info(f"Image for Seeker: {self.seeker_image}")
                # Pass the response from Guide to Seeker
                self.set_context_for(self.seeker, self.seeker_prompt, image=[self.seeker_image])
            else:
                # Pass the response from Guide as is, This should only contain "ANSWER:...."
                # DESCRIPTION: ... is only for the first turn
                stdout_logger.info(f"Set Prompt for Seeker: {utterance}")
                stdout_logger.info(f"Image for Seeker: {self.seeker_image}")
                self.set_context_for(self.seeker, utterance, image=[self.seeker_image])
                
        else:
            utterance = utterance.lower()
            splits = utterance.split(":")
            tag = splits[0]

            if tag == "move" and not self.fail:
                if self.reprompt_fail:
                    # Skip updating environment, pass same image,moves, but different reprompt
                    next_moves = self.game_map.get_next_moves()  # Update next possible moves
                    stdout_logger.info(f"Next Moves: {next_moves}")
                    self.seeker_failed_reprompt = self.seeker_base_failed_reprompt.replace(self.directions_tag,
                                                                                               next_moves)
                    self.set_context_for(self.seeker, self.seeker_failed_reprompt,
                                         image=[self.seeker_image])  # Pass the updated str
                    
                    stdout_logger.info(f"Reprompt Seeker: {self.seeker_failed_reprompt}")
                    stdout_logger.info(f"Image for Seeker: {self.seeker_image}")
                    stdout_logger.info(f"Resetting reprompt_fail flag")
                    self.reprompt_fail = False
                else:
                    move = splits[1].strip().lower()
                    seeker_action = self.game_map._move_to_action[move]
                    self.game_map.step(seeker_action)  # Update Seeker state
                    # Update seeker image
                    self.seeker_image = self.game_instance["node_to_image"][str(tuple(self.game_map._agent_location))]
                    next_moves = self.game_map.get_next_moves()  # Update next possible moves
                    stdout_logger.info(f"Next Moves: {next_moves}")
                    self.seeker_reprompt = self.seeker_base_reprompt.replace(self.directions_tag, next_moves)
                    # Pass the updated str
                    self.set_context_for(self.seeker, self.seeker_reprompt, image=[self.seeker_image])
                    
                    stdout_logger.info(f"Reprompt Seeker: {self.seeker_reprompt}")
                    stdout_logger.info(f"Image for Seeker: {self.seeker_image}")
            if tag == "question":
                self.set_context_for(self.guide, utterance, image=[self.guide_image])
                
                stdout_logger.info(f"Set Prompt for Guide: {utterance}")
                stdout_logger.info(f"Image for Guide: {self.guide_image}")


    def _on_after_game(self):
        # record final results once game episode has ended:
        pass


class EscapeRoomBenchmark(GameBenchmark):
    """Integrate this game in the overall benchmark runs"""

    def __init__(self,  game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return EscapeRoom(self.game_spec, experiment=experiment, player_models=player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return EscapeRoomScorer(self.game_name, experiment, game_instance)