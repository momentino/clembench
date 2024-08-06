from clemgame import get_logger
from clemgame.clemgame import GameBenchmark

import json

logger = get_logger(__name__)


class GymMaster:
    def __init__(self, game_list, student_model):
        self.game_list = game_list
        self.game_names =  [g.name for g in self.game_list]
        self.student_model = student_model
        self.resource_root = f'../gym/resources'
        self.configs = self.get_configs()

    def get_configs(self):
        return json.load(open(f'{self.resource_root}/config.json'))

    def get_game_request_prompt(self):
        game_request_prompt = json.load(open(f'{self.resource_root}/{self.configs["game_request_prompt_file_name"]}'))
        game_request_prompt[0]['content'] = game_request_prompt[0]['content'].replace('[games]', f'[{", ".join(self.game_names)}]')
        return game_request_prompt

    def get_game(self):
        _, _, response_text = self.student_model.generate_response(self.get_game_request_prompt())
        return self.parse_game(response_text)

    def parse_game(self, response_text):
        # Get the game name from the model's answer, but avoid taking also a subversion of the same game (e.g. 'wordle' if the game in the answer is 'wordle_with_critic')
        parsed_game_name = [game_name for game_name in self.game_names if game_name in response_text and game_name+'_' not in response_text]
        if len(parsed_game_name) > 1:
            raise Exception("The model answer returns two games")
        elif len(parsed_game_name) == 0:
            raise Exception("The model answer returns no games")
        return parsed_game_name[0]