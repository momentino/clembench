from clemgame import get_logger
from gym.gym_utils import parse_experiment, parse_game

import json

logger = get_logger(__name__)


class GymMaster:
    def __init__(self, game_list, student_model):
        self.game_list = game_list
        self.game_names = ["taboo"] #[g.name for g in self.game_list]
        self.student_model = student_model
        self.resource_root = f'../gym/resources'
        self.configs = self.get_configs()

    def get_configs(self):
        return json.load(open(f'{self.resource_root}/config.json'))

    def get_game_selection_prompt(self):
        game_selection_prompt = json.load(open(f'{self.resource_root}/{self.configs["game_selection_prompt_file_name"]}'))
        game_selection_prompt[0]['content'] = game_selection_prompt[0]['content'].replace('[games]', f'[{", ".join(self.game_names)}]')
        return game_selection_prompt

    def get_experiment_selection_prompt(self, game_name, available_experiment_names):
        experiment_selection_prompt = json.load(open(f'{self.resource_root}/{self.configs["experiment_selection_prompt_file_name"]}'))
        experiment_selection_prompt[0]['content'] = experiment_selection_prompt[0]['content'].replace('[game]', f'"{game_name}"').replace('[available_experiments]', f'[{", ".join(available_experiment_names)}]')
        return experiment_selection_prompt

    def get_game(self):
        _, _, response_text = self.student_model.generate_response(self.get_game_selection_prompt())
        return parse_game(self.game_names, response_text)

    def get_experiment(self, game_name, available_experiments):
        # todo add a description of each experiment to present the model?
        available_experiment_names = [experiment['name'] for experiment in available_experiments]
        _, _, response_text = self.student_model.generate_response(self.get_experiment_selection_prompt(game_name, available_experiment_names))
        return parse_experiment(response_text, available_experiments)



