import os.path
from typing import Dict, List

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer, GameRecorder
from clemcore.clemgame.metrics import METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
from clemcore.utils import file_utils

class Participant(Player):
    def _custom_response(self, context: Dict) -> str:
        return "m"

    def __init__(self, model: Model, game_recorder: GameRecorder):
        super().__init__(model, "Participant", game_recorder)



class WorkingMemory(DialogueGameMaster):
    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance

        self.stimuli = game_instance['stimuli_list']
        self.targets = game_instance['target_list']
        self.initial_prompt = self.experiment["initial_prompt"]

        self.participant = Participant(self.player_models[0], self.game_recorder)
        self.add_player(self.participant, initial_prompt=self.initial_prompt)

        self.all_stimuli_done = False

        if "grid" in self.experiment["name"]:
            self.grids = self.experiment["grids"]

    def _on_before_game(self):
        context = self.stimuli[0] if "grid" not in self.experiment["name"] else self.grids[
            self.stimuli[0]]
        self.set_context_for(self.participant, context)

    def _does_game_proceed(self):
        if self.all_stimuli_done:
            self.log_to_self("all stimuli done", "end game")
            return False
        return True

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        if utterance == self.targets[self.current_round]:
            stimuli = self.stimuli[self.current_round] if "grid" not in self.experiment["name"] else self.grids[self.stimuli[self.current_round+1]]
            self.log_to_self("guess",stimuli)
        elif utterance not in ["m","-"]:
            self.log_to_self("invalid response",utterance)
        return True

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        if self.current_round < len(self.stimuli)-1:
            context = self.stimuli[self.current_round+1] if "grid" not in self.experiment["name"] else self.grids[self.stimuli[self.current_round+1]]
            self.set_context_for(self.participant, context)

    def _on_after_round(self):
        if self.current_round == len(self.stimuli)-1:
            self.all_stimuli_done = True

class WorkingMemoryScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        turn_scores = []
        invalid_response = False
        guesses = 0
        n_stimuli = len(self.game_instance['stimuli_list'])

        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score = {"request_count":1}
            guess = 0
            for event_idx, event in enumerate(turn):
                action = event["action"]
                if action["type"] == "guess":
                    guesses += 1
                    guess = 1
                if action["type"] == "invalid response":
                    invalid_response = True
            if invalid_response:
                turn_score["parsed_request_count"] = 1
            else:
                turn_score["parsed_request_count"] = 0
            self.log_turn_score(turn_idx, 'Accuracy', 1 if guess else 0)
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT, turn_score["request_count"])
            self.log_turn_score(turn_idx, METRIC_REQUEST_COUNT_PARSED, turn_score["parsed_request_count"])
            turn_scores.append(turn_score)
        parsed_request_count = sum(turn["parsed_request_count"] for turn in turn_scores)
        self.log_episode_score(METRIC_REQUEST_COUNT_PARSED, parsed_request_count)

        request_count = sum(turn["request_count"] for turn in turn_scores)
        self.log_episode_score(METRIC_REQUEST_COUNT, request_count)

        # Compute the request success ratio
        if request_count != 0:
            self.log_episode_score(METRIC_REQUEST_SUCCESS, parsed_request_count / request_count)
        else:
            self.log_episode_score(METRIC_REQUEST_SUCCESS, 0)

        bench_score = guesses / n_stimuli
        self.log_episode_score(BENCH_SCORE, bench_score)



class WorkingMemoryGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return WorkingMemory(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return WorkingMemoryScorer(self.game_name, experiment, game_instance)


def main():
    # select one experiment and instance
    game_path = os.path.dirname(os.path.abspath(__file__))
    experiments = file_utils.load_json("in/instances.json", game_path)
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = WorkingMemory("working_memory", game_path, experiment_1, ["mock"])
    master.setup(**game_1)
    master.play()


if __name__ == '__main__':
    main()