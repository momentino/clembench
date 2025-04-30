import re
import os.path
from typing import Dict, List, Optional

from clemcore.backends import Model
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer, GameRecorder
from clemcore.clemgame.metrics import METRIC_REQUEST_COUNT, METRIC_REQUEST_COUNT_PARSED, METRIC_REQUEST_SUCCESS, BENCH_SCORE
from clemcore.utils import file_utils

class Participant(Player):
    def _custom_response(self, context: Dict) -> str:
        pass

    def __init__(self, model: Model, game_recorder: GameRecorder):
        super().__init__(model, "Participant", game_recorder)



class LNTTest(DialogueGameMaster):
    def __init__(self, game_name: str, game_path: str, experiment: Dict, player_models: List[Model]):
        super().__init__(game_name, game_path, experiment, player_models)

    def _on_setup(self, **game_instance):
        self.game_instance = game_instance

        self.initial_prompt = self.experiment["initial_prompt"]
        self.current_task = self.game_instance["starting_rule"]
        self.sequences = self.game_instance["sequence_list"]

        self.participant = Participant(self.player_models[0], self.game_recorder)
        self.add_player(self.participant, initial_prompt=self.initial_prompt)

        self.end_reached = False
        self.num_successes_before_switch = 6
        self.successes = 0
        self.prev_successful = False

    def _on_before_game(self):
        context = f"\nSequence: {self.sequences[0]}\n"
        self.set_context_for(self.participant, context)

    def _does_game_proceed(self):
        if self.end_reached:
            self.log_to_self("all sequences done", "end game")
            return False
        return True

    def _is_vowel(self, letter: str) -> bool:
        """Check if a letter is a vowel."""
        return letter.lower() in 'aeiou'

    def _is_even(self, number: int) -> bool:
        """Check if a number is even."""
        return number % 2 == 0

    def _extract_ln_response(self, utterance: str) -> Optional[str]:
        """Extract letter-number task response."""
        matches = re.findall(r"vowel|consonant|even|odd", utterance.lower())
        return matches[0] if matches else None

    def _validate_player_response(self, player: Player, utterance: str) -> bool:
        parsed_response = self._extract_ln_response(utterance)
        if parsed_response is None:
            self.log_to_self("invalid response", utterance)
        current_sequence = str(self.sequences[self.current_round])
        letter, number = current_sequence[0], int(current_sequence[1])

        if self.current_task == 'letter':
            is_correct = ((self._is_vowel(letter) and parsed_response == 'vowel') or
                          (not self._is_vowel(letter) and parsed_response == 'consonant'))
            self.log_to_self("correct guess", utterance)
        else:  # number task
            is_correct = ((self._is_even(number) and parsed_response == 'even') or
                          (not self._is_even(number) and parsed_response == 'odd'))
            self.log_to_self("correct guess", utterance)
        if is_correct:
            self.successes+=1
            self.prev_successful = True
        else:
            self.log_to_self("wrong guess", parsed_response)
            self.prev_successful = False
        if self.successes == self.num_successes_before_switch:
            self.current_task = 'letter' if self.current_task == 'number' else 'number'
            self.successes = 0
        return True

    def _parse_response(self, player: Player, response: str) -> str:
        return self._extract_ln_response(response)

    def _on_valid_player_response(self, player: Player, parsed_response: str):
        if self.current_round < len(self.sequences) - 1 and parsed_response is not None:
            context = "Correct!\n" if self.prev_successful else "Incorrect!\n"
        elif parsed_response is None:
            context = ""
        context = context + f"\nSequence: {self.sequences[self.current_round + 1]}"
        self.set_context_for(self.participant, context)

    def _on_after_round(self):
        if self.current_round == len(self.sequences)-1:
            self.end_reached = True

class LNTTestScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_scores(self, episode_interactions: Dict) -> None:
        turn_scores = []
        invalid_response = False
        guesses = 0
        correct_guesses = 0

        for turn_idx, turn in enumerate(episode_interactions["turns"]):
            turn_score = {"request_count":1}
            correct_guess = 0
            for event_idx, event in enumerate(turn):
                action = event["action"]
                if action["type"] == "correct guess":
                    guesses += 1
                    correct_guess = 1
                    correct_guesses += 1
                if action["type"] == "wrong guess":
                    guesses += 1
                if action["type"] == "invalid response":
                    invalid_response = True
            if invalid_response:
                turn_score["parsed_request_count"] = 0
            else:
                turn_score["parsed_request_count"] = 1
            self.log_turn_score(turn_idx, 'Accuracy', 1 if correct_guess else 0)
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

        bench_score = correct_guesses / parsed_request_count
        self.log_episode_score(BENCH_SCORE, bench_score)



class LNTTestGameBenchmark(GameBenchmark):
    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return LNTTest(self.game_name, self.game_path, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return LNTTestScorer(self.game_name, experiment, game_instance)


def main():
    # select one experiment and instance
    game_path = os.path.dirname(os.path.abspath(__file__))
    experiments = file_utils.load_json("in/instances.json", game_path)
    experiment_1 = experiments["experiments"][0]
    game_1 = experiment_1["game_instances"][0]
    master = LNTTest("lnt_test", game_path, experiment_1, ["mock"])
    master.setup(**game_1)
    master.play()


if __name__ == '__main__':
    main()