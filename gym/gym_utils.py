def parse_game(game_names, response_text):
    # Get the game name from the model's answer, but avoid taking also a subversion of the same game (e.g. 'wordle' if the game in the answer is 'wordle_with_critic')
    parsed_game_name = [game_name for game_name in game_names if
                        game_name in response_text and game_name + '_' not in response_text]
    if len(parsed_game_name) > 1:
        raise Exception("The model answer returns two games")
    elif len(parsed_game_name) == 0:
        raise Exception("The model answer returns no game")
    return parsed_game_name[0]


def parse_experiment(response_text, available_experiments):
    parsed_experiment = [experiment for experiment in available_experiments if
                              experiment["name"] in response_text]
    if len(parsed_experiment) > 1:
        raise Exception("The model answer returns two experiments")
    elif len(parsed_experiment) == 0:
        raise Exception("The model answer returns no experiment")
    return parsed_experiment[0]