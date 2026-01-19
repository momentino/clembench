from pathlib import Path
import json

base_path = Path(__file__).parent / "results"
output_path = Path(__file__).parent

for model_dir in [p for p in base_path.iterdir() if p.is_dir()]:
    model_name = model_dir.name
    games_dict = {}

    for game_dir in [p for p in model_dir.iterdir() if p.is_dir()]:
        game_name = game_dir.name
        if game_name != "sudokugame":
            n_episodes = 0
            n_played_eps = 0
            aborted_games = 0
            main_scores = 0.0

            for score_file in game_dir.rglob("scores.json"):
                n_episodes += 1
                with score_file.open() as f:
                    data = json.load(f)

                episode_scores = data.get("episode scores", {})
                aborted = episode_scores.get("Aborted", 0)

                if aborted == 0:
                    n_played_eps += 1
                    main_scores += episode_scores.get("Main Score", 0)
            if n_episodes > 0:
                quality_score = 0 if n_played_eps == 0 else (main_scores / n_played_eps) / 100
                p_played = n_played_eps / n_episodes
                clemscore = quality_score * p_played

                games_dict[game_name] = {
                    "quality_score": quality_score,
                    "p_played": p_played,
                    "clemscore": clemscore
                }
    print("Saving file..")
    with (output_path / f"{model_name}.json").open("w") as f:
        json.dump(games_dict, f, indent=2)