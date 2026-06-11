# Gomoku AI Training

This folder contains the local AI experiments for the Gomoku project.

Current features:

- 15x15 Gomoku rule engine.
- Random AI.
- Heuristic rule-based AI.
- AI self-play.
- JSONL game record output.
- Lightweight trainable linear policy model.
- PyTorch CNN policy model with CUDA support.
- Shallow minimax search agent.
- Structured threat detection for double-four, four-three, and double-open-three patterns.
- Conservative VCF search for continuous-four forcing lines.
- Simplified VCT search for the basic `search` agent. The stronger local `expert` agent no longer uses this as a hard first-choice move because the simplified version was too noisy.
- Dynamic candidate width that expands by both tactical severity and the number of threat-producing candidate moves.
- Extra diagonal pressure scoring so diagonal three/four threats are less likely to be truncated out of the candidate list.
- Evaluation between trained models and rule-based agents.

The CNN model can use your NVIDIA GPU through PyTorch CUDA.

## Self-Play

Generate teacher games with the heuristic AI:

```powershell
python ai-training\scripts\self_play.py --games 50 --black heuristic --white heuristic --seed 99 --output ai-training\outputs\teacher_games.jsonl
```

## Train A Lightweight Policy

Train a linear policy model from self-play records:

```powershell
python ai-training\scripts\train_policy.py --input ai-training\outputs\teacher_games.jsonl --output ai-training\models\linear_policy.json --epochs 10 --learning-rate 0.06 --negative-samples 10
```

The model learns to rank candidate moves so that the teacher move scores higher than nearby alternatives.

## Evaluate

Evaluate the trained model against the heuristic AI:

```powershell
python ai-training\scripts\evaluate_agents.py --games 20 --agent linear:ai-training\models\linear_policy.json --opponent heuristic
```

You can also use the trained model in self-play:

```powershell
python ai-training\scripts\self_play.py --games 20 --black linear:ai-training\models\linear_policy.json --white heuristic
```

## Train CNN Policy

Generate mixed teacher games:

```powershell
python ai-training\scripts\self_play.py --games 200 --black heuristic --white heuristic --seed 314 --output ai-training\outputs\teacher_games_200.jsonl
python ai-training\scripts\self_play.py --games 200 --black heuristic-temp:8000 --white heuristic-temp:8000 --seed 2718 --output ai-training\outputs\teacher_games_temp_200.jsonl
```

Create distilled position labels from mixed games:

```powershell
python ai-training\scripts\label_positions.py --input ai-training\outputs\teacher_games_mix_400.jsonl --output ai-training\outputs\distilled_positions.jsonl
```

The browser UI can also save selected human-vs-AI games for later learning with the `纳入训练` button. Pressing the button starts recording the current game from its first move. After recording starts, undo is disabled. When either side makes five in a row, the full game is automatically appended to:

```text
ai-training/outputs/human_selected_games.jsonl
```

This file uses the same move-record shape as self-play output, with extra metadata such as `source`, `mode`, and `ai_model`.
Saving a game does not automatically retrain the CNN. Treat this file as curated raw material; for best results, convert it into labels that favor human-selected or winning-side moves instead of blindly imitating every move, because the record can include AI mistakes too.

Train the CNN:

```powershell
python ai-training\scripts\train_cnn_policy.py --input ai-training\outputs\distilled_positions.jsonl --output ai-training\models\cnn_policy_distilled.pt --epochs 10 --batch-size 128 --channels 64 --device auto
```

Evaluate the CNN:

```powershell
python ai-training\scripts\evaluate_agents.py --games 40 --agent cnn:ai-training\models\cnn_policy_distilled.pt --opponent heuristic
```

Use the CNN in self-play:

```powershell
python ai-training\scripts\self_play.py --games 20 --black cnn:ai-training\models\cnn_policy_distilled.pt --white heuristic
```

## Search Agent

The search agent adds shallow lookahead on top of tactical and heuristic scoring:

```powershell
python ai-training\scripts\evaluate_agents.py --games 20 --agent search:2:12 --opponent heuristic
```

Current search result against heuristic AI, 20 games per color:

```text
search:2:12 as black: 20 wins, 0 draws, 0 losses
search:2:12 as white: 15 wins, 0 draws, 5 losses
```

Against the distilled CNN, 10 games per color:

```text
search as black: 9 wins, 1 draw, 0 losses
search as white: 7 wins, 1 draw, 2 losses
```

The default `search` agent currently maps to `search:2:12`.
The local AI server uses `expert`, a stronger depth-3 expert-search variant for human play.
It also filters unsafe candidate moves that allow the opponent an immediate win.
Before normal search, it checks for VCF attacking moves and basic VCF defenses.
It uses dynamic width in candidate ordering. Quiet positions stay narrower; complex positions with many possible threats automatically search more candidates.
The current expert defaults are `depth=3`, `width=50`, `max_extra_width=50`, and a 26-second per-move budget. Complex positions can expand up to about 100 root candidates. The first 8 moves use a faster opening profile (`depth=2`, `width=14`, 2-second budget) after tactical checks, so early white moves stay responsive. Root candidate sorting stays broad, while deeper nodes narrow their width so search remains usable for browser play. Candidate sorting recognizes diagonal pressure and broken patterns such as `XX_X` / `X_XX`, but ordinary defensive weight is kept close to attack weight so the agent does not become overly passive against human players.

The local server also exposes a hybrid `expert-cnn` mode. It keeps the same tactical checks and expert search, but uses the distilled CNN policy as a root candidate prior. The CNN does not override forced wins, forced blocks, or tactical verification; it only nudges candidate ordering toward moves the policy model considers natural.

The engine also has a lightweight formula-opening reference layer based on the 26 common renju-style openings: 13 indirect and 13 direct patterns. These references are used as exposure and opening awareness, not as a forced book. Formula-like early positions bypass the shallow opening shortcut and receive a fuller search profile.

## Tactical Regression Suite

Run fixed tactical positions after changing search or evaluation:

```powershell
python ai-training\scripts\tactical_suite.py --agent expert
python ai-training\scripts\tactical_suite.py --agent "expert-cnn:ai-training\models\cnn_policy_distilled.pt"
```

The suite covers immediate wins, forced blocks, diagonal fours, broken diagonal `XX_X` / `X_XX` patterns, attack-over-defense selection, center opening, recorded human-game failures such as horizontal-plus-diagonal and three-four double threats, plus attacking fork cases where the agent should create cross-line double-kill pressure.

## Local AI Server

Run a local AI server for the browser UI:

```powershell
python ai-training\scripts\ai_server.py
```

The server listens at:

```text
http://127.0.0.1:8765
```

The browser can use:

```text
server-search
server-cnn
server-hybrid
```

## Current Models

Linear model:

```text
ai-training/models/linear_policy.json
```

CNN models:

```text
ai-training/models/cnn_policy.pt
ai-training/models/cnn_policy_mix.pt
ai-training/models/cnn_policy_distilled.pt
```

Latest CNN distilled training summary:

```text
examples: 32174
epochs: 10
top1 imitation accuracy: 0.784
```

Self-play continuation update:

```text
source games: ai-training/outputs/cnn_selfplay_20.jsonl
games: 20 CNN-vs-CNN
winner examples: 340
continued from: ai-training/models/cnn_policy_distilled.pt
output: ai-training/models/cnn_policy_selfplay20.pt
installed as: ai-training/models/cnn_policy_distilled.pt
backup: ai-training/models/cnn_policy_distilled_before_selfplay20_20260609_191719.pt
epochs: 2
learning rate: 0.0001
```

Formula opening reference update:

```text
reference patterns: 26
examples: 312
source: ai-training/outputs/formula_opening_positions.jsonl
output: ai-training/models/cnn_policy_opening_ref.pt
installed as: ai-training/models/cnn_policy_distilled.pt
backup: ai-training/models/cnn_policy_distilled_before_opening_ref_20260609_195749.pt
epochs: 1
learning rate: 0.00005
```

Latest evaluation against the heuristic AI, 40 games per color:

```text
CNN as black: 10 wins, 30 draws, 0 losses
CNN as white: 2 wins, 1 draw, 37 losses
```

The CNN agent has a small tactical guard: before using network logits, it first checks immediate wins, immediate blocks, open fours, and closed fours. This keeps it from missing simple forcing threats.

## Next Step

The natural next step is combining the CNN with search more tightly: use the CNN to rank candidate moves, then use shallow minimax or MCTS to verify tactics.
