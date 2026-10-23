# Antokenizer: notes for agents

Start with `src/mavt/dino_v0/README.md`: it is the current line of work (MAVT on a frozen DINOv3, v0 and v1),
with the design, file map, v0 vs v1 flags, commands, runs on alpha and results. The repo-root `README.md` and
the `MAVT_*.md` files describe older, paused pipelines.

- Tests for the current code: `python -m pytest tests/dino_v0 -q`.
- Keep v0 behaviour unchanged when the v1 flags in `src/mavt/dino_v0/config.py` are at their defaults.
- On the GPU server alpha use GPUs 0-3 only; ask an before installing packages or deleting anything;
  never write credentials into files or logs.
- On an's Mac work in `~/Workspace/AToken/atoken`; do not create new git worktrees there.
