# Antokenizer: notes for agents

Start with `src/mavt/dino_v0/README.md`: it is the current line of work (MAVT on a frozen DINOv3, v0 and v1),
with the design, file map, v0 vs v1 flags and commands. The repo-root `README.md` and the `MAVT_*.md` files
describe older, paused pipelines. This repo is public: keep run paths, results and internal notes out of it.

- Tests for the current code: `python -m pytest tests/dino_v0 -q`.
- Keep v0 behaviour unchanged when the v1 flags in `src/mavt/dino_v0/config.py` are at their defaults.
- On the shared training server use only this project's GPUs; ask the owner before installing packages or
  deleting anything; never write credentials into files or logs.
