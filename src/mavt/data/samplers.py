from __future__ import annotations

import itertools
import random
from collections.abc import Iterator


class WeightedModalLoader:
    def __init__(self, loaders: dict[str, object], weights: dict[str, float], steps_per_epoch: int) -> None:
        self.loaders = loaders
        self.weights = weights
        self.steps_per_epoch = steps_per_epoch

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self) -> Iterator[dict]:
        names = list(self.loaders)
        probs = [self.weights.get(name, 1.0) for name in names]
        iterators = {name: itertools.cycle(loader) for name, loader in self.loaders.items()}
        for _ in range(self.steps_per_epoch):
            modality = random.choices(names, weights=probs, k=1)[0]
            yield next(iterators[modality])
