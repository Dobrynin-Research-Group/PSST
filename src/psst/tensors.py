from math import log10
from typing import Optional
import numpy as np
import torch

from psst import Range


__all__ = ["GridTensor", "NormedTensor"]


class GridTensor(torch.Tensor):
    min_value: float = 0.0
    max_value: float = 0.0
    log_scale: bool = False

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def create(
        cls,
        min_value: float,
        max_value: float,
        steps: int,
        log_scale: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> "GridTensor":
        instance = cls()
        instance.to(device=device)
        if log_scale:
            torch.logspace(log10(min_value), log10(max_value), steps, out=instance)
        else:
            torch.linspace(min_value, max_value, steps, out=instance)
        return instance

    @classmethod
    def create_from_range(
        cls, range: Range, device: torch.device = torch.device("cpu")
    ) -> "GridTensor":
        product = 1
        for s in range.shape:
            product *= s

        t = cls.create(
            min_value=range.min_value,
            max_value=range.max_value,
            steps=product,
            log_scale=range.log_scale,
            device=device,
        ).reshape(range.shape)

        if not isinstance(t, GridTensor):
            t = GridTensor(t)
        return t


class NormedTensor(torch.Tensor):
    min_value: float = 0.0
    max_value: float = 0.0
    log_scale: bool = False
    generator: Optional[torch.Generator] = None
    is_normalized: bool = False
    difference: float = 0.0
    norm_min = 0.0
    norm_max = 0.0

    def normalize(self) -> "NormedTensor":
        if self.is_normalized:
            return self
        if self.log_scale:
            self.log10_()
        self.sub_(self.norm_min)
        self.div_(self.difference)
        return self

    def unnormalize(self) -> "NormedTensor":
        if not self.is_normalized:
            return self
        self.mul_(self.difference)
        self.add_(self.norm_min)
        if self.log_scale:
            torch.pow(10, self, out=self)
        return self

    def generate(self, normalized: bool = False) -> "NormedTensor":
        if normalized:
            self.uniform_(0, 1, generator=self.generator)
        else:
            self.uniform_(self.min_value, self.max_value, generator=self.generator)
        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = dict()
        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def create(
        cls,
        *shape: int,
        min_value: float,
        max_value: float,
        device: torch.device = torch.device("cpu"),
        log_scale: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> "NormedTensor":
        if isinstance(shape[0], tuple):
            shape = shape[0]

        instance = cls()
        torch.zeros(shape, device=device, out=instance)

        instance.min_value = min_value
        instance.max_value = max_value
        instance.norm_min = log10(min_value) if log_scale else min_value
        instance.norm_max = log10(max_value) if log_scale else max_value

        instance.log_scale = log_scale
        instance.difference = instance.max_value - instance.min_value
        if log_scale:
            instance.difference = instance.norm_max - instance.norm_min

        instance.is_normalized = False
        instance.generator = generator

        return instance

    @classmethod
    def create_from_range(
        cls,
        range: Range,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
    ) -> "NormedTensor":
        return cls.create(
            *range.shape,
            min_value=range.min_value,
            max_value=range.max_value,
            log_scale=range.log_scale,
            device=device,
            generator=generator,
        )

    @classmethod
    def create_from_numpy(
        cls,
        arr: np.ndarray,
        min_value: float,
        max_value: float,
        log_scale: bool = False,
        is_normalized: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> "NormedTensor":
        instance = cls(torch.as_tensor(arr), device=device)
        instance.min_value = min_value
        instance.max_value = max_value
        instance.log_scale = log_scale
        instance.is_normalized = is_normalized

        return instance
