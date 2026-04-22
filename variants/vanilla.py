"""Vanilla (no-op) SIREN variant. Default choice; reproduces original training."""

import torch

from . import register


@register("vanilla")
class Vanilla:
    @staticmethod
    def add_args(parser):
        pass

    @staticmethod
    def build(base_model, args):
        return base_model

    @staticmethod
    def penalty(model, args):
        return torch.zeros((), device=next(model.parameters()).device)

    @staticmethod
    def slug(args):
        return "vanilla"
