"""SIREN training variants (registry + dispatch).

Each variant is a class decorated with ``@register("name")`` exposing four
static methods: ``add_args``, ``build``, ``penalty``, ``slug``. See
``variants/vanilla.py`` for the minimal example and ``variants/soft_lipschitz.py``
for a variant that contributes an auxiliary training loss.
"""

REGISTRY = {}


def register(name):
    def deco(cls):
        assert name not in REGISTRY, f"variant '{name}' already registered"
        REGISTRY[name] = cls
        cls.variant_name = name
        return cls
    return deco


def available():
    return sorted(REGISTRY)


def get(name):
    return REGISTRY[name]


def add_all_variant_args(parser):
    for cls in REGISTRY.values():
        cls.add_args(parser)


def build(name, base_model, args):
    return REGISTRY[name].build(base_model, args)


def penalty(name, model, args):
    return REGISTRY[name].penalty(model, args)


def slug(name, args):
    return REGISTRY[name].slug(args)


def _extract_variant_args(args, name):
    """Return a dict with only the CLI attributes whose names start with the
    variant's argparse prefix. The prefix is inferred from the variant name
    by lowercasing and replacing underscores with the short form used in
    argparse dest (e.g. ``soft_lipschitz`` -> ``soft_lip_``)."""
    prefix_map = {
        "vanilla": None,
        "soft_lipschitz": "soft_lip_",
    }
    prefix = prefix_map.get(name)
    if prefix is None:
        return {}
    return {k: v for k, v in vars(args).items() if k.startswith(prefix)}


from . import vanilla, soft_lipschitz  # noqa: F401, E402 — registers on import
