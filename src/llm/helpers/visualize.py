from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn


def print_module_tree(
    module: nn.Module,
    max_depth: int = 3,
    prefix: str = "",
    depth: int = 0,
):
    if depth >= max_depth:
        return

    children = list(module.named_children())

    for i, (name, child) in enumerate(children):
        last = i == len(children) - 1
        branch = "└── " if last else "├── "

        params = sum(p.numel() for p in child.parameters(recurse=False))

        print(f"{prefix}{branch}{name}: {child.__class__.__name__} [{params:,} direct params]")

        child_prefix = prefix + ("    " if last else "│   ")

        print_module_tree(
            child,
            max_depth=max_depth,
            prefix=child_prefix,
            depth=depth + 1,
        )
