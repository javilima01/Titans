"""Microbenchmark; run from the repo root with python -m tests.benchmark_titans."""

import argparse
import copy
import statistics
import time

import torch

from src.llm.modules.titans import TitansMemory
from tests.test_titans_training import reference_memory


def measure(model, inputs, forward, repeats):
    times = []
    for iteration in range(repeats + 1):
        model.zero_grad(set_to_none=True)
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)
        start = time.perf_counter()
        forward(model, inputs).square().mean().backward()
        if inputs.is_cuda:
            torch.cuda.synchronize(inputs.device)
        if iteration:
            times.append(time.perf_counter() - start)
    return statistics.median(times) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--sequence", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(1)
    memory = TitansMemory(args.dim, args.hidden, chunk_size=args.chunk).to(args.device)
    sequential = copy.deepcopy(memory)
    sequential.chunk_size = 1
    inputs = torch.randn(args.batch, args.sequence, args.dim, device=args.device)
    baseline = measure(sequential, inputs, reference_memory, args.repeats)
    same_rule = measure(memory, inputs, reference_memory, args.repeats)
    optimized = measure(memory, inputs, lambda model, x: model(x), args.repeats)
    print(f"Sequential autograd: {baseline:.2f} ms")
    print(f"Chunk-start autograd reference: {same_rule:.2f} ms")
    print(f"Optimized + checkpointing: {optimized:.2f} ms")
    print(f"Speedup over sequential: {baseline / optimized:.2f}x")
    print(
        "Forward + backward, FP32; memory module only. Chunk > 1 changes the update approximation."
    )


if __name__ == "__main__":
    main()
