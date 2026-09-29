"""Train/test an activation-driven fast network on previously unseen fact values.

The writer sees only a supplied statement. Slow projections learn across
training episodes; fast weights learn observed tokens at test time. A question
is then decoded with no source text or KV cache. Latest-support selection is an
oracle in this diagnostic, so it does not test importance or conflict detection.
"""

import argparse
import json
from pathlib import Path
import random
import time

from safetensors.torch import load_file, save_file
import torch
from torch.nn import functional

from src.llm.helpers.neural_memory_probe import SYSTEM
from src.llm.helpers.neural_statement_probe import record_probe
from src.llm.modules.activation_memory import ActivationFastMemory
from src.llm.modules.qwen import Qwen35Wrapper


def question_ids(model, question):
    return model.tokenizer.encode(
        model.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        ),
        add_special_tokens=False,
    )


@torch.no_grad()
def token_context(model, ids):
    """Causal pooled input activations; retains identity cues without any text index."""
    embedded = model.model.model.embed_tokens(torch.tensor([ids], device=model.device))[0].float()
    pooled = functional.normalize(embedded, dim=-1).cumsum(0)
    return (functional.normalize(pooled, dim=-1) * embedded.shape[-1] ** 0.5).cpu()


@torch.no_grad()
def features(model, ids, *, context_pool=False):
    output = model.model.model(
        input_ids=torch.tensor([ids], device=model.device),
        use_cache=False,
        output_hidden_states=True,
    )
    # All 24 block outputs contribute, including DeltaNet and full attention.
    # The final normalized representation remains separately available.
    states = output.hidden_states[1:]
    pooled = torch.stack([functional.normalize(state[0].float(), dim=-1) for state in states]).mean(
        0
    )
    pooled = functional.normalize(pooled, dim=-1) * output.last_hidden_state.shape[-1] ** 0.5
    result = torch.cat((output.last_hidden_state[0].float(), pooled), dim=-1).cpu()
    return torch.cat((result, token_context(model, ids)), dim=-1) if context_pool else result


def augment_cache(args):
    model = Qwen35Wrapper(device=args.device)
    tensors = load_file(args.cache / "features.safetensors")
    metadata = json.loads((args.cache / "cases.json").read_text())
    for row in metadata:
        prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
        ids = model.tokenizer.encode(prefix + row["source"], add_special_tokens=False)
        source_name, query_name = f"{row['key']}.source_hidden", f"{row['key']}.query_hidden"
        context = token_context(model, ids)[-len(tensors[source_name]) :]
        tensors[source_name] = torch.cat((tensors[source_name], context.half()), dim=-1)
        targets = tensors[f"{row['key']}.target_ids"].tolist()
        query_prefix = question_ids(model, row["question"])
        context = token_context(model, query_prefix + targets[:-1])[len(query_prefix) - 1 :]
        tensors[query_name] = torch.cat((tensors[query_name], context.half()), dim=-1)
    save_file(tensors, args.output / "features.safetensors")
    (args.output / "cases.json").write_text(json.dumps(metadata, indent=2) + "\n")
    record_probe(
        args.output,
        {
            "architecture": "Qwen block activations plus causal mean of input-token embeddings",
            "parent": str(args.cache),
            "count": len(metadata),
        },
        note="Adds causal token-context activations to test whether identity information was lost before the fast-memory writer/read projections. No explicit entity extraction or stored identity index.",
    )


def prepare(model, row):
    start, end = row["metadata"]["support_spans"][-1]
    source = row["context"][start:end]
    answer = row["answers"][0]
    prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
    encoded = model.tokenizer(
        prefix + source, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = encoded["input_ids"]
    answer_start = len(prefix) + source.rindex(answer)
    answer_end = answer_start + len(answer)
    positions = [
        i
        for i, (a, b) in enumerate(encoded["offset_mapping"])
        if b > answer_start and a < answer_end
    ]
    targets = [ids[i] for i in positions] + [model.tokenizer.eos_token_id]
    ids.append(model.tokenizer.eos_token_id)
    source_start = next(i for i, (a, b) in enumerate(encoded["offset_mapping"]) if b > len(prefix))
    # Activation before token t predicts observed token t; never include t in its key.
    source_ids = ids[source_start:]
    source_positions = [i - source_start for i in positions] + [len(source_ids) - 1]
    source_hidden = features(model, ids[:-1])[source_start - 1 :]
    query_prefix = question_ids(model, row["question"])
    query_hidden = features(model, query_prefix + targets[:-1])[len(query_prefix) - 1 :]
    return {
        "source_hidden": source_hidden,
        "query_hidden": query_hidden,
        "source_ids": torch.tensor(source_ids),
        "target_ids": torch.tensor(targets),
        "target_positions": torch.tensor(source_positions),
    }, {
        "id": row["id"],
        "source": source,
        "question": row["question"],
        "answer": answer,
        "task": row["metadata"]["task"],
    }


@torch.no_grad()
def observe(model, memory, statement: str):
    """Update only fast weights from a live statement, with no question/answer labels."""
    prefix = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n"
    encoded = model.tokenizer(
        prefix + statement, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = [*encoded["input_ids"], model.tokenizer.eos_token_id]
    if len(ids) > 512:
        msg = "This live probe supports statements up to 512 Qwen tokens"
        raise ValueError(msg)
    start = next(i for i, (_, end) in enumerate(encoded["offset_mapping"]) if end > len(prefix))
    hidden = features(model, ids[:-1], context_pool=memory.to_key[0].normalized_shape[0] == 3072)[
        start - 1 :
    ]
    values = (
        model.model.lm_head.weight[torch.tensor(ids[start:], device=model.device)].cpu().float()
    )
    return memory.write(hidden, values)


def session(args):
    """Separate writer/reader processes; readers require no source-text artifacts."""
    model = Qwen35Wrapper(device=args.device)
    config = json.loads((args.checkpoint / "network_config.json").read_text())
    memory = ActivationFastMemory(**config)
    memory.load_state_dict(load_file(args.checkpoint / "network.safetensors"))
    memory.eval()
    cases = json.loads(args.cases.read_text())
    results = []
    if args.stage == "session-write":
        results.extend(
            {"source": statement, **observe(model, memory, statement)}
            for statement in cases["sources"]
        )
        save_file(memory.state_dict(), args.output / "network.safetensors")
        (args.output / "network_config.json").write_text(json.dumps(config, indent=2) + "\n")
        note = f"Wrote {len(results)} live statements directly from Qwen activations; no answer annotations or generated questions. Only neural and optimizer tensors are needed for subsequent sessions."
    else:
        for case in cases["reads"]:
            prediction = answer(model, memory, case["question"], scale=args.scale)
            row = case | {
                "prediction": prediction,
                "correct": prediction == case["answer"],
                "contains_answer": case["answer"] in prediction
                if case["answer"]
                else not prediction,
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        note = f"Fresh process, neural state only: {sum(row['correct'] for row in results)}/{len(results)} exact and {sum(row['contains_answer'] for row in results)}/{len(results)} contain the supplied reference value. The latter is a lenient diagnostic, not a semantic correctness guarantee."
    record_probe(
        args.output,
        {
            "architecture": "ActivationFastMemory",
            "checkpoint": str(args.checkpoint),
            "cases": str(args.cases),
            "stage": args.stage,
            "results": results,
            "scale": args.scale,
        },
        note=note,
    )


def cache(args):
    model = Qwen35Wrapper(device=args.device)
    tensors, metadata = {}, []
    started = time.monotonic()
    for split, limit in (
        ("train", args.train_limit),
        ("validation", args.eval_limit),
        ("test", args.eval_limit),
    ):
        rows = [
            json.loads(line) for line in (args.data / f"{split}.jsonl").read_text().splitlines()
        ][:limit]
        for index, row in enumerate(rows):
            data, meta = prepare(model, row)
            key = f"{split}.{index}"
            for name, value in data.items():
                tensors[f"{key}.{name}"] = value.half() if value.is_floating_point() else value
            metadata.append(meta | {"key": key, "split": split})
            if (index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "split": split,
                            "prepared": index + 1,
                            "seconds": time.monotonic() - started,
                        }
                    ),
                    flush=True,
                )
    used_ids = torch.unique(
        torch.cat([value for key, value in tensors.items() if key.endswith("source_ids")])
    )
    tensors["vocabulary_ids"] = used_ids
    tensors["vocabulary_embeddings"] = functional.normalize(
        model.model.lm_head.weight[used_ids.to(model.device)].cpu().float(), dim=-1
    )
    save_file(tensors, args.output / "features.safetensors")
    (args.output / "cases.json").write_text(json.dumps(metadata, indent=2) + "\n")
    record_probe(
        args.output,
        {
            "architecture": "Qwen final activations plus normalized mean of all 24 block outputs",
            "count": len(metadata),
            "elapsed_seconds": time.monotonic() - started,
            "data": str(args.data),
        },
        note="Feature cache for outer learning. Source statements and teacher-forced query prefixes are training/evaluation artifacts, never persistent inference memory. The latest supporting statement is selected by benchmark annotations.",
    )


def batch(tensors, names):
    groups = [[name] if isinstance(name, str) else name for name in names]
    source_len = max(sum(len(tensors[f"{name}.source_ids"]) for name in group) for group in groups)
    query_len = max(sum(len(tensors[f"{name}.target_ids"]) for name in group) for group in groups)
    hidden_width = tensors[f"{groups[0][0]}.source_hidden"].shape[-1]
    source = torch.zeros(len(names), source_len, hidden_width)
    query = torch.zeros(len(names), query_len, hidden_width)
    values = torch.zeros(len(names), source_len, 1024)
    targets = torch.zeros(len(names), query_len, 1024)
    source_mask = torch.zeros(len(names), source_len, dtype=torch.bool)
    query_mask = torch.zeros(len(names), query_len, dtype=torch.bool)
    labels = torch.zeros(len(names), query_len, dtype=torch.long)
    for i, group in enumerate(groups):
        source_ids = torch.cat([tensors[f"{name}.source_ids"] for name in group])
        target_ids = torch.cat([tensors[f"{name}.target_ids"] for name in group])
        sl, ql = len(source_ids), len(target_ids)
        source[i, :sl] = torch.cat([tensors[f"{name}.source_hidden"] for name in group]).float()
        query[i, :ql] = torch.cat([tensors[f"{name}.query_hidden"] for name in group]).float()
        source_mask[i, :sl] = True
        query_mask[i, :ql] = True
        values[i, :sl] = tensors["vocabulary_embeddings"][
            torch.searchsorted(tensors["vocabulary_ids"], source_ids)
        ]
        targets[i, :ql] = tensors["vocabulary_embeddings"][
            torch.searchsorted(tensors["vocabulary_ids"], target_ids)
        ]
        offset, positions = 0, []
        for name in group:
            positions.append(tensors[f"{name}.target_positions"] + offset)
            offset += len(tensors[f"{name}.source_ids"])
        labels[i, :ql] = torch.cat(positions)
    return source, query, values, targets, source_mask, query_mask, labels


def group_names(names, count):
    groups = []
    for name in names:
        pair = int(name.rsplit(".", 1)[1]) // 2
        if (
            not groups
            or len(groups[-1]) >= count
            or any(int(item.rsplit(".", 1)[1]) // 2 == pair for item in groups[-1])
        ):
            groups.append([])
        groups[-1].append(name)
    return groups


def train(args):
    torch.manual_seed(902)
    tensors = load_file(args.cache / "features.safetensors")
    metadata = json.loads((args.cache / "cases.json").read_text())
    names = [row["key"] for row in metadata if row["split"] == "train"]
    validation_names = [row["key"] for row in metadata if row["split"] == "validation"]
    validation = group_names(validation_names[::2] + validation_names[1::2], args.facts_per_memory)
    input_width = tensors[f"{names[0]}.source_hidden"].shape[-1]
    memory = ActivationFastMemory(input_width=input_width, key_width=args.key_width)
    optimizer = torch.optim.AdamW([p for p in memory.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(902)
    started = time.monotonic()
    events = []
    best_loss, best_epoch = float("inf"), 0

    def loss_for(selected):
        source, query, values, targets, sm, qm, labels = batch(tensors, selected)
        predicted, attention = memory.episode_read(source, query, values, sm)
        reconstruction = (1 - functional.cosine_similarity(predicted, targets, dim=-1))[qm].mean()
        logits = (attention / 0.07).masked_fill(~sm[:, None, :], -1e4)
        address = functional.cross_entropy(logits[qm], labels[qm])
        accuracy = (logits.argmax(-1)[qm] == labels[qm]).float().mean()
        return reconstruction + address, reconstruction, accuracy

    for epoch in range(args.epochs):
        rng.shuffle(names)
        groups = group_names(names, args.facts_per_memory)
        losses = []
        for start in range(0, len(groups), args.batch_size):
            optimizer.zero_grad()
            loss, _, _ = loss_for(groups[start : start + args.batch_size])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(memory.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        with torch.no_grad():
            scores = [
                loss_for(validation[i : i + args.batch_size])
                for i in range(0, len(validation), args.batch_size)
            ]
        event = {
            "epoch": epoch + 1,
            "loss": sum(losses) / len(losses),
            "validation_loss": sum(float(row[0]) for row in scores) / len(scores),
            "validation_address_accuracy": sum(float(row[2]) for row in scores) / len(scores),
            "seconds": time.monotonic() - started,
        }
        events.append(event)
        if event["validation_loss"] < best_loss:
            best_loss, best_epoch = event["validation_loss"], epoch + 1
            save_file(memory.state_dict(), args.output / "network.safetensors")
        print(json.dumps(event), flush=True)
        with (args.output / "training.jsonl").open("a") as file:
            file.write(json.dumps(event) + "\n")
    save_file(memory.state_dict(), args.output / "final_network.safetensors")
    (args.output / "network_config.json").write_text(
        json.dumps({"input_width": input_width, "key_width": args.key_width}, indent=2) + "\n"
    )
    record_probe(
        args.output,
        {
            "architecture": "ActivationFastMemory with differentiable inner ridge solve and auxiliary token addressing loss",
            "events": events,
            "training_examples": len(names),
            "selected_epoch": best_epoch,
            "cache": str(args.cache),
            "config": {
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "facts_per_memory": args.facts_per_memory,
            },
        },
        note="Slow read/write projections trained through an inner fast-weight solve on observed statement tokens. No test-time question/answer demonstrations or separate semantic encoder. Generation still requires independent evaluation; address accuracy is not answer accuracy.",
    )


@torch.no_grad()
def answer(model, memory, question, *, scale, max_tokens=24):
    ids = question_ids(model, question)
    generated = []
    norms = model.model.lm_head.weight.norm(dim=-1).clamp_min(1e-8)
    for _ in range(max_tokens):
        hidden = features(model, ids, context_pool=memory.to_key[0].normalized_shape[0] == 3072)[-1]
        prediction = functional.normalize(memory(hidden), dim=-1).to(model.device)
        logits = model.model.lm_head(prediction) / norms
        if scale is not None:
            logits = logits * scale + model.model.lm_head(hidden[:1024].to(model.device))
        token = int(logits.argmax())
        if token == model.tokenizer.eos_token_id:
            break
        generated.append(token)
        ids.append(token)
    return model.tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate(args):
    model = Qwen35Wrapper(device=args.device)
    config = json.loads((args.checkpoint / "network_config.json").read_text())
    torch.manual_seed(902)
    memory = ActivationFastMemory(**config)
    if not args.untrained:
        memory.load_state_dict(load_file(args.checkpoint / "network.safetensors"))
    memory.eval()
    tensors = load_file(args.cache / "features.safetensors")
    rows = [
        row
        for row in json.loads((args.cache / "cases.json").read_text())
        if row["split"] == args.split
    ][: args.eval_limit]
    results = []
    started = time.monotonic()
    lookup = {row["key"]: row for row in rows}
    names = list(lookup)
    groups = group_names(names[::2] + names[1::2], args.facts_per_memory)
    for group in groups:
        memory.reset()
        writes = []
        for name in group:
            hidden = tensors[f"{name}.source_hidden"].float()
            ids = tensors[f"{name}.source_ids"]
            values = tensors["vocabulary_embeddings"][
                torch.searchsorted(tensors["vocabulary_ids"], ids)
            ]
            writes.append(memory.write(hidden, values))
        # A fresh module carries only slow weights, fast weights and optimizer state.
        save_file(memory.state_dict(), args.output / "last_fast_state.safetensors")
        restored = ActivationFastMemory(**config)
        restored.load_state_dict(load_file(args.output / "last_fast_state.safetensors"))
        for name in group:
            row = lookup[name]
            prediction = answer(model, restored, row["question"], scale=args.scale)
            result = row | {
                "prediction": prediction,
                "correct": prediction == row["answer"],
                "contains_answer": row["answer"] in prediction,
                "writes": writes,
                "facts_in_memory": len(group),
            }
            results.append(result)
            print(
                json.dumps({key: result[key] for key in ("id", "answer", "prediction", "correct")}),
                flush=True,
            )
    correct = sum(row["correct"] for row in results)
    record_probe(
        args.output,
        {
            "architecture": "ActivationFastMemory",
            "results": results,
            "correct": correct,
            "count": len(results),
            "checkpoint": str(args.checkpoint),
            "split": args.split,
            "scale": args.scale,
            "untrained_projections": args.untrained,
            "facts_per_memory": args.facts_per_memory,
            "elapsed_seconds": time.monotonic() - started,
        },
        note=f"{correct}/{len(results)} exact answers on novel values after statement-only activation writes and tensor-state reload, with up to {args.facts_per_memory} coexisting facts. Importance and corrections are not tested. Evaluation queries are not used during writes.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("cache", "augment-cache", "train", "evaluate", "session-write", "session-read"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(".datasets_cache/session-open-values-v1"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--eval-limit", type=int, default=80)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--key-width", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--facts-per-memory", type=int, default=1)
    parser.add_argument("--untrained", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new experiment folder")
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    {
        "cache": cache,
        "augment-cache": augment_cache,
        "train": train,
        "evaluate": evaluate,
        "session-write": session,
        "session-read": session,
    }[args.stage](args)


if __name__ == "__main__":
    main()
