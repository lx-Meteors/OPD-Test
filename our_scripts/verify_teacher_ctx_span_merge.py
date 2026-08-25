"""Check that the windowed teacher is wired up and that the span merge changes nothing.

The windowed teacher chunks the response into read-out spans and re-encodes
``prompt + response[ctx_start:end]`` for each. Truncation only bites once a span
starts past ``window``, so the span starting at exactly ``window`` still carries
``ctx_start == 0`` and re-encodes the same prefix as the very first span. Emitting the
two as one span is meant to be free: same teacher input, one pass fewer.

Four checks, only the last needing a GPU.

  0. On the plumbing. The window has to survive four hops -- dataclass field, shell
     hydra override, trainer meta_info, reward worker -- and a break in any of them is
     silent: the run name still says onset_6144, the startup banner still prints
     W=2048, teacher_ctx/* metrics still appear, and the teacher still quietly sees the
     full prefix. The meta_info hop is the dangerous one, because the reward worker's
     fallback is ``self.config``, which is ``config.reward_model`` and has no such
     field, so the fallback silently reads 0 rather than failing.

  1. On the span arithmetic alone, over a grid of (response length, W, S): every
     read-out position must get the same ctx_start under both schemes, and the
     read-out must stay a partition of the response. Also reports the cost in
     re-encoded response tokens, which is what the merge is buying.

  2. On the rank invariance of the teacher forward count. The span list is fixed by the
     padded response length and the two knobs, so it is the same everywhere, but the
     rows within a span are the local sequences reaching that depth, and past the onset
     that count is data dependent and often zero. Each teacher forward is an FSDP
     all-gather over the whole group, so ranks that disagree on how many to issue hang
     the job with the GPUs pinned. Neither the arithmetic check nor a single-process
     forward can see this, so it is checked directly.

  3. On the real teacher weights, comparing the per-token teacher log probs that OPD
     actually consumes. The two schemes split at the onset:

       depth >= onset  both schemes feed the model an elementwise identical tensor of
                       identical shape, so the read-out must agree to the last bit.

       depth <  onset  both schemes see the untruncated prefix and are mathematically
                       equal to a plain full-context pass, but the merged scheme reads
                       these positions out of a longer tensor (prompt + W + S rather
                       than prompt + W). Causal attention makes that irrelevant
                       mathematically; kernels may still reduce in a different order.
                       So this range is checked against a plain full-context pass, run
                       for both schemes, and the two deviations are compared. The
                       merge is sound if it is no further from the plain pass than the
                       legacy scheme already was.

     A ragged batch is included so the per-span row selection is exercised.

Usage:
    python our_scripts/verify_teacher_ctx_span_merge.py                  # all four
    python our_scripts/verify_teacher_ctx_span_merge.py --no-model       # skip GPU part
"""

import argparse
import ast
import pathlib
import re

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FSDP_WORKERS = REPO_ROOT / "verl/verl/workers/fsdp_workers.py"
RAY_TRAINER = REPO_ROOT / "verl/verl/trainer/ppo/ray_trainer.py"
ROLLOUT_CONFIG = REPO_ROOT / "verl/verl/workers/config/rollout.py"
SHELL_COMMONS = [REPO_ROOT / "experiments_scripts/common.sh", REPO_ROOT / "our_scripts/opd_common.sh"]
# The teacher of the qwen-4b OPD runs, at the path the experiment scripts use.
DEFAULT_MODEL = str(REPO_ROOT / "models/Qwen3-4B-Non-Thinking-RL-Math-Step500")

WINDOW_KEYS = ("teacher_ctx_window", "teacher_ctx_segment")


def load_production_spans():
    """Pull `teacher_ctx_spans` out of fsdp_workers.py without importing verl.

    verl drags in ray/vllm, which need not be installed to check arithmetic. The
    function is self-contained, so exec'ing just its AST node keeps this test bound to
    the shipped implementation instead of a copy that can drift.
    """
    tree = ast.parse(FSDP_WORKERS.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "teacher_ctx_spans":
            node.decorator_list = []
            ns = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(FSDP_WORKERS), "exec"), ns)
            return ns["teacher_ctx_spans"]
    raise RuntimeError(f"teacher_ctx_spans not found in {FSDP_WORKERS}")


def _func_node(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise RuntimeError(f"function {name} not found")


def meta_info_reads(tree, func_name):
    """Keys a function pulls out of meta_info, as `<obj>.meta_info.get("key", ...)`."""
    keys = set()
    for node in ast.walk(_func_node(tree, func_name)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "get" or not isinstance(node.func.value, ast.Attribute):
            continue
        if node.func.value.attr != "meta_info" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            keys.add(node.args[0].value)
    return keys


def meta_info_writes(tree):
    """Keys assigned anywhere in a module as `<obj>.meta_info["key"] = ...`."""
    keys = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)):
                continue
            if target.value.attr != "meta_info":
                continue
            if isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
                keys.add(target.slice.value)
    return keys


def dataclass_fields(tree, class_name):
    node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == class_name), None
    )
    if node is None:
        raise RuntimeError(f"class {class_name} not found")
    return {
        stmt.target.id
        for stmt in node.body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }


def check_plumbing():
    print("=" * 78)
    print("Part 0: plumbing")
    print("=" * 78)
    ok = True

    fields = dataclass_fields(ast.parse(ROLLOUT_CONFIG.read_text()), "RolloutConfig")
    for key in WINDOW_KEYS:
        present = key in fields
        ok &= present
        print(f"  {'ok ' if present else 'MISSING'} RolloutConfig declares {key}")

    for path in SHELL_COMMONS:
        text = path.read_text()
        for key in WINDOW_KEYS:
            env = key.upper()
            exported = re.search(rf'^\s*export {env}="\$\{{{env}:-', text, re.M) is not None
            passed = f"+actor_rollout_ref.rollout.{key}=${{{env}}}" in text
            ok &= exported and passed
            print(f"  {'ok ' if exported and passed else 'BROKEN '} {path.name}: {env} "
                  f"{'exported' if exported else 'NOT EXPORTED'}, override "
                  f"{'passed' if passed else 'NOT PASSED'}")

    # The hop that fails silently. Every key the reward worker reads out of meta_info
    # has to be put there by the trainer first; its own config fallback is the wrong
    # config object, so a missing key reads as a default instead of raising.
    reads = meta_info_reads(ast.parse(FSDP_WORKERS.read_text()), "compute_rm_score")
    writes = meta_info_writes(ast.parse(RAY_TRAINER.read_text()))
    missing = sorted(reads - writes)
    print()
    print(f"  compute_rm_score reads {len(reads)} meta_info keys: {sorted(reads)}")
    print(f"  ray_trainer writes {len(writes)} meta_info keys")
    if missing:
        ok = False
        print(f"  BROKEN  never written by the trainer, will silently take defaults: {missing}")
    else:
        print("  ok      every key read is written by the trainer")

    print()
    assert ok, "windowed teacher plumbing is broken, see BROKEN/MISSING lines above"
    print("PASS: config field -> shell override -> meta_info -> reward worker is intact.")


def spans_legacy(resp_len, window, segment):
    """The pre-merge scheme: the first span stops at `window`, then fixed segments."""
    spans, cur = [], min(window, resp_len)
    if cur > 0:
        spans.append((0, cur, 0))
    while cur < resp_len:
        end = min(cur + segment, resp_len)
        spans.append((cur, end, max(0, cur - window)))
        cur = end
    return spans


def ctx_start_per_position(spans, resp_len):
    """Map each read-out position to the ctx_start of the span that produced it."""
    out = [None] * resp_len
    for start, end, ctx_start in spans:
        for p in range(start, end):
            assert out[p] is None, f"position {p} read out twice"
            out[p] = ctx_start
    assert all(v is not None for v in out), "read-out does not cover the response"
    return out


def cost_tokens(spans, prompt_len):
    """Tokens fed to the teacher, prompt re-encoding included, over all passes."""
    return sum(prompt_len + end - ctx_start for _, end, ctx_start in spans)


def check_spans(spans_new, prompt_len, resp_len):
    print("=" * 78)
    print("Part 1: span arithmetic")
    print("=" * 78)
    grid = [
        # The qwen-4b geometry: MAX_PROMPT_LENGTH 2048, MAX_RESP_LENGTH 16384.
        (2048, 16384, 4096, 4096),
        (2048, 16384, 2048, 4096),
        (2048, 16384, 1024, 4096),
        (2048, 16384, 4096, 2048),
        (2048, 16384, 8192, 4096),
        # The 1.5b geometry these settings were first measured on.
        (1024, 12288, 4096, 4096),
        (1024, 12288, 2048, 4096),
        (1024, 12288, 1024, 4096),
        # Degenerate cases: onset at or past the response, tiny responses, S = 1.
        (1024, 8192, 4096, 4096),
        (2048, 16384, 16384, 4096),
        (1024, 100, 4096, 4096),
        (1024, 12288, 1, 1),
    ]
    if (prompt_len, resp_len) not in {(p, l) for p, l, _, _ in grid}:
        grid.insert(0, (prompt_len, resp_len, 2048, 4096))

    print(f"{'P':>5} {'L':>6} {'W':>6} {'S':>6} {'onset':>6} | "
          f"{'passes':>6} {'cost':>7} {'ratio':>5} | {'passes':>6} {'cost':>7} {'ratio':>5} | "
          f"{'saved':>6} equal")
    print(f"{'':>5} {'':>6} {'':>6} {'':>6} {'':>6} | {'merged':>6} {'':>7} {'':>5} | "
          f"{'legacy':>6} {'':>7} {'':>5} |")
    all_equal = True
    for p_len, r_len, window, segment in grid:
        seg = max(1, min(segment, r_len))
        new = spans_new(r_len, window, seg)
        old = spans_legacy(r_len, window, seg)
        equal = ctx_start_per_position(new, r_len) == ctx_start_per_position(old, r_len)
        all_equal &= equal
        c_new, c_old = cost_tokens(new, p_len), cost_tokens(old, p_len)
        plain = p_len + r_len
        print(
            f"{p_len:>5} {r_len:>6} {window:>6} {segment:>6} {window + segment:>6} | "
            f"{len(new):>6} {c_new:>7} {c_new / plain:>5.2f} | "
            f"{len(old):>6} {c_old:>7} {c_old / plain:>5.2f} | "
            f"{1 - c_new / c_old:>5.1%}  {'yes' if equal else 'NO'}"
        )
    print()
    print("cost = teacher tokens per sequence including re-encoded prompt.")
    print("ratio = against the plain untruncated pass (= P + L).")
    print(f"per-position ctx_start identical everywhere: {all_equal}")
    assert all_equal, "merge changed which context a position is read out under"
    return all_equal


def _span_logprobs(model, ids, mask, targets, lo, chunk=2048):
    """Log prob of `targets` read out of one forward pass, starting at logits index lo.

    Chunked because a full [rows, seq, vocab] log_softmax in fp32 is tens of GB.
    """
    n = targets.size(1)
    picked = torch.empty(targets.shape, dtype=torch.float32, device=targets.device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask).logits
        for i in range(0, n, chunk):
            j = min(i + chunk, n)
            logp = torch.log_softmax(logits[:, lo + i : lo + j].float(), dim=-1)
            picked[:, i:j] = logp.gather(-1, targets[:, i:j].unsqueeze(-1)).squeeze(-1)
    return picked


def teacher_logprobs(model, prompt_ids, prompt_mask, responses, resp_mask, spans):
    """Per-token teacher log prob of the student's tokens, span scheme as given.

    Mirrors _teacher_forward_windowed: one pass per span over
    `prompt + response[ctx_start:end]`, read-out kept only for [start, end), rows
    selected by the mask at the span start.
    """
    bsz, resp_len = responses.shape
    prompt_len = prompt_ids.size(1)
    out = torch.zeros(bsz, resp_len, dtype=torch.float32, device=responses.device)
    passes = 0
    tokens = 0
    for start, end, ctx_start in spans:
        keep = resp_mask[:, start].bool()
        if not bool(keep.any()):
            continue
        rows = keep.nonzero(as_tuple=True)[0]
        ids = torch.cat([prompt_ids[rows], responses[rows, ctx_start:end]], dim=1)
        mask = torch.cat([prompt_mask[rows], resp_mask[rows, ctx_start:end]], dim=1)
        passes += 1
        tokens += ids.numel()
        # logits[:, i] predicts ids[:, i + 1]; response position p sits at index
        # prompt_len + p - ctx_start, so it is predicted from one index earlier.
        out[rows, start:end] = _span_logprobs(
            model, ids, mask, responses[rows, start:end], prompt_len + start - ctx_start - 1
        )
    return out, passes, tokens


def check_model(spans_new, model_path, resp_len, prompt_len, window, segment, short_len, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print()
    print("=" * 78)
    print(f"Part 3: real teacher weights ({model_path}, {dtype})")
    print("=" * 78)

    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch_dtype, attn_implementation="sdpa"
    ).to("cuda")
    model.eval()

    # Real text rather than random ids, so the teacher is on a plausible manifold and
    # the log probs being compared are not degenerate.
    seed_text = (
        "Solve the problem step by step. Let n be a positive integer and consider the "
        "sum of the first n odd numbers. We have 1 + 3 + 5 + ... + (2n-1) = n^2. "
        "To see why, pair the first and last terms: their sum is 2n, and there are n/2 "
        "such pairs. Therefore the total is n^2. Now suppose instead we are asked for "
        "the sum of the first n even numbers, which is n(n+1). Checking n = 3 gives "
        "2 + 4 + 6 = 12 = 3 * 4, as expected. Wait, let me double check the odd case "
        "for n = 4: 1 + 3 + 5 + 7 = 16 = 4^2. Good. "
    )
    offset = 512
    need = prompt_len + resp_len + offset
    per_rep = len(tok(seed_text).input_ids)
    ids = tok(seed_text * (need // per_rep + 4), return_tensors="pt").input_ids[0]
    assert ids.numel() >= need, f"seed text too short: {ids.numel()} < {need}"

    row_a = ids[:prompt_len + resp_len]
    row_b = ids[offset:offset + prompt_len + resp_len]
    prompt_ids = torch.stack([row_a[:prompt_len], row_b[:prompt_len]]).cuda()
    responses = torch.stack([row_a[prompt_len:], row_b[prompt_len:]]).cuda()
    prompt_mask = torch.ones_like(prompt_ids)
    resp_mask = torch.ones_like(responses)
    # Ragged second row: right-padded short response, to exercise row selection.
    resp_mask[1, short_len:] = 0
    responses[1, short_len:] = tok.pad_token_id or 0

    seg = max(1, min(segment, resp_len))
    new = spans_new(resp_len, window, seg)
    old = spans_legacy(resp_len, window, seg)
    print(f"resp_len={resp_len} prompt_len={prompt_len} W={window} S={segment} onset={window + segment}")
    print(f"row 0 length={resp_len}, row 1 length={short_len}")
    print(f"merged spans : {new}")
    print(f"legacy spans : {old}")

    lp_new, passes_new, tokens_new = teacher_logprobs(
        model, prompt_ids, prompt_mask, responses, resp_mask, new
    )
    lp_old, passes_old, tokens_old = teacher_logprobs(
        model, prompt_ids, prompt_mask, responses, resp_mask, old
    )
    # The untruncated teacher, as one pass over the whole prefix. Ground truth for
    # every position shallower than the onset.
    lp_plain, _, _ = teacher_logprobs(
        model, prompt_ids, prompt_mask, responses, resp_mask, [(0, resp_len, 0)]
    )

    onset = window + seg
    valid = resp_mask.bool()
    depth = torch.arange(resp_len, device=responses.device)[None, :].expand_as(valid)
    shallow = valid & (depth < onset)
    deep = valid & (depth >= onset)

    print()
    print(f"teacher logp, mean |value| over valid tokens : {lp_old[valid].abs().mean().item():.6f}")
    print(f"valid tokens: {int(valid.sum())} total, {int(shallow.sum())} shallow, {int(deep.sum())} deep")

    print()
    print(f"depth >= onset ({onset}), identical input shapes, must be bit-exact:")
    if bool(deep.any()):
        deep_diff = (lp_new - lp_old).abs()[deep].max().item()
        print(f"  max abs diff merged vs legacy : {deep_diff:.3e}")
        assert deep_diff == 0.0, f"truncated region is not bit-exact: {deep_diff}"
        trunc_effect = (lp_new - lp_plain).abs()[deep].mean().item()
        print(f"  mean abs (merged - plain)     : {trunc_effect:.3e}   <- the intervention itself")
    else:
        print("  no tokens beyond the onset in this batch")
        deep_diff = 0.0

    print()
    print(f"depth < onset, untruncated in both schemes, checked against the plain pass:")
    d_new = (lp_new - lp_plain).abs()[shallow]
    d_old = (lp_old - lp_plain).abs()[shallow]
    d_pair = (lp_new - lp_old).abs()[shallow]
    shallow_new, shallow_old = d_new.max().item(), d_old.max().item()
    print(f"  merged vs plain   max {shallow_new:.3e}  mean {d_new.mean().item():.3e}")
    print(f"  legacy vs plain   max {shallow_old:.3e}  mean {d_old.mean().item():.3e}"
          f"  nonzero {(d_old > 0).float().mean().item():.1%}")
    print(f"  merged vs legacy  max {d_pair.max().item():.3e}  mean {d_pair.mean().item():.3e}")
    # The merge must not push these positions further from the untruncated teacher than
    # the scheme it replaces. Both are reading the same mathematical quantity out of
    # differently shaped tensors, so neither is expected to be exactly zero.
    assert shallow_new <= max(shallow_old, 1e-6) * 1.5 + 1e-9, (
        f"merged read-out drifted further from the plain pass than legacy: "
        f"{shallow_new:.3e} vs {shallow_old:.3e}"
    )

    print()
    print(f"cost  passes merged={passes_new} legacy={passes_old}")
    print(f"      tokens merged={tokens_new} legacy={tokens_old}  saved={1 - tokens_new / tokens_old:.1%}")

    # Padded positions inside a span carry don't-care values in both schemes, so they
    # are excluded above. Report them rather than silently ignoring.
    if (~valid).any():
        print(f"      max abs diff over padded positions (don't care): "
              f"{(lp_new - lp_old).abs()[~valid].max().item():.3e}")

    print()
    print("PASS: the merge is bit-exact where it can be, and no further from the")
    print("      untruncated teacher than the legacy scheme where it cannot be.")


def worker_method_source(name):
    """Source of one reward-worker method, read from the shipped file."""
    text = FSDP_WORKERS.read_text()
    lines = text.splitlines()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"{name} not found in {FSDP_WORKERS}")


def pass_counts(spans, per_rank_lengths, micro_bsz, equalize):
    """Forward passes each rank issues per span, under the static micro-batch split.

    A rank forwards the rows whose response reaches the span's start, split into
    micro batches of ``micro_bsz``. Without equalizing, a rank with no such row
    forwards nothing at all for that span.
    """
    counts = []
    for local in per_rank_lengths:
        row_counts = [sum(1 for length in local if length > start) for start, _, _ in spans]
        if equalize:
            row_counts = [max(rows, 1) for rows in row_counts]
        counts.append([-(-rows // micro_bsz) for rows in row_counts])
    if equalize:
        agreed = [max(rank[i] for rank in counts) for i in range(len(spans))]
        counts = [list(agreed) for _ in counts]
    return counts


def check_rank_invariance(spans_fn, resp_len, window, segment, micro_bsz=4, n_ranks=8, rows_per_rank=128):
    """The teacher forward count must not depend on the local batch.

    The span list is rank-invariant: it is fixed by the padded response length and the
    two knobs. The rows are not. Past the onset, the rows are the local sequences that
    reach that depth, and for a model whose responses are mostly short that count is a
    small rank-dependent number, often zero. Every teacher forward is an FSDP all-gather
    over the whole group, so ranks that disagree on how many to issue leave the group
    waiting on a collective that never comes, and the job hangs with the GPUs busy.

    Nothing about this is visible in the span arithmetic or in a single-process forward,
    which is why the other checks here pass while a real 8-GPU run deadlocks.
    """
    import random

    print("=" * 78)
    print("Part 2: rank invariance of the teacher forward count")
    print("=" * 78)

    src = worker_method_source("_teacher_forward_windowed")
    forward_at = src.find("_teacher_run_micro_batches")
    skip_at = src.find("if not live:")
    checks = [
        ("pass count is agreed across ranks", "dist.all_reduce" in src and "ReduceOp.MAX" in src),
        ("rearrange's own dp sync is off", "same_micro_num_in_dp=False" in src),
        ("a span with no local row still forwards", "rows.new_zeros(1)" in src),
        ("the read-out, not the forward, is skipped", 0 <= forward_at < skip_at),
    ]
    for label, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}    {label}")
    assert all(ok for _, ok in checks), "the windowed teacher can desynchronize its ranks"

    # Qwen3-4B non-thinking: mostly short answers with a thin long tail. This is the
    # regime that breaks, and the opposite of the 1.5b runs, where nearly every
    # sequence ran past the onset and every rank happened to agree by luck.
    rng = random.Random(0)
    lengths = [
        [rng.randint(200, 3000) if rng.random() > 0.03 else rng.randint(3000, resp_len) for _ in range(rows_per_rank)]
        for _ in range(n_ranks)
    ]
    spans = spans_fn(resp_len, window, segment)
    onset = window + segment
    deep = sum(1 for local in lengths for length in local if length > onset)
    print(f"\n  {n_ranks} ranks x {rows_per_rank} rows, onset {onset}, {deep} rows past it")
    print(f"  spans at depth {[s for s, _, _ in spans]}, micro batch {micro_bsz} rows\n")

    for label, equalize in (("legacy", False), ("fixed", True)):
        counts = pass_counts(spans, lengths, micro_bsz, equalize)
        agree = len({tuple(c) for c in counts}) == 1
        print(f"  {label:6s}  per-rank passes per span:")
        for rank, c in enumerate(counts):
            print(f"            rank {rank}: {c}")
        print(f"          {'all ranks agree' if agree else 'RANKS DISAGREE -> deadlock'}\n")
        if equalize:
            assert agree, "forward count still depends on the local batch"
        else:
            assert not agree, "the length sample no longer exercises the bug; this check has no teeth"

    print("PASS: every rank issues the same teacher forwards regardless of how many")
    print("      of its sequences reach each depth.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-model", action="store_true", help="skip the GPU part")
    ap.add_argument("--resp-len", type=int, default=16384)
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--segment", type=int, default=4096)
    ap.add_argument("--short-len", type=int, default=5000)
    # The qwen-4b runs train the teacher forward in bf16 (TEACHER_MODEL_DTYPE).
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "bf16"])
    args = ap.parse_args()

    check_plumbing()
    print()
    spans_new = load_production_spans()
    check_spans(spans_new, args.prompt_len, args.resp_len)
    print()
    check_rank_invariance(spans_new, args.resp_len, args.window, args.segment)
    if not args.no_model:
        check_model(
            spans_new,
            args.model,
            args.resp_len,
            args.prompt_len,
            args.window,
            args.segment,
            args.short_len,
            args.dtype,
        )


if __name__ == "__main__":
    main()
