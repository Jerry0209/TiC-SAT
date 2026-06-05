"""Transformer debug and C-output comparison utilities.

This module is used by the Transformer/GEMM generator notebooks and debug
flows to build Python-side reference tensors for generated TiC-SAT transformer
kernels. It produces and consumes two related kinds of debug data:

1. Python reference outputs for one transformer block, either with integer
   arithmetic that mirrors the generated C kernels or with the normal FP32
   PyTorch path. The reference output dictionary includes intermediate tensors
   such as Q/K/V projections, attention head outputs, feed-forward outputs, and
   the final add-and-normalize result.
2. Plain text matrix files used to compare Python reference tensors against
   tensors dumped by generated C/C++ code. The comparison helpers reload those
   files, check shape and value differences, and print compact mismatch
   summaries for each learner.

The integer helpers intentionally reproduce C-style behavior rather than ideal
PyTorch behavior: values are wrapped to signed int8, softmax uses a small lookup
table, normalization uses integer truncation, and intermediate outputs are kept
with names that match the C debug dumps. This makes the module useful for
finding where a generated transformer kernel first diverges from the Python
reference implementation.
"""

import math
from pathlib import Path

import torch


# Lookup table used by the integer softmax approximation in generated C code.
# Scores are first interpreted as unsigned bytes, shifted right by three bits,
# and then used as indexes into this table.
SOFTMAX_LOOKUP = torch.tensor(
    [
        4, 5, 7, 8, 11, 14, 18, 23,
        30, 38, 49, 63, 80, 103, 132, 170,
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 2, 2, 3,
    ],
    dtype=torch.int32,
)


def to_int32_tensor(values):
    """Return ``values`` as a detached CPU ``torch.int32`` tensor.

    Most C-style helpers operate on int32 intermediates even when their inputs
    or outputs are int8. This helper also detaches tensors from autograd so the
    debug path behaves like a pure value computation.
    """

    return torch.as_tensor(values).detach().cpu().to(torch.int32)


def wrap_to_int8(values):
    """Wrap integer values into signed int8 range using C-like overflow rules.

    PyTorch casts can clamp or reinterpret depending on context. The generated
    kernels rely on two's-complement byte wrapping, so this maps every value to
    the range ``[-128, 127]`` with modulo 256 arithmetic before converting to
    ``torch.int8``.
    """

    arr = to_int32_tensor(values)
    wrapped = torch.remainder(arr + 128, 256) - 128
    return wrapped.to(torch.int8)


def linear_to_tensors(linear):
    """Extract a ``torch.nn.Linear`` layer's weight and optional bias as int32.

    The returned weight keeps PyTorch's linear layout of
    ``[out_features, in_features]``. Callers transpose it when performing the
    matrix multiply.
    """

    weight = linear.weight.detach().cpu().to(torch.int32)
    bias = None
    if linear.bias is not None:
        bias = linear.bias.detach().cpu().to(torch.int32)
    return weight, bias


def c_style_dense(input_int8, linear):
    """Run a dense layer with int32 accumulation and int8 wrapped output.

    This mirrors the generated integer C dense path:
    1. Convert the input, weight, and bias to CPU int32 tensors.
    2. Multiply ``input`` by the transposed linear weight matrix.
    3. Add bias if the layer has one.
    4. Wrap the accumulated result back to signed int8.
    """

    weight, bias = linear_to_tensors(linear)
    accum = torch.matmul(to_int32_tensor(input_int8), weight.transpose(0, 1))
    if bias is not None:
        accum = accum + bias
    return wrap_to_int8(accum)


def c_style_softmax(scores_int8):
    """Approximate softmax using the same lookup-and-scale path as C code.

    The generated integer kernels do not compute a floating-point softmax.
    Instead, each int8 score is treated as an unsigned byte, coarsened to a
    5-bit table index, converted through ``SOFTMAX_LOOKUP``, and normalized by
    an integer denominator. The result is then wrapped back to int8.
    """

    # Reinterpret signed int8 values as unsigned byte values in the range
    # [0, 255], matching how the generated C path indexes the lookup table.
    scores_u8 = torch.bitwise_and(to_int32_tensor(scores_int8), 0xFF).to(torch.int64)
    out = torch.zeros_like(scores_u8, dtype=torch.int32)

    for row_idx in range(scores_u8.shape[0]):
        # Step 1: Reduce each unsigned score to a table index by shifting out
        # the lowest three bits, then gather approximate exponent values.
        row = SOFTMAX_LOOKUP.index_select(0, torch.bitwise_right_shift(scores_u8[row_idx], 3))
        row_sum = int(row.sum().item())
        if row_sum == 0:
            row_sum = 1
        # Step 2: Convert the row sum into the fixed-point denominator used by
        # the kernel. The zero guards avoid division by zero for degenerate rows.
        denom = row_sum >> 8
        if denom == 0:
            denom = 1
        # Step 3: Integer-divide every approximate exponent by the denominator.
        out[row_idx] = torch.div(row, denom, rounding_mode="floor")

    return wrap_to_int8(out)


def c_style_post_softmax(head_out_int8):
    """Apply the generated C post-attention downscale to a head output."""

    return wrap_to_int8(torch.bitwise_right_shift(to_int32_tensor(head_out_int8), 6))


def c_style_addnorm(residual_int8, candidate_int8):
    """Run the integer residual add and layer-normalization approximation.

    For each row, this function first adds the residual and candidate tensors
    with int8 wrapping. It then computes an integer mean, integer variance,
    fixed-point reciprocal standard deviation, and finally normalizes the row
    with an arithmetic right shift. The formula follows the generated C kernel
    closely so mismatches are easier to attribute to a specific stage.
    """

    residual = to_int32_tensor(residual_int8)
    candidate = to_int32_tensor(candidate_int8)
    result = torch.zeros_like(candidate, dtype=torch.int8)

    for row_idx in range(candidate.shape[0]):
        # Step 1: Add residual and candidate in the same wrapped int8 domain as
        # the generated kernel, then promote to int32 for reductions.
        row = wrap_to_int8(candidate[row_idx] + residual[row_idx]).to(torch.int32)
        row_sum = int(row.sum().item())
        mean = math.trunc(row_sum / row.shape[0])

        # Step 2: Compute variance with integer truncation, then approximate the
        # inverse standard deviation as an 8-bit fixed-point scale factor.
        diffs = row - mean
        variance = math.trunc(int(torch.sum(diffs * diffs).item()) / row.shape[0])
        sd = math.sqrt(float(variance))
        sd_inv = int((1 << 8) / (sd + 1.0))

        # Step 3: Apply the fixed-point scale and wrap the normalized result.
        normalized = torch.bitwise_right_shift(diffs * sd_inv, 8)
        result[row_idx] = wrap_to_int8(normalized)

    return result


def transformer_block_forward_c_style(network_one_learner, x, num_head, d_q):
    """Run one transformer block with C-style int8 debug arithmetic.

    ``network_one_learner`` is expected to contain layers named ``q_h0``,
    ``k_h0``, ``v_h0``, etc. for each head, plus ``condense``, ``ff0``, and
    ``ff1``. The returned dictionary stores intermediate tensors under names
    that match the text files generated by the C debug path.

    ``d_q`` is accepted for API symmetry with the FP32 reference path, but the
    integer approximation does not use the usual ``1 / sqrt(d_q)`` attention
    scale.
    """

    del d_q

    x_int8 = wrap_to_int8(x)
    residual0 = x_int8.clone()

    outputs = {}
    head_outputs = []

    for h in range(num_head):
        # Per-head Q/K/V projections are stored separately so a comparison can
        # identify whether divergence starts before or after attention scores.
        q = c_style_dense(x_int8, network_one_learner[f"q_h{h}"])
        k = c_style_dense(x_int8, network_one_learner[f"k_h{h}"])
        v = c_style_dense(x_int8, network_one_learner[f"v_h{h}"])

        outputs[f"q_h{h}"] = q
        outputs[f"k_h{h}"] = k
        outputs[f"v_h{h}"] = v

        # Attention path:
        # 1. Compute QK^T in int32 and wrap scores to int8.
        # 2. Apply the lookup-table softmax approximation.
        # 3. Multiply attention weights by V and wrap the head output.
        scores = wrap_to_int8(torch.matmul(q.to(torch.int32), k.to(torch.int32).transpose(0, 1)))
        attn = c_style_softmax(scores)
        head_out = wrap_to_int8(torch.matmul(attn.to(torch.int32), v.to(torch.int32)))

        outputs[f"head_out_h{h}"] = head_out
        head_outputs.append(c_style_post_softmax(head_out))

    # Combine heads, project back to the model dimension, apply the first
    # residual addnorm, run the two feed-forward layers, then apply final addnorm.
    multihead_out = torch.cat(head_outputs, dim=1).to(torch.int8)
    condense_out = c_style_dense(multihead_out, network_one_learner["condense"])
    after_attn_addnorm = c_style_addnorm(residual0, condense_out)
    ff0_out = c_style_dense(after_attn_addnorm, network_one_learner["ff0"])
    ff1_out = c_style_dense(ff0_out, network_one_learner["ff1"])
    final_out = c_style_addnorm(after_attn_addnorm, ff1_out)

    outputs["multihead_out"] = multihead_out
    outputs["condense_out"] = condense_out
    outputs["after_attn_addnorm"] = after_attn_addnorm
    outputs["ff0_out"] = ff0_out
    outputs["ff1_out"] = ff1_out
    outputs["final_out"] = final_out

    return outputs


def fp32_addnorm(residual_fp32, candidate_fp32, eps=1.0e-5):
    """Run standard FP32 residual add followed by row-wise layer normalization."""

    row = candidate_fp32 + residual_fp32
    mean = row.mean(dim=1, keepdim=True)
    variance = torch.mean((row - mean) * (row - mean), dim=1, keepdim=True)
    return (row - mean) * torch.rsqrt(variance + eps)


def transformer_block_forward_fp32(network_one_learner, x, num_head, d_q):
    """Run one transformer block with normal FP32 PyTorch operations.

    This is the high-level reference path used to compare against floating-point
    generated code or to understand the intended mathematical behavior before
    integer approximations are applied. It records a superset of attention
    outputs using names that line up with existing debug dump conventions.
    """

    x_fp32 = torch.as_tensor(x).detach().cpu().to(torch.float32)
    residual0 = x_fp32.clone()

    outputs = {}
    head_outputs = []
    scale = 1.0 / math.sqrt(float(d_q))

    for h in range(num_head):
        # Standard scaled dot-product attention for one head.
        q = network_one_learner[f"q_h{h}"](x_fp32)
        k = network_one_learner[f"k_h{h}"](x_fp32)
        v = network_one_learner[f"v_h{h}"](x_fp32)

        outputs[f"q_h{h}"] = q
        outputs[f"k_h{h}"] = k
        outputs[f"v_h{h}"] = v

        scores = torch.matmul(q, k.transpose(0, 1))
        attn = torch.softmax(scores * scale, dim=1)
        head_out = torch.matmul(attn, v)

        outputs[f"softmax_qk_h{h}"] = attn
        outputs[f"softmax_v_pre_post_h{h}"] = head_out
        outputs[f"head_out_h{h}"] = head_out
        outputs[f"head_out_post_h{h}"] = head_out
        head_outputs.append(head_out)

    multihead_out = torch.cat(head_outputs, dim=1).to(torch.float32)
    condense_out = network_one_learner["condense"](multihead_out)
    after_attn_addnorm = fp32_addnorm(residual0, condense_out)
    ff0_out = network_one_learner["ff0"](after_attn_addnorm)
    ff1_out = network_one_learner["ff1"](ff0_out)
    final_out = fp32_addnorm(after_attn_addnorm, ff1_out)

    outputs["multihead_out"] = multihead_out
    outputs["condense_out"] = condense_out
    outputs["after_attn_addnorm"] = after_attn_addnorm
    outputs["ff0_out"] = ff0_out
    outputs["ff1_out"] = ff1_out
    outputs["final_out"] = final_out

    return outputs


def save_int_matrix_text(path, matrix):
    """Write an integer matrix as whitespace-separated rows in a text file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = to_int32_tensor(matrix)

    with path.open("w", encoding="utf-8") as fout:
        for row in tensor.tolist():
            fout.write(" ".join(str(int(value)) for value in row))
            fout.write("\n")


def load_int_matrix_text(path):
    """Load an integer matrix text file, returning ``None`` if it is missing.

    Empty files or files containing only blank lines also return ``None`` so
    callers can report them as unavailable debug outputs.
    """

    path = Path(path)
    if not path.exists():
        return None

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([int(token) for token in line.split()])

    if not rows:
        return None

    return torch.tensor(rows, dtype=torch.int32)


def export_transformer_outputs(outputs, learner_dir):
    """Save every integer debug tensor in ``outputs`` under ``learner_dir``."""

    learner_dir = Path(learner_dir)
    learner_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in outputs.items():
        save_int_matrix_text(learner_dir / f"{name}.txt", matrix)


def save_float_matrix_text(path, matrix):
    """Write a floating-point matrix as whitespace-separated text rows."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.as_tensor(matrix).detach().cpu().to(torch.float32)

    with path.open("w", encoding="utf-8") as fout:
        for row in tensor.tolist():
            fout.write(" ".join(f"{float(value):.9g}" for value in row))
            fout.write("\n")


def load_float_matrix_text(path):
    """Load a floating-point matrix text file, returning ``None`` if absent."""

    path = Path(path)
    if not path.exists():
        return None

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(token) for token in line.split()])

    if not rows:
        return None

    return torch.tensor(rows, dtype=torch.float32)


def export_transformer_float_outputs(outputs, learner_dir):
    """Save every FP32 debug tensor in ``outputs`` under ``learner_dir``."""

    learner_dir = Path(learner_dir)
    learner_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in outputs.items():
        save_float_matrix_text(learner_dir / f"{name}.txt", matrix)


def summarize_diff(reference, candidate):
    """Summarize exact integer differences between two tensors or matrices.

    The result dictionary is designed for compact debug reporting. If shapes
    differ, only shape information is returned. If shapes match, the summary
    includes maximum absolute difference, mismatch count, total value count, and
    the first mismatching element when one exists.
    """

    reference_arr = to_int32_tensor(reference)
    candidate_arr = to_int32_tensor(candidate)

    if tuple(reference_arr.shape) != tuple(candidate_arr.shape):
        return {
            "shape_match": False,
            "reference_shape": tuple(reference_arr.shape),
            "candidate_shape": tuple(candidate_arr.shape),
        }

    diff = reference_arr - candidate_arr
    abs_diff = torch.abs(diff)
    mismatch_locs = torch.nonzero(abs_diff != 0, as_tuple=False)

    summary = {
        "shape_match": True,
        "reference_shape": tuple(reference_arr.shape),
        "candidate_shape": tuple(candidate_arr.shape),
        "max_abs_diff": int(abs_diff.max().item()) if abs_diff.numel() else 0,
        "mismatch_count": int(mismatch_locs.shape[0]),
        "total_values": int(reference_arr.numel()),
    }

    if mismatch_locs.numel() != 0:
        first = mismatch_locs[0]
        idx = tuple(int(v) for v in first.tolist())
        summary["first_mismatch_index"] = idx
        summary["reference_value"] = int(reference_arr[idx].item())
        summary["candidate_value"] = int(candidate_arr[idx].item())

    return summary


def compare_saved_outputs_with_reference(reference_outputs, c_learner_dir):
    """Compare integer reference tensors against C-dumped text files.

    ``reference_outputs`` is usually the dictionary returned by
    ``transformer_block_forward_c_style``. For each named tensor, this function
    looks for ``<name>.txt`` in ``c_learner_dir`` and either marks it missing or
    stores the exact integer diff summary.
    """

    c_learner_dir = Path(c_learner_dir)
    comparisons = {}

    for name, reference in reference_outputs.items():
        candidate = load_int_matrix_text(c_learner_dir / f"{name}.txt")
        if candidate is None:
            comparisons[name] = {"missing": True}
            continue
        comparisons[name] = summarize_diff(reference, candidate)

    return comparisons


def summarize_float_diff(reference, candidate, atol=1.0e-3, rtol=1.0e-4):
    """Summarize floating-point differences using absolute/relative tolerance.

    A value is counted as mismatching when
    ``abs(reference - candidate) > atol + rtol * abs(reference)``. The returned
    fields mirror ``summarize_diff`` but keep floating-point values and include
    the tolerances used for the comparison.
    """

    reference_arr = torch.as_tensor(reference).detach().cpu().to(torch.float32)
    candidate_arr = torch.as_tensor(candidate).detach().cpu().to(torch.float32)

    if tuple(reference_arr.shape) != tuple(candidate_arr.shape):
        return {
            "shape_match": False,
            "reference_shape": tuple(reference_arr.shape),
            "candidate_shape": tuple(candidate_arr.shape),
        }

    diff = reference_arr - candidate_arr
    abs_diff = torch.abs(diff)
    tolerance = atol + rtol * torch.abs(reference_arr)
    mismatch_locs = torch.nonzero(abs_diff > tolerance, as_tuple=False)

    summary = {
        "shape_match": True,
        "reference_shape": tuple(reference_arr.shape),
        "candidate_shape": tuple(candidate_arr.shape),
        "max_abs_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mismatch_count": int(mismatch_locs.shape[0]),
        "total_values": int(reference_arr.numel()),
        "atol": float(atol),
        "rtol": float(rtol),
    }

    if mismatch_locs.numel() != 0:
        first = mismatch_locs[0]
        idx = tuple(int(v) for v in first.tolist())
        summary["first_mismatch_index"] = idx
        summary["reference_value"] = float(reference_arr[idx].item())
        summary["candidate_value"] = float(candidate_arr[idx].item())

    return summary


def compare_saved_float_outputs_with_reference(reference_outputs, c_learner_dir, atol=1.0e-3, rtol=1.0e-4):
    """Compare FP32 reference tensors against C-dumped text files."""

    c_learner_dir = Path(c_learner_dir)
    comparisons = {}

    for name, reference in reference_outputs.items():
        candidate = load_float_matrix_text(c_learner_dir / f"{name}.txt")
        if candidate is None:
            comparisons[name] = {"missing": True}
            continue
        comparisons[name] = summarize_float_diff(reference, candidate, atol=atol, rtol=rtol)

    return comparisons


def print_comparison_summary(learner_idx, comparisons):
    """Print a human-readable summary for integer C-vs-Python comparisons."""

    print(f"\n===== C vs Python comparison | learner {learner_idx} =====")
    for name, summary in comparisons.items():
        if summary.get("missing"):
            print(f"{name}: missing C output")
            continue
        if not summary.get("shape_match", False):
            print(
                f"{name}: shape mismatch "
                f"{summary['reference_shape']} vs {summary['candidate_shape']}"
            )
            continue
        print(
            f"{name}: max_abs_diff={summary['max_abs_diff']}, "
            f"mismatches={summary['mismatch_count']}/{summary['total_values']}"
        )
        if summary["mismatch_count"] != 0:
            print(
                f"  first mismatch at {summary['first_mismatch_index']}: "
                f"python={summary['reference_value']} c={summary['candidate_value']}"
            )


def print_float_comparison_summary(learner_idx, comparisons):
    """Print a human-readable summary for FP32 C-vs-Python comparisons."""

    print(f"\n===== C FP32 vs Python FP32 comparison | learner {learner_idx} =====")
    for name, summary in comparisons.items():
        if summary.get("missing"):
            print(f"{name}: missing C output")
            continue
        if not summary.get("shape_match", False):
            print(
                f"{name}: shape mismatch "
                f"{summary['reference_shape']} vs {summary['candidate_shape']}"
            )
            continue
        print(
            f"{name}: max_abs_diff={summary['max_abs_diff']:.6g}, "
            f"mismatches={summary['mismatch_count']}/{summary['total_values']} "
            f"(atol={summary['atol']}, rtol={summary['rtol']})"
        )
        if summary["mismatch_count"] != 0:
            print(
                f"  first mismatch at {summary['first_mismatch_index']}: "
                f"python={summary['reference_value']:.9g} c={summary['candidate_value']:.9g}"
            )


def c_output_dir_has_any_outputs(c_outputs_root, n_learners):
    """Return whether any learner directory contains C debug output text files."""

    root = Path(c_outputs_root)
    if not root.exists():
        return False
    for learner in range(n_learners):
        learner_dir = root / f"learner{learner}"
        if learner_dir.exists() and any(learner_dir.glob("*.txt")):
            return True
    return False
