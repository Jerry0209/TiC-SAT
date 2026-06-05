"""
Input tensor and TiC-SAT packed-input generator.

This module is used by the Transformer/GEMM generator notebooks to create test
inputs for generated TiC-SAT kernels. It produces two related kinds of input
artifacts:

1. A C/C++ header containing random floating-point input tensors or GEMM input
   matrices. These headers are consumed by generated benchmark or validation
   code and expose data through static arrays such as input_flat or
   input_matrix.

2. A plain text TiC-SAT packed-input file for integer GEMM inputs. This export
   converts each int8 row into packed 32-bit words, then reorders those words
   into the blockwise layout expected by the TiC-SAT kernels.

The generated C/C++ code uses these files as deterministic input data for
kernel execution and comparison. Floating-point headers are convenient for
ordinary GEMM/NN tests, while the packed int8 export matches the memory layout
used by TiC-SAT hardware-oriented kernels.
"""

import os
from string import Template

import numpy as np

np.random.seed(42) # Ensure reproducibility of random values across runs


def generate_input_file(filename, in_channels, in_height, in_width, use_f16):
    """
    Create random input data, inject it into a header template, and save to `filename`.

    Returns:
        input_values (list): flat list of generated random values (Python floats).
    """
    # Build the C array string + keep raw numeric values
    image_string, input_values = generate_input_image(
        in_channels, in_height, in_width, use_f16
    )

    # Load template file and replace placeholders
    with open("./templates/input_header_template.tpl") as f:
        cont = f.read()
        tpl = Template(cont)

        header = tpl.substitute(
            input_channels=in_channels,
            input_height=in_height,
            input_width=in_width,
            input_string_values=image_string
        )

    # Write the final generated header content to disk
    with open(filename, "w") as f:
        f.write(header)

    return input_values


def generate_input_image(in_channels, in_height, in_width, use_f16):
    """
    Generate a flat random input tensor and format it as a C array declaration string.

    Array size is: IN_D * IN_H * IN_W
    Values are random in range [-1, 1].
    """
    # Start building C declaration string.
    # If use_f16=True, type becomes float16_t, otherwise float.
    input_string = "static float{} input_flat[IN_D * IN_H * IN_W] = {{\n"
    if not use_f16:
        input_string = input_string.format("")       # -> "static float input_flat[...]"
    else:
        input_string = input_string.format("16_t")   # -> "static float16_t input_flat[...]"

    # Generate random values in flat layout: channel-major flattened list
    input_values = []
    for i in range(in_channels * in_height * in_width):
        new_val = np.random.uniform(-1, 1)
        input_values.append(new_val)

    # Format values into rows by width, with blank line after each height block
    cnt_cols = 0
    cnt_rows = 0
    for v in input_values:
        input_string += "\t{}, ".format(v)

        cnt_cols += 1
        if cnt_cols == in_width:
            input_string += "\n"
            cnt_cols = 0
            cnt_rows += 1

        # After in_height rows, add an extra newline (visually separates channels)
        if cnt_rows == in_height:
            input_string += "\n"
            cnt_rows = 0

    # Close C array declaration
    input_string += "};"

    return input_string, input_values


def generate_gemm_input_file(filename, M, K, use_f16=False):
    X = np.random.uniform(-1.0, 1.0, size=(M, K))
    X = X.astype(np.float16 if use_f16 else np.float32)

    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write(f"#define GEMM_M {M}\n")
        f.write(f"#define GEMM_K {K}\n\n")

        c_type = "float16_t" if use_f16 else "float"
        f.write(f"static const {c_type} input_matrix[GEMM_M * GEMM_K] = {{\n")
        for row in X:
            f.write("    ")
            for v in row:
                suffix = "f" if not use_f16 else ""
                f.write(f"{float(v):.8f}{suffix}, ")
            f.write("\n")
        f.write("};\n")

    return X


def _pack_int8_row_to_words(row):
    """
    Pack one row of int8 values into little-endian 32-bit words.

    Purpose:
        TiC-SAT integer input files store four signed 8-bit values inside one
        32-bit integer word. This is more compact than writing one integer per
        element and matches the word-oriented layout that the generated kernels
        read from memory.

    Input:
        row: One already-clipped row of int8-compatible values.

    Output:
        words: A list of packed integers. Each output word contains four row
        entries, with the first row value in the lowest byte, the second value
        in the next byte, and so on.

    Example byte layout for values [a, b, c, d]:
        word bits  7..0   = a
        word bits 15..8   = b
        word bits 23..16  = c
        word bits 31..24  = d

    Notes:
        The row length must be divisible by four because each packed output word
        is exactly four bytes wide.
    """
    # A 32-bit packed word holds exactly four int8 values, so partial groups
    # would create an ambiguous final word. Reject those rows before packing.
    if len(row) % 4 != 0:
        raise ValueError("Each TiC-SAT input row must contain a multiple of 4 values.")

    words = []
    # Walk through the row in four-value chunks. Each chunk becomes one packed
    # 32-bit word in the output list.
    for idx in range(0, len(row), 4):
        word = 0
        # Pack the four signed int8 values into bytes.
        #
        # Step by step:
        # 1. np.int8(value) forces the value into signed 8-bit range/encoding.
        # 2. int(...) converts it back to a Python integer for bit operations.
        # 3. & 0xFF keeps only the low byte, preserving two's-complement bits
        #    for negative values.
        # 4. << (8 * byte_idx) places that byte into byte position 0, 1, 2, or 3.
        # 5. |= merges the byte into the accumulating 32-bit word.
        for byte_idx, value in enumerate(row[idx:idx + 4]):
            word |= (int(np.int8(value)) & 0xFF) << (8 * byte_idx)
        words.append(word)
    return words


def _rowwise_words_to_blockwise(rowwise_words, n_row, n_col_words, max_col):
    """
    Reorder packed row-major input words into TiC-SAT blockwise layout.

    Purpose:
        _pack_int8_row_to_words first creates a simple row-major stream:
        all packed words from row 0, then all packed words from row 1, and so on.
        TiC-SAT kernels consume packed inputs in column blocks instead. This
        helper performs that layout conversion without changing the packed word
        contents themselves.

    Arguments:
        rowwise_words: Flat list of packed words in row-major order.
        n_row: Number of rows in the original matrix, usually seq_len or GEMM_M.
        n_col_words: Number of packed words per row. Since each word stores four
            int8 values, this is input_size / 4.
        max_col: Number of packed words in one TiC-SAT kernel column block.
            This is kernel_dim / 4.

    Output:
        blockwise_words: Flat list ordered by column block first, then row, then
        packed word inside that block.

    Layout transformation:
        For every column block:
            visit row 0, row 1, ..., row n_row - 1
            copy max_col packed words from that row/block

        This creates the order expected by kernels that process a fixed-width
        input tile across all rows before moving to the next tile.
    """
    # The input width must split cleanly into fixed-size TiC-SAT blocks. If it
    # does not, the kernel would not know how to interpret the final partial
    # block.
    if n_col_words % max_col != 0:
        raise ValueError(
            f"Packed input columns={n_col_words} must be divisible by max_col={max_col}."
        )

    blockwise_words = []
    # Outer loop: select one packed-column block at a time.
    for col in range(n_col_words // max_col):
        base = col * max_col
        # Middle loop: for the selected column block, visit every matrix row.
        for row in range(n_row):
            row_start = row * n_col_words
            # Inner loop: copy every packed word inside the current block.
            for i in range(max_col):
                blockwise_words.append(rowwise_words[row_start + base + i])
    return blockwise_words


def export_ticsat_input_file(filename, X, seq_len, input_size, kernel_dim=16):
    """
    Export an integer input matrix in the packed layout consumed by TiC-SAT.

    Purpose:
        Generated TiC-SAT kernels expect int8 input data to be stored as packed
        32-bit words and ordered by kernel-sized blocks. This function converts
        a normal Python/NumPy matrix into that exact file format.

    Arguments:
        filename: Destination path for the plain text packed input file.
        X: Source matrix or flat array containing integer-like values.
        seq_len: Number of input rows.
        input_size: Number of int8 values per row.
        kernel_dim: Kernel input tile width in int8 elements. The default value
            of 16 means each block contains four packed 32-bit words.

    Returns:
        int_input: The clipped int8 matrix with shape (seq_len, input_size).
        blockwise_words: The packed and reordered words written to filename.

    File format:
        The output file is a single whitespace-separated list of decimal integer
        words. Each word represents four int8 values packed by
        _pack_int8_row_to_words.
    """
    # Both the kernel tile width and the input row width must be multiples of
    # four because packing groups four int8 values into one 32-bit word.
    if kernel_dim % 4 != 0:
        raise ValueError("kernel_dim must be divisible by 4.")
    if input_size % 4 != 0:
        raise ValueError("input_size must be divisible by 4 for TiC-SAT packed input.")

    # Convert the input to a 2-D integer matrix, clamp it to signed int8 range,
    # then store it as np.int8 so the later byte packing has a precise type.
    int_input = np.asarray(X, dtype=np.int16).reshape(seq_len, input_size)
    int_input = np.clip(int_input, -128, 127).astype(np.int8)

    # packed_cols counts 32-bit words per row. max_col counts 32-bit words per
    # TiC-SAT kernel block.
    packed_cols = input_size // 4
    max_col = kernel_dim // 4

    rowwise_words = []
    # First pack each row independently in ordinary row-major order.
    for row in int_input:
        rowwise_words.extend(_pack_int8_row_to_words(row))

    # Then reorder the packed row-major stream into the blockwise layout expected
    # by the generated TiC-SAT kernel.
    blockwise_words = _rowwise_words_to_blockwise(
        rowwise_words,
        seq_len,
        packed_cols,
        max_col,
    )

    # Create the destination directory and emit the compact text representation.
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as fout:
        fout.write(" ".join(str(int(word)) for word in blockwise_words))

    return int_input, blockwise_words


def generate_gemm_int_input_file(
    filename,
    M,
    K,
    use_f16=False,
    value_min=-2,
    value_max=2,
    ticsat_export_path=None,
    legacy_export_path=None,
    ticsat_kernel_dim=16,
):
    """
    Generate an integer GEMM input header and optional TiC-SAT packed exports.

    Purpose:
        This function is the integer-input counterpart to generate_gemm_input_file.
        It creates a random M x K integer matrix, writes it as a C/C++ static
        array for generated GEMM tests, and can additionally export the same
        values in TiC-SAT's packed int8 text format.

    Arguments:
        filename: Destination path for the generated C/C++ header.
        M: GEMM row count.
        K: GEMM column count.
        use_f16: If true, write the header matrix as float16_t values. Otherwise
            write it as float values.
        value_min: Inclusive lower bound for generated integer values.
        value_max: Inclusive upper bound for generated integer values.
        ticsat_export_path: Optional path for the TiC-SAT packed input file.
        legacy_export_path: Optional second packed export path kept for callers
            that still expect the older output location/name.
        ticsat_kernel_dim: Kernel tile width passed to export_ticsat_input_file.

    Returns:
        X_int: The generated integer matrix before conversion to header float
        type and before TiC-SAT int8 clipping.
    """
    # Generate small integer test values as int16 first. This keeps random
    # generation simple while still allowing export_ticsat_input_file to clip
    # explicitly into the final int8 range.
    X_int = np.random.randint(value_min, value_max + 1, size=(M, K), dtype=np.int16)
    # The C header stores numeric inputs in the floating-point type expected by
    # the generated GEMM code, even when the source values are integer-valued.
    X_header = X_int.astype(np.float16 if use_f16 else np.float32)

    # Write a self-contained header with matrix dimensions and a flat static
    # input_matrix array.
    with open(filename, "w") as f:
        f.write("#pragma once\n")
        f.write(f"#define GEMM_M {M}\n")
        f.write(f"#define GEMM_K {K}\n\n")

        c_type = "float16_t" if use_f16 else "float"
        f.write(f"static const {c_type} input_matrix[GEMM_M * GEMM_K] = {{\n")
        for row in X_header:
            f.write("    ")
            for v in row:
                suffix = "f" if not use_f16 else ""
                f.write(f"{float(v):.8f}{suffix}, ")
            f.write("\n")
        f.write("};\n")

    # Optional TiC-SAT export: write the same generated values in packed int8
    # blockwise format for kernels that read the external packed input file.
    if ticsat_export_path is not None:
        export_ticsat_input_file(
            ticsat_export_path,
            X_int,
            M,
            K,
            kernel_dim=ticsat_kernel_dim,
        )

    # Optional legacy export: keep producing a second packed file when older
    # scripts or notebooks still point at the legacy path.
    if legacy_export_path is not None:
        export_ticsat_input_file(
            legacy_export_path,
            X_int,
            M,
            K,
            kernel_dim=ticsat_kernel_dim,
        )

    return X_int
