# 1. What this code does (big picture)

# This Python script generates a C header file containing a random input tensor.

# Workflow:

# generate_input_file()

# Calls generate_input_image() to create random input data.

# Inserts that data into a C template file.

# Writes the final .h file.

# generate_input_image()

# Generates random numbers between -1 and 1.

# Formats them into a C array string like:

import numpy as np
from string import Template

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
