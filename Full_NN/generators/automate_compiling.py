import papermill as pm
import subprocess


base_dir_path = "/home/albini/Documents/ESL/gem5/apps/NN_layers/Full_NN/experiments/FINAL/AlexNet_Chris/"

n_learners = 1

tile_L2_size = 6
tile_L1_size = 12

cb_sizes = [4, 8, 16]
sve_sizes = [4, 8, 16, 32, 64]


# base_name = "N_learn_{}/sve_{}/alex_ens{}_cb{}_sve{}_interl_f32_sameIdxs"
base_name = "N_learn_{}/sve_{}/alex_ens{}_cb{}_sve{}_lbl_SIMD"

for sve_len in sve_sizes:

    out_dir_path = base_dir_path
    print(out_dir_path)

    for cb_len in cb_sizes:

        pm.execute_notebook(
            "AlexNet_generator.ipynb",
            "AlexNet_generator.ipynb",
            parameters={
                "TILE_L2_SIZE"  :   tile_L2_size,
                "TILE_L1_SIZE"  :   tile_L1_size,
                "CODEBOOK_SIZE" :   cb_len,
                "SVE_LANES"     :   sve_len,
                "N_LEARNERS"    :   n_learners
            }
        ) 

        # exec_name = "sve_{}/alex_ens1_cb{}_sve{}_lbl".format(sve_len, cb_len, sve_len)
        exec_name = base_name.format(n_learners, sve_len, n_learners, cb_len, sve_len)


        subprocess.run(["make", "-C", "./../", "v8", "ALEXNET=1", "OUT_NAME={}".format(out_dir_path + exec_name)])
        print("\n=================================================================================================================\n")
