# telnet localhost 3456
# ./util/term/gem5term localhost 3456
# mount -t 9p -o trans=virtio,version=9p2000.L,aname=/home/jerry/gem5/shared_folder gem5 /mnt
# mount -t 9p -o trans=virtio,version=9p2000.L,aname=/home/thu/gem5/shared_folder gem5 /mnt
# mount -t 9p -o trans=virtio,version=9p2000.L,aname=/home/thu/TiC-SAT gem5 /mnt
# Gem5 should be re-complied after VirtIO90.py is modified

# export M5_PATH=/home/thu/gem5_resources

# Checkpoint
# /sbin/m5 checkpoint

# source launch_simulation.sh
# bash launch_simulation_session.sh # use screen to run the tasks in the background

# m5 exit
# /sbin/m5 exit


set -euo pipefail

# Timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
run_dir="output/run_${TIMESTAMP}"
stats_filename="stats_${TIMESTAMP}.txt"
config_filename="config_${TIMESTAMP}.json"
log_dir="logs"
session_name="gem5_run_${TIMESTAMP}"

mkdir -p "${run_dir}" "${log_dir}"

echo "Starting gem5 in screen session: ${session_name}"
echo "Run dir: ${run_dir}"



# New run with new kernel
# ./build/ARM/gem5.fast \
#     -d "${run_dir}" \
#     --stats-file="${stats_filename}" \
#     --dump-config="${config_filename}"\
#     configs/example/arm/starter_fs.py \
#     --kernel=$HOME/kernel-build/linux/vmlinux \
#     --disk-image=../gem5_resources/arm64-ubuntu-20220727.img \
#     --script=$(pwd)/scripts/drop_to_shell.rcS \
#     --vio-9p=/home/thu/TiC-SAT \
#     --cpu=atomic



# Screen new kernel (SVE Length= 128 bits)
screen -dmS "${session_name}" bash -lc "
nice -n 0 ./build/ARM/gem5.fast \
    -d '${run_dir}' \
    --stats-file='${stats_filename}' \
    --dump-config='${config_filename}' \
    configs/example/arm/starter_fs.py \
    --kernel=$HOME/kernel-build/linux/vmlinux \
    --disk-image=../gem5_resources/arm64-ubuntu-20220727.img \
    --interactive-terminal \
    --vio-9p=/home/thu/TiC-SAT \
    --restore=/home/thu/gem5/output/run_20260423_115522/cpt.4042546229250 \
    --cpu=minor \
    > '${log_dir}/gem5_${TIMESTAMP}.log' 2>&1
"

echo "Started."
echo "Screen session: ${session_name}"
echo "Log file: ${log_dir}/gem5_${TIMESTAMP}.log"
echo "To inspect log: tail -f ${log_dir}/gem5_${TIMESTAMP}.log"

# Screen new kernel (SVE Length= 256 bits)
# screen -dmS "${session_name}" bash -lc "
# nice -n 0 ./build/ARM/gem5.fast \
#     -d '${run_dir}' \
#     --stats-file='${stats_filename}' \
#     --dump-config='${config_filename}' \
#     configs/example/arm/starter_fs.py \
#     --kernel=$HOME/kernel-build/linux/vmlinux \
#     --disk-image=../gem5_resources/arm64-ubuntu-20220727.img \
#     --interactive-terminal \
#     --vio-9p=/home/thu/TiC-SAT \
#     --restore=/home/thu/gem5/output/run_20260503_130206/cpt.21111005842000 \
#     --cpu=minor \
#     > '${log_dir}/gem5_${TIMESTAMP}.log' 2>&1
# "

# echo "Started."
# echo "Screen session: ${session_name}"
# echo "Log file: ${log_dir}/gem5_${TIMESTAMP}.log"
# echo "To inspect log: tail -f ${log_dir}/gem5_${TIMESTAMP}.log"


# Screen new kernel (SVE Length= 512 bits)
# screen -dmS "${session_name}" bash -lc "
# nice -n 0 ./build/ARM/gem5.fast \
#     -d '${run_dir}' \
#     --stats-file='${stats_filename}' \
#     --dump-config='${config_filename}' \
#     configs/example/arm/starter_fs.py \
#     --kernel=$HOME/kernel-build/linux/vmlinux \
#     --disk-image=../gem5_resources/arm64-ubuntu-20220727.img \
#     --interactive-terminal \
#     --vio-9p=/home/thu/TiC-SAT \
#     --restore=/home/thu/gem5/output/run_20260503_132054/cpt.23999404757250 \
#     --cpu=minor \
#     > '${log_dir}/gem5_${TIMESTAMP}.log' 2>&1
# "

# echo "Started."
# echo "Screen session: ${session_name}"
# echo "Log file: ${log_dir}/gem5_${TIMESTAMP}.log"
# echo "To inspect log: tail -f ${log_dir}/gem5_${TIMESTAMP}.log"

