# compiler and flags
CC = /home/thu/miniforge3/envs/gem5_env/bin/aarch64-conda-linux-gnu-gcc
CFLAGS = -O2 -Wall -march=armv8-a+sve

# target
TARGET = hello_sve

# source files
SRC = hello_sve.c

# qemu and sysroot
QEMU_AARCH64 = $(HOME)/opt/qemu-sve/bin/qemu-aarch64
SYSROOT = $(shell $(CC) -print-sysroot)

# SVE size control
SVE_FLAGS = sve=on,sve-default-vector-length=16

SVE_SIZE ?= 128

ifeq ($(SVE_SIZE), 256)
	SVE_FLAGS = sve=on,sve-default-vector-length=32
endif

ifeq ($(SVE_SIZE), 512)
	SVE_FLAGS = sve=on,sve-default-vector-length=64
endif

ifeq ($(SVE_SIZE), 1024)
	SVE_FLAGS = sve=on,sve-default-vector-length=128
endif

ifeq ($(SVE_SIZE), 2048)
	SVE_FLAGS = sve=on,sve-default-vector-length=256
endif

# build rules
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) -g

qemu-run: all
	$(QEMU_AARCH64) -cpu max,$(SVE_FLAGS) -L "$(SYSROOT)" ./$(TARGET)

# workaround for the current qemu-user behavior you observed
qemu-run-workaround: all
	$(QEMU_AARCH64) -cpu max,$(SVE_FLAGS) -L "$(SYSROOT)" ./$(TARGET) < ./$(TARGET)

clean:
	rm -f $(TARGET)


# make clean
# make

# make qemu-run
# make qemu-run-workaround

# make clean
# make qemu-run-workaround SVE_SIZE=256
# make qemu-run-workaround SVE_SIZE=512
# make qemu-run-workaround SVE_SIZE=1024