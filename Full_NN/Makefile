
CC = gcc
ARMCC ?= $(or $(shell command -v aarch64-none-linux-gnu-gcc 2>/dev/null),$(shell command -v aarch64-linux-gnu-gcc 2>/dev/null))

ifeq ($(ARMCC),)
$(error No AArch64 cross-compiler found. Install aarch64-linux-gnu-gcc or set ARMCC=/path/to/compiler.)
endif

CFLAGS = -Wall

OPTIM = -O2


SRC_DIRS = . ./src ./../sve_prints ./print_functs

INC_DIRS 	= 	./ ./input_values ./inc ./../sve_prints/ ./print_functs/



# Optional file, compiled only when LENET is set
ifdef LENET
SRC_DIRS += ./LeNet
INC_DIRS += ./LeNet ./lenet_definitions
CFLAGS += -DLENET
endif


# Optional file, compiled only when ALEXNET is set
ifdef ALEXNET
SRC_DIRS += ./AlexNet
INC_DIRS += ./AlexNet ./alexnet_definitions
CFLAGS += -DALEXNET
endif


# Optional file, compiled only when VGG16 is set
ifdef VGG16
SRC_DIRS += ./VGG16
INC_DIRS += ./VGG16 ./VGG16_definitions
CFLAGS += -DVGG16
endif


# Optional file, compiled only when RESNET18 is set
ifdef RESNET18
SRC_DIRS += ./ResNet18
INC_DIRS += ./ResNet18 ./ResNet18_definitions
CFLAGS += -DRESNET18
endif


# Optional file, compiled only when TRANSFORMER is set
ifdef TRANSFORMER
SRC_DIRS += ./Transformer
INC_DIRS += ./Transformer ./gemm_definitions
CFLAGS += -DTRANSFORMER
endif


SRC = $(wildcard $(addsuffix /*.c, $(SRC_DIRS)))
SRC := $(filter-out ./dense_only_lenet.c ./test_gemm.c ./Transformer/Transformer_ffn_test.c, $(SRC))
INC 		= 	$(foreach d, $(INC_DIRS), -I$d)


OUT_NAME ?= net_app.out
TRANSFORMER_NOSVE_OUT ?= transformer_gemm_only.out


SVE_FALGS = sve128=on

SVE_SIZE ?= 128

ifeq ($(SVE_SIZE), 256)
	SVE_FALGS = sve256=on
endif

ifeq ($(SVE_SIZE), 512)
	SVE_FALGS = sve512=on
endif

ifeq ($(SVE_SIZE), 1024)
	SVE_FALGS = sve1024=on
endif

ifeq ($(SVE_SIZE), 2048)
	SVE_FALGS = sve2048=on
endif




all:
	$(CC) $(CFLAGS) $(OPTIM) $(SRC) $(INC) -o $(OUT_NAME)
	@echo Done compiling...

run: all
	@echo Running...
	@echo "\n"
	./$(OUT_NAME)




v8:
	echo "Output name: $(OUT_NAME)"
#	$(ARMCC) $(CFLAGS) -march=armv8-a+sve -g -fverbose-asm -lm $(OPTIM) $(SRC) $(INC) -o $(OUT_NAME)
	$(ARMCC) $(CFLAGS) -march=armv8-a+sve -lm $(OPTIM) $(SRC) $(INC) -o $(OUT_NAME)
	@echo Done compiling for ARMv8...

run-v8:
	@echo "\n\nSVE Vect Size = $(SVE_SIZE)\n"
	qemu-aarch64 -cpu max,$(SVE_FALGS) -L /usr/aarch64-linux-gnu/ ./$(OUT_NAME)

transformer-nosve:
	echo "Output name: $(TRANSFORMER_NOSVE_OUT)"
	$(ARMCC) $(CFLAGS) $(OPTIM) ./transformer_gemm_only.c ./src/gemm_exec.c ./Transformer/Transformer.c \
		-I./ -I./inc -I./Transformer -I./gemm_definitions -o $(TRANSFORMER_NOSVE_OUT)
	@echo Done compiling transformer GEMM without SVE...

run-transformer-nosve:
	qemu-aarch64 -L /usr/aarch64-linux-gnu/ ./$(TRANSFORMER_NOSVE_OUT)


exec-leNet:
	make v8 LENET=1 && make run-v8


exec-alexNet:
	make v8 ALEXNET=1 && make run-v8


exec-vgg16:
	make v8 VGG16=1 && make run-v8


exec-resnet18:
	make v8 RESNET18=1 && make run-v8

exec-transformer:
	make v8 TRANSFORMER=1 && make run-v8

exec-transformer-nosve:
	make transformer-nosve && make run-transformer-nosve
