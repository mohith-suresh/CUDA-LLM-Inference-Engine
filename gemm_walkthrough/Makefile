# Build the GEMM walkthrough.
#   make            # build ./gemm and ./autotune
#   make ARCH=sm_75 # target a specific GPU (default sm_75 = Turing / T4 / 1650 Ti)
#   make run        # build and run the ladder
#   make autotune   # build and run the autotuner (step 9)
#   make clean

NVCC ?= nvcc
ARCH ?= sm_75
FLAGS = -arch=$(ARCH) -O3 -lineinfo -lcublas

all: gemm autotune profile

gemm: main.cu *.cuh
	$(NVCC) $(FLAGS) main.cu -o gemm

autotune: 09_autotune.cu 05_blocktiling_2d.cuh common.cuh
	$(NVCC) $(FLAGS) 09_autotune.cu -o autotune

profile: profile.cu *.cuh
	$(NVCC) $(FLAGS) profile.cu -o profile

run: gemm
	./gemm

autotune-run: autotune
	./autotune

clean:
	rm -f gemm autotune profile

.PHONY: all run autotune-run clean
