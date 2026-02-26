# aarch64-linux-gnu-g++ transformer.cpp transformer_layers/*.cc accelerator/smm_gem.cpp accelerator/systolic_m2m.cc   -o transformer.o -DSA_SIZE=16 -DDEVELOP -DCORE_NUM=1 -fopenmp -O2
	
	
aarch64-linux-gnu-g++ -static transformer.cpp transformer_layers/*.cc accelerator/smm_gem.cpp accelerator/systolic_m2m.cc -o transformer.o -DSA_SIZE=16 -DDEVELOP -DCORE_NUM=1 -fopenmp -O2
	
	
	
# cp ~/TiC-SAT-Jerry/transformer.o /home/jerry/gem5/shared_folder/