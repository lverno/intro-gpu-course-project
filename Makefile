main: main.cu
	nvcc $< --output-file $@ -std=c++20 --optimize 3 -Wno-deprecated-gpu-targets -lraylib
