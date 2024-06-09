# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/e/jinjie/CUDA/codes/CUDABasis

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/e/jinjie/CUDA/codes/CUDABasis/build

# Include any dependencies generated for this target.
include performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/compiler_depend.make

# Include the progress variables for this target.
include performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/progress.make

# Include the compile flags for this target's objects.
include performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/flags.make

performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o: performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o.depend
performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o: performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o.cmake
performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o: ../performance_analysis_optimization/vec_add_timing_gpu.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/e/jinjie/CUDA/codes/CUDABasis/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o"
	cd /mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir && /usr/bin/cmake -E make_directory /mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir//.
	cd /mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir//./vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o -D generated_cubin_file:STRING=/mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir//./vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o.cubin.txt -P /mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir//vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o.cmake

# Object files for target vec_add_timing_gpu
vec_add_timing_gpu_OBJECTS =

# External object files for target vec_add_timing_gpu
vec_add_timing_gpu_EXTERNAL_OBJECTS = \
"/mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o"

performance_analysis_optimization/vec_add_timing_gpu: performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o
performance_analysis_optimization/vec_add_timing_gpu: performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/build.make
performance_analysis_optimization/vec_add_timing_gpu: /usr/local/cuda-11.6/lib64/libcudart_static.a
performance_analysis_optimization/vec_add_timing_gpu: /usr/lib/x86_64-linux-gnu/librt.a
performance_analysis_optimization/vec_add_timing_gpu: /usr/local/cuda-11.6/lib64/libcudart_static.a
performance_analysis_optimization/vec_add_timing_gpu: /usr/lib/x86_64-linux-gnu/librt.a
performance_analysis_optimization/vec_add_timing_gpu: performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/e/jinjie/CUDA/codes/CUDABasis/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable vec_add_timing_gpu"
	cd /mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vec_add_timing_gpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/build: performance_analysis_optimization/vec_add_timing_gpu
.PHONY : performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/build

performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/clean:
	cd /mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization && $(CMAKE_COMMAND) -P CMakeFiles/vec_add_timing_gpu.dir/cmake_clean.cmake
.PHONY : performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/clean

performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/depend: performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/vec_add_timing_gpu_generated_vec_add_timing_gpu.cu.o
	cd /mnt/e/jinjie/CUDA/codes/CUDABasis/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/e/jinjie/CUDA/codes/CUDABasis /mnt/e/jinjie/CUDA/codes/CUDABasis/performance_analysis_optimization /mnt/e/jinjie/CUDA/codes/CUDABasis/build /mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization /mnt/e/jinjie/CUDA/codes/CUDABasis/build/performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : performance_analysis_optimization/CMakeFiles/vec_add_timing_gpu.dir/depend

