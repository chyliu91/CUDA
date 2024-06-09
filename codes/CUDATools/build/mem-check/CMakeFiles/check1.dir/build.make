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
CMAKE_SOURCE_DIR = /mnt/e/jinjie/CUDA/codes/CUDATools

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/e/jinjie/CUDA/codes/CUDATools/build

# Include any dependencies generated for this target.
include mem-check/CMakeFiles/check1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include mem-check/CMakeFiles/check1.dir/compiler_depend.make

# Include the progress variables for this target.
include mem-check/CMakeFiles/check1.dir/progress.make

# Include the compile flags for this target's objects.
include mem-check/CMakeFiles/check1.dir/flags.make

mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o: mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o.depend
mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o: mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o.cmake
mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o: ../mem-check/mem-check1.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/mnt/e/jinjie/CUDA/codes/CUDATools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building NVCC (Device) object mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o"
	cd /mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check/CMakeFiles/check1.dir && /usr/bin/cmake -E make_directory /mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check/CMakeFiles/check1.dir//.
	cd /mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check/CMakeFiles/check1.dir && /usr/bin/cmake -D verbose:BOOL=$(VERBOSE) -D build_configuration:STRING= -D generated_file:STRING=/mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check/CMakeFiles/check1.dir//./check1_generated_mem-check1.cu.o -D generated_cubin_file:STRING=/mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check/CMakeFiles/check1.dir//./check1_generated_mem-check1.cu.o.cubin.txt -P /mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check/CMakeFiles/check1.dir//check1_generated_mem-check1.cu.o.cmake

# Object files for target check1
check1_OBJECTS =

# External object files for target check1
check1_EXTERNAL_OBJECTS = \
"/mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o"

mem-check/check1: mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o
mem-check/check1: mem-check/CMakeFiles/check1.dir/build.make
mem-check/check1: /usr/local/cuda-11.6/lib64/libcudart_static.a
mem-check/check1: /usr/lib/x86_64-linux-gnu/librt.a
mem-check/check1: /usr/local/cuda-11.6/lib64/libcudart_static.a
mem-check/check1: /usr/lib/x86_64-linux-gnu/librt.a
mem-check/check1: mem-check/CMakeFiles/check1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/e/jinjie/CUDA/codes/CUDATools/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable check1"
	cd /mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/check1.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
mem-check/CMakeFiles/check1.dir/build: mem-check/check1
.PHONY : mem-check/CMakeFiles/check1.dir/build

mem-check/CMakeFiles/check1.dir/clean:
	cd /mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check && $(CMAKE_COMMAND) -P CMakeFiles/check1.dir/cmake_clean.cmake
.PHONY : mem-check/CMakeFiles/check1.dir/clean

mem-check/CMakeFiles/check1.dir/depend: mem-check/CMakeFiles/check1.dir/check1_generated_mem-check1.cu.o
	cd /mnt/e/jinjie/CUDA/codes/CUDATools/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/e/jinjie/CUDA/codes/CUDATools /mnt/e/jinjie/CUDA/codes/CUDATools/mem-check /mnt/e/jinjie/CUDA/codes/CUDATools/build /mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check /mnt/e/jinjie/CUDA/codes/CUDATools/build/mem-check/CMakeFiles/check1.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : mem-check/CMakeFiles/check1.dir/depend
