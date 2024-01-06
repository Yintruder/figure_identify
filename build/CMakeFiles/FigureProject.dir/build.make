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
CMAKE_SOURCE_DIR = /home/yintruder/MonoProject/figure_identify

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/yintruder/MonoProject/figure_identify/build

# Include any dependencies generated for this target.
include CMakeFiles/FigureProject.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/FigureProject.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/FigureProject.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/FigureProject.dir/flags.make

CMakeFiles/FigureProject.dir/figure.cpp.o: CMakeFiles/FigureProject.dir/flags.make
CMakeFiles/FigureProject.dir/figure.cpp.o: ../figure.cpp
CMakeFiles/FigureProject.dir/figure.cpp.o: CMakeFiles/FigureProject.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/yintruder/MonoProject/figure_identify/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/FigureProject.dir/figure.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/FigureProject.dir/figure.cpp.o -MF CMakeFiles/FigureProject.dir/figure.cpp.o.d -o CMakeFiles/FigureProject.dir/figure.cpp.o -c /home/yintruder/MonoProject/figure_identify/figure.cpp

CMakeFiles/FigureProject.dir/figure.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/FigureProject.dir/figure.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/yintruder/MonoProject/figure_identify/figure.cpp > CMakeFiles/FigureProject.dir/figure.cpp.i

CMakeFiles/FigureProject.dir/figure.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/FigureProject.dir/figure.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/yintruder/MonoProject/figure_identify/figure.cpp -o CMakeFiles/FigureProject.dir/figure.cpp.s

# Object files for target FigureProject
FigureProject_OBJECTS = \
"CMakeFiles/FigureProject.dir/figure.cpp.o"

# External object files for target FigureProject
FigureProject_EXTERNAL_OBJECTS =

FigureProject: CMakeFiles/FigureProject.dir/figure.cpp.o
FigureProject: CMakeFiles/FigureProject.dir/build.make
FigureProject: /usr/local/lib/libopencv_dnn.so.3.4.16
FigureProject: /usr/local/lib/libopencv_highgui.so.3.4.16
FigureProject: /usr/local/lib/libopencv_ml.so.3.4.16
FigureProject: /usr/local/lib/libopencv_objdetect.so.3.4.16
FigureProject: /usr/local/lib/libopencv_shape.so.3.4.16
FigureProject: /usr/local/lib/libopencv_stitching.so.3.4.16
FigureProject: /usr/local/lib/libopencv_superres.so.3.4.16
FigureProject: /usr/local/lib/libopencv_videostab.so.3.4.16
FigureProject: /usr/local/lib/libopencv_viz.so.3.4.16
FigureProject: /usr/local/lib/libopencv_calib3d.so.3.4.16
FigureProject: /usr/local/lib/libopencv_features2d.so.3.4.16
FigureProject: /usr/local/lib/libopencv_flann.so.3.4.16
FigureProject: /usr/local/lib/libopencv_photo.so.3.4.16
FigureProject: /usr/local/lib/libopencv_video.so.3.4.16
FigureProject: /usr/local/lib/libopencv_videoio.so.3.4.16
FigureProject: /usr/local/lib/libopencv_imgcodecs.so.3.4.16
FigureProject: /usr/local/lib/libopencv_imgproc.so.3.4.16
FigureProject: /usr/local/lib/libopencv_core.so.3.4.16
FigureProject: CMakeFiles/FigureProject.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/yintruder/MonoProject/figure_identify/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable FigureProject"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/FigureProject.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/FigureProject.dir/build: FigureProject
.PHONY : CMakeFiles/FigureProject.dir/build

CMakeFiles/FigureProject.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/FigureProject.dir/cmake_clean.cmake
.PHONY : CMakeFiles/FigureProject.dir/clean

CMakeFiles/FigureProject.dir/depend:
	cd /home/yintruder/MonoProject/figure_identify/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/yintruder/MonoProject/figure_identify /home/yintruder/MonoProject/figure_identify /home/yintruder/MonoProject/figure_identify/build /home/yintruder/MonoProject/figure_identify/build /home/yintruder/MonoProject/figure_identify/build/CMakeFiles/FigureProject.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/FigureProject.dir/depend

