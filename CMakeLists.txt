cmake_minimum_required(VERSION 3.10)
project(StereoRectification)
set(CMAKE_C_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/cl.exe" CACHE FILEPATH "MSVC Compiler")
set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.43.34808/bin/Hostx64/x64/cl.exe" CACHE FILEPATH "MSVC Compiler")



# Trouver OpenCV
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
# Ajouter l'exécutable
add_executable(Stereo_rectification Stereo_rectification.cpp)

# Lier OpenCV avec l'exécutable
target_link_libraries(Stereo_rectification ${OpenCV_LIBS})

# Afficher les infos OpenCV trouvées
message(STATUS "OpenCV Include Dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV Libraries: ${OpenCV_LIBS}")
