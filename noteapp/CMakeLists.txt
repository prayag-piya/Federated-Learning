cmake_minimum_required(VERSION 3.16)
project(NoteApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set wxWidgets root dir using WSL path with capital 'L'
set(wxWidgets_ROOT_DIR "/mnt/d/Libraries/wxWidgets-3.3.1")

# Find wxWidgets (core and base components)
find_package(wxWidgets REQUIRED COMPONENTS core base)
if(NOT wxWidgets_FOUND)
  message(FATAL_ERROR "Could not find wxWidgets!")
endif()
include(${wxWidgets_USE_FILE})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Specify curl include and library directories manually
set(CURL_INCLUDE_DIR "/mnt/d/Libraries/curl/include")
set(CURL_LIBRARY "/mnt/d/Libraries/curl/lib/libcurl.so")

# Create executable
add_executable(NoteApp main.cpp)

# Include directories
target_include_directories(NoteApp PRIVATE ${CURL_INCLUDE_DIR})

# Link wxWidgets libraries and curl
target_link_libraries(NoteApp PRIVATE ${wxWidgets_LIBRARIES} ${CURL_LIBRARY})
