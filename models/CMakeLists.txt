set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(ModelsLibrary "${CMAKE_CURRENT_SOURCE_DIR}/models.hpp"
                         "${CMAKE_CURRENT_SOURCE_DIR}/models.cpp")

find_package(Boost REQUIRED)
target_link_libraries(ModelsLibrary LINK_PUBLIC ${Boost_LIBRARIES})
target_include_directories(ModelsLibrary PUBLIC "${Boost_INCLUDE_DIRS}")
target_include_directories(ModelsLibrary PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_definitions(ModelsLibrary PUBLIC MODELS_VERSION=0.1)
target_compile_features(ModelsLibrary PRIVATE cxx_std_17)

