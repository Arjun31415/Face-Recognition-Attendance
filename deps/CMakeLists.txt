# Set the dependencies directory
set(DEPS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
# Get all the subdirectories present in the deps folder
file(GLOB_RECURSE DEPS_SUBDIRS ${DEPS_DIR}/*)

foreach(SUBDIR ${DEPS_SUBDIRS})
  if(IS_DIRECTORY ${SUBDIR})
    # Check if the subdirectory is dlib
    if("${SUBDIR}" STREQUAL "${DEPS_DIR}/dlib")
      continue()
    endif()
    add_subdirectory(${SUBDIR})
  endif()
endforeach()

add_subdirectory(${DEPS_DIR}/dlib)
