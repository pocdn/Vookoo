cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(VookooExamplesSDL2)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/../external)
include_directories(${PROJECT_SOURCE_DIR}/../include)

add_definitions(-DSOURCE_DIR="${CMAKE_SOURCE_DIR}/")
add_definitions(-DBINARY_DIR="${PROJECT_BINARY_DIR}/")

find_package(Vulkan REQUIRED)
find_package(SDL2  REQUIRED)

find_program(GLSLANG glslangValidator)
find_program(GLSLC   glslc)

function(example_sdl2 order exname)
  set(shaders "")

  foreach(shader ${ARGN})
    if(GLSLC)
      add_custom_command(
        OUTPUT  ${shader}.spv
        COMMAND ${GLSLC} ${PROJECT_SOURCE_DIR}/${exname}/${shader} -o ${PROJECT_BINARY_DIR}/${shader}.spv
        MAIN_DEPENDENCY ${exname}/${shader}
      )
    elseif(GLSLANG)
      add_custom_command(
        OUTPUT  ${shader}.spv
        COMMAND ${GLSLANG} -V ${PROJECT_SOURCE_DIR}/${exname}/${shader} -o ${PROJECT_BINARY_DIR}/${shader}.spv
        MAIN_DEPENDENCY ${exname}/${shader}
      )
    else()
      message(WARNING "No GLSL compiler found — ${shader} will not be compiled")
    endif()
    list(APPEND shaders "${exname}/${shader}")
  endforeach()

  add_executable(${order}-${exname}
    ${exname}/${exname}.cpp
    ${shaders}
    ../include/vku/vku.hpp
    ../include/vku/vku_framework_sdl2.hpp
  )

  target_link_libraries(${order}-${exname}
    SDL2::SDL2
    Vulkan::Vulkan
  )
endfunction()

example_sdl2(01 helloSDL2 helloSDL2.vert helloSDL2.frag helloSDL2.comp)
example_sdl2(02 MSAA MSAA.vert MSAA.frag)
example_sdl2(03 noisedemo
  noisedemo.vert
  noisedemo.frag
  noisedemo_s2d.frag
  noisedemo_s4d.frag
  noisedemo_c2d.frag
  noisedemo_c3d.frag
  noisedemo_c4d.frag
)
example_sdl2(04 fluid3D
  fluid_advect.comp
  fluid_forces.comp
  fluid_divg.comp
  fluid_jacobi.comp
  fluid_grad.comp
  fluid_display.vert
  fluid_display.frag
)
