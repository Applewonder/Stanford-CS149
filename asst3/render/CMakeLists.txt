cmake_minimum_required(VERSION 3.10)

# 项目名称，声明需要使用CXX和CUDA两种语言
project(Render LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# 选项与配置
# -----------------------------------------------------------------------------

# 如果你需要手动指定 CUDA 安装路径，或者在不同平台上有不同的路径需求，
# 可以添加一个选项来决定链接哪个目录：
option(USE_LOCAL_CUDA "Use local CUDA at /opt/cuda-8.0" OFF)

# 设置 C++ 编译选项 (对应 Makefile 中的 CXXFLAGS = -O3 -Wall -g)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -g")

# 设置 CUDA 编译选项 (对应 Makefile 中的 NVCCFLAGS = -O3 -m64)
# 这里使用 CMake 的 CUDA 语言特性
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -m64")

# -----------------------------------------------------------------------------
# 查找需要的依赖库 (OpenGL / GLUT)
# 如果你使用的是现代 CMake，可以使用 find_package(OpenGL) find_package(GLUT)
# -----------------------------------------------------------------------------

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# -----------------------------------------------------------------------------
# 源文件列表（把所有 .cpp 和 .cu 文件列出来）
# -----------------------------------------------------------------------------

set(SOURCES_CPP
    main.cpp
    display.cpp
    refRenderer.cpp
    noise.cpp
    ppm.cpp
    sceneLoader.cpp
)

set(SOURCES_CU
    cudaRenderer.cu
    benchmark.cu
)

# -----------------------------------------------------------------------------
# 生成可执行文件
# -----------------------------------------------------------------------------

# 将所有源文件打包生成名为 "render" 的可执行文件
add_executable(${PROJECT_NAME}
    ${SOURCES_CPP}
    ${SOURCES_CU}
)

# -----------------------------------------------------------------------------
# 链接目录与库 (对应 Makefile 中的 -L 以及 -l)
# -----------------------------------------------------------------------------

# 如果我们需要根据不同路径来链接 cudart，可以这样处理：
if(USE_LOCAL_CUDA)
    # 例如在 /opt/cuda-8.0 下
    target_link_directories(${PROJECT_NAME} PRIVATE /opt/cuda-8.0/lib64)
else()
    # 例如在 /usr/local/cuda 下
    target_link_directories(${PROJECT_NAME} PRIVATE /usr/local/cuda/lib64)
endif()

# 链接库：cudart / OpenGL::GL / GLUT::GLUT
#   - FindOpenGL 和 FindGLUT 成功后，CMake 通常会提供 OpenGL::GL 和 GLUT::GLUT 等 target
target_link_libraries(${PROJECT_NAME}
    PRIVATE
        cudart               # 来自 CUDA 的 runtime
        OpenGL::GL           # OpenGL
        GLUT::GLUT           # GLUT
)

# 若需要其他库 (例如 -lGLU 等)，可以继续添加
# target_link_libraries(${PROJECT_NAME} PRIVATE GLU)

# -----------------------------------------------------------------------------
# 额外的自定义目标 (对应原 Makefile 中的 clean、check 等)
# -----------------------------------------------------------------------------

# CMake 自带的 "clean" target，不需要特别再定义
# 如果你要自定义 'check'，可以用 add_custom_target：
add_custom_target(check
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/checker.pl
    COMMENT "Running checker script..."
    DEPENDS ${PROJECT_NAME}
)

# 同理，如果你想用一个 target 来清理日志：
add_custom_target(clean_logs
    COMMAND ${CMAKE_COMMAND} -E remove_directory logs
    COMMENT "Removing logs directory..."
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
