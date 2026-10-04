cmake_minimum_required(VERSION 3.16)
project(wheel_leg_simulation C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译成功后自动将可执行程序拷贝至 bin/ 目录

# 1. 头文件包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    C:/Users/lenovo/miniforge3/envs/wheel_leg/Lib/site-packages/mujoco/include
    C:/Users/lenovo/miniforge3/envs/wheel_leg/Lib/site-packages/mujoco/include/mujoco
    C:/msys64/ucrt64/include
)

# 2. 库查找目录
link_directories(
    C:/Users/lenovo/miniforge3/envs/wheel_leg/Lib/site-packages/mujoco
    C:/msys64/ucrt64/lib
)

# 3. 收集三层架构源文件
set(SOURCES
    src/interface.cpp
    src/controller.cpp
    src/runner.cpp
    src/main.cpp
)

# 4. 生成可执行文件
add_executable(wheel_leg_sim ${SOURCES})

# 5. 链接 MuJoCo 与图形渲染依赖库
target_link_libraries(wheel_leg_sim
    PRIVATE
    C:/Users/lenovo/miniforge3/envs/wheel_leg/Lib/site-packages/mujoco/mujoco.dll
    glfw3
    opengl32
    gdi32
)

# 自动部署到 bin/ 目录
add_custom_command(TARGET wheel_leg_sim POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:wheel_leg_sim>"
        "${CMAKE_CURRENT_LIST_DIR}/bin/wheel_leg_sim.exe"
)
