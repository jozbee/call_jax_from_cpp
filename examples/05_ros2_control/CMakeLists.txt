# examples/05_ros2_control/CMakeLists.txt -- the ament_cmake package that
# docs/guides/integration.md's Recipe 1 describes, built with colcon rather
# than by the parent build: nothing here compiles without a ROS 2 workspace.

cmake_minimum_required(VERSION 3.21)
project(jax_arm_controller LANGUAGES CXX)

# docs: begin cmake-ros2
find_package(ament_cmake REQUIRED)
find_package(controller_interface REQUIRED)
find_package(hardware_interface REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)

# Vendored, not found: three translation units pinned to one JAX version.
# colcon builds offline, so nothing may be downloaded at configure time; point
# the build at the plugin `make plugin` already wrote instead.
set(PJRT_EXEC_FETCH_PLUGIN OFF CACHE BOOL "" FORCE)
set(PJRT_EXEC_PLUGIN_PATH
    "${CMAKE_CURRENT_SOURCE_DIR}/../../build/plugin/libpjrt_c_api_cpu_plugin.so"
    CACHE FILEPATH "" FORCE)
# In a package of your own this is `third_party/call_jax_from_cpp`, a
# submodule. EXCLUDE_FROM_ALL keeps its examples, tests and benchmark out of
# the colcon build.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../.." call_jax_from_cpp
                 EXCLUDE_FROM_ALL)

add_library(jax_arm_controller SHARED src/jax_arm_controller.cpp)
target_link_libraries(jax_arm_controller PUBLIC pjrt_exec::pjrt_exec)
ament_target_dependencies(jax_arm_controller PUBLIC
  controller_interface hardware_interface pluginlib rclcpp rclcpp_lifecycle)

# The line that makes `controller_manager` able to find the class by name.
pluginlib_export_plugin_description_file(controller_interface
                                         jax_arm_controller.xml)

install(TARGETS jax_arm_controller DESTINATION lib)
# docs: end cmake-ros2

install(DIRECTORY launch config urdf DESTINATION share/${PROJECT_NAME})
ament_package()
