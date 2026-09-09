/**
 * @file jax_arm_controller.cpp
 * @brief A ros2_control position controller whose step function is an
 *        ahead-of-time exported JAX program.
 *
 * The controller owns no numerics.  `examples/05_ros2_control/export.py`
 * writes `step(q, t, dt) -> q_cmd`, resolved-rate control of a two-link planar
 * arm, and everything here is the piping around it: three parameters, one
 * position interface per joint, a `pjrt::Function` loaded in `on_configure`,
 * and an `update()` that copies, calls and writes back.
 *
 * The hardening this file can apply is the process's half -- the allocator and
 * the resident set.  The update thread belongs to `controller_manager`, which
 * pins and prioritises it from its own parameters, so nothing here touches
 * affinity or scheduling class.
 */
#include <cstddef>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pjrt_exec/rt.hpp"
#include "pjrt_exec/runtime.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace jax_arm_controller {

namespace {

/// The process's one client.  `controller_manager` hosts every controller in
/// one process, and creating a second `pjrt::Runtime` mid-run starts a second
/// set of XLA thread pools, which is a latency spike.
pjrt::Runtime& runtime() {
  static pjrt::Runtime instance;  // default options: inline, one device,
  return instance;                // one worker thread
}

}  // namespace

class JaxArmController : public controller_interface::ControllerInterface {
 public:
  controller_interface::CallbackReturn on_init() override {
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<std::string>("artifact", "");
    auto_declare<std::vector<int64_t>>("xla_cpus", {});
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::InterfaceConfiguration command_interface_configuration()
      const override {
    return {controller_interface::interface_configuration_type::INDIVIDUAL,
            interface_names()};
  }

  controller_interface::InterfaceConfiguration state_interface_configuration()
      const override {
    return {controller_interface::interface_configuration_type::INDIVIDUAL,
            interface_names()};
  }

  controller_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State&) override {
    joints_ = get_node()->get_parameter("joints").as_string_array();
    const std::string artifact =
        get_node()->get_parameter("artifact").as_string();
    const std::vector<int64_t> xla_cpus =
        get_node()->get_parameter("xla_cpus").as_integer_array();
    if (joints_.empty() || artifact.empty()) {
      RCLCPP_ERROR(get_node()->get_logger(),
                   "both `joints` and `artifact` have to be set");
      return controller_interface::CallbackReturn::ERROR;
    }

    try {
      // docs: begin ros2-configure
      // First, so that the heap the plugin and the executable then grow is
      // already the hardened one.
      report("harden_malloc", pjrt::rt::harden_malloc());

      pjrt::FunctionOptions options;
      options.load_policy = pjrt::LoadPolicy::BinaryOnly;  // never compile here
      function_ =
          std::make_unique<pjrt::Function>(runtime(), artifact, options);
      if (function_->input_numel(0) != joints_.size()) {
        RCLCPP_ERROR(get_node()->get_logger(),
                     "%s takes %zu joints, but %zu were configured",
                     artifact.c_str(), function_->input_numel(0),
                     joints_.size());
        return controller_interface::CallbackReturn::ERROR;
      }
      q_ = function_->input<double>(0);  // resolved once; update() only reads
      t_ = function_->input<double>(1);  // and writes through them
      dt_ = function_->input<double>(2);
      q_cmd_ = function_->output<double>(0);

      // After the load, so what it prefaults is the memory a call will touch.
      report("lock_memory", pjrt::rt::lock_memory());
      if (!xla_cpus.empty()) {
        // XLA's pools do not exist until the Runtime does, so not earlier.
        report("corral_xla_threads",
               pjrt::rt::corral_xla_threads(
                   std::vector<int>(xla_cpus.begin(), xla_cpus.end())));
      }
      // docs: end ros2-configure
    } catch (const std::exception& error) {
      RCLCPP_ERROR(get_node()->get_logger(), "loading %s failed: %s",
                   artifact.c_str(), error.what());
      return controller_interface::CallbackReturn::ERROR;
    }

    RCLCPP_INFO(get_node()->get_logger(), "%s",
                function_->load_detail().c_str());
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State&) override {
    *t_ = 0.0;
    return controller_interface::CallbackReturn::SUCCESS;
  }

  controller_interface::return_type update(
      const rclcpp::Time&, const rclcpp::Duration& period) override {
    // docs: begin ros2-update
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      q_[i] = state_interfaces_[i].get_optional().value_or(q_[i]);
    }
    *dt_ = period.seconds();
    *t_ += *dt_;
    function_->call();
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      if (!command_interfaces_[i].set_value(q_cmd_[i])) {
        return controller_interface::return_type::ERROR;
      }
    }
    return controller_interface::return_type::OK;
    // docs: end ros2-update
  }

  controller_interface::CallbackReturn on_cleanup(
      const rclcpp_lifecycle::State&) override {
    function_.reset();
    return controller_interface::CallbackReturn::SUCCESS;
  }

 private:
  /// `<joint>/position` for each configured joint, in configuration order --
  /// which is the order `update()` indexes both interface vectors by.
  std::vector<std::string> interface_names() const {
    std::vector<std::string> names;
    names.reserve(joints_.size());
    for (const std::string& joint : joints_) {
      names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
    }
    return names;
  }

  /// One line per hardening step: what was asked for, and what came of it.
  void report(const char* name, const pjrt::rt::Status& status) {
    RCLCPP_INFO(get_node()->get_logger(), "[%s] %s: %s",
                status.ok ? "ok  " : "skip", name, status.detail.c_str());
  }

  std::vector<std::string> joints_;
  std::unique_ptr<pjrt::Function> function_;
  double* q_ = nullptr;
  double* t_ = nullptr;
  double* dt_ = nullptr;
  const double* q_cmd_ = nullptr;
};

}  // namespace jax_arm_controller

PLUGINLIB_EXPORT_CLASS(jax_arm_controller::JaxArmController,
                       controller_interface::ControllerInterface)
