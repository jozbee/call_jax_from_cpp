/**
 * @file fixture.hpp
 * @brief Load reference input/output cases for correctness checking.
 *
 * A case is a flat little-endian float64 blob: inputs in call order followed
 * by outputs in call order, described by a `{name}_cases.json` manifest.
 * Written by `tools/npz_to_bin.py`.
 */
#pragma once

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/nlohmann/json.hpp"

namespace bench {

/// One reference case: the inputs to feed, and the outputs to expect.
struct Case {
  std::vector<std::vector<double>> inputs;
  std::vector<std::vector<double>> outputs;
};

/// A fixture: shared shapes plus one or more reference cases.
class Fixture {
 public:
  Fixture(const std::string& dir, const std::string& name) {
    const std::string manifest_path = dir + "/" + name + "_cases.json";
    std::ifstream manifest(manifest_path);
    if (!manifest) {
      throw std::runtime_error("cannot open " + manifest_path);
    }
    nlohmann::json meta;
    manifest >> meta;

    input_sizes_ = meta["input_sizes"].get<std::vector<std::size_t>>();
    output_sizes_ = meta["output_sizes"].get<std::vector<std::size_t>>();
    integral_output_ = meta.value("integral_output", -1);

    for (const auto& case_file : meta["cases"]) {
      cases_.push_back(read_case(dir + "/" +
                                 case_file.get<std::string>()));
    }
    if (cases_.empty()) {
      throw std::runtime_error("no cases in " + manifest_path);
    }
  }

  const std::vector<std::size_t>& input_sizes() const { return input_sizes_; }
  const std::vector<std::size_t>& output_sizes() const { return output_sizes_; }
  const std::vector<Case>& cases() const { return cases_; }
  int integral_output() const { return integral_output_; }

  /// Largest relative error of `actual` against case `c`'s reference outputs.
  double max_rel_error(std::size_t c,
                       const std::vector<const double*>& actual) const {
    double worst = 0.0;
    const Case& ref = cases_[c];
    for (std::size_t i = 0; i < ref.outputs.size(); ++i) {
      for (std::size_t j = 0; j < ref.outputs[i].size(); ++j) {
        const double want = ref.outputs[i][j];
        const double got = actual[i][j];
        const double denom = std::abs(want) + 1e-12;
        worst = std::max(worst, std::abs(got - want) / denom);
      }
    }
    return worst;
  }

  /// True when the designated integral output is exactly integral.
  bool integral_output_ok(const std::vector<const double*>& actual) const {
    if (integral_output_ < 0) {
      return true;
    }
    const double v = actual[static_cast<std::size_t>(integral_output_)][0];
    return v == std::floor(v);
  }

 private:
  Case read_case(const std::string& path) const {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
      throw std::runtime_error("cannot open " + path);
    }
    Case c;
    auto read_group = [&](const std::vector<std::size_t>& sizes,
                          std::vector<std::vector<double>>& dst) {
      for (std::size_t n : sizes) {
        // scalars are stored with size 0 but occupy one double
        const std::size_t count = n == 0 ? 1 : n;
        std::vector<double> v(count);
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(count * sizeof(double)));
        if (!f) {
          throw std::runtime_error("short read in " + path);
        }
        dst.push_back(std::move(v));
      }
    };
    read_group(input_sizes_, c.inputs);
    read_group(output_sizes_, c.outputs);
    return c;
  }

  std::vector<std::size_t> input_sizes_;
  std::vector<std::size_t> output_sizes_;
  std::vector<Case> cases_;
  int integral_output_ = -1;
};

}  // namespace bench
