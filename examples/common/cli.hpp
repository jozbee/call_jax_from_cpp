/**
 * @file cli.hpp
 * @brief The flag parser the examples share.
 *
 * Deliberately tiny: the examples exist to show a call path, and a real
 * argument-parsing library in the middle of that would be the largest thing on
 * the page.  It understands two forms, `--name value` and a bare `--flag`, and
 * it refuses anything it was not told about -- a mistyped `--iterationss`
 * silently falling back to a default is exactly how a benchmark ends up
 * measuring a configuration nobody chose.
 *
 * Everything happens in the constructor, which is startup code.  The accessors
 * are linear scans over a handful of strings; resolve them once into locals
 * rather than calling them from a loop.
 *
 * @code
 *   cjfc::Cli cli(argc, argv,
 *                 {"artifact", "iterations", "cpu", "rt"},
 *                 "usage: realtime [--artifact P] [--iterations N] "
 *                 "[--cpu auto|none|N] [--rt]");
 *   if (cli.help()) return 0;
 *   const std::size_t iterations = cli.get_size("iterations", 10000);
 * @endcode
 */
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// call_jax_from_cpp: helpers the examples share; not the library
namespace cjfc {

/**
 * @brief Parsed command line: known flags, their values, and `--help`.
 *
 * Flag names may be written with or without leading dashes, in the known-flag
 * list and at every accessor, so `"cpu"` and `"--cpu"` name the same flag.
 * Messages always spell them back with dashes, the way the user typed them.
 */
class Cli {
 public:
  /**
   * @brief Parse @p argv against @p known.
   *
   * A token is taken as the previous flag's value unless it begins with `--`,
   * so `--cpu -1` works and `--rt --mlock` is two bare flags.  `--name=value`
   * is accepted as well, which is the unambiguous spelling when a value could
   * be mistaken for a flag.
   *
   * `--help` (or `-h`) prints @p usage to stdout and sets `help()`; the rest of
   * the command line is then not validated, because a user asking for help has
   * probably just typed something wrong.  The caller is expected to return 0.
   *
   * @param argc  As handed to `main`.
 * @param argv  As handed to `main`.  `argv[0]` becomes `program()`, which is
 *              what error messages spell back.
 * @param known Flag names this program accepts, with or without dashes.
   * @param usage One-paragraph usage text, printed on `--help`.
   * @throws std::runtime_error on an unknown flag or a stray argument, naming
   *         it.
   */
  Cli(int argc, char** argv, std::initializer_list<const char*> known,
      std::string usage)
      : usage_(std::move(usage)) {
    known_.reserve(known.size());
    for (const char* name : known) {
      known_.emplace_back(normalize(name));
    }
    if (argc > 0 && argv[0] != nullptr) {
      program_ = argv[0];
    }

    for (int i = 1; i < argc; ++i) {
      const std::string_view arg(argv[i]);
      if (arg == "--help" || arg == "-h") {
        help_ = true;
        continue;
      }
      if (arg.size() < 2 || arg[0] != '-') {
        throw std::runtime_error("unexpected argument '" + std::string(arg) +
                                 "': every value must follow its flag");
      }

      const std::string_view body = normalize(arg);
      const std::size_t eq = body.find('=');
      Entry entry;
      entry.name = std::string(body.substr(0, eq));
      if (eq != std::string_view::npos) {
        entry.value = std::string(body.substr(eq + 1));
        entry.has_value = true;
      } else if (i + 1 < argc &&
                 std::string_view(argv[i + 1]).substr(0, 2) != "--") {
        entry.value = argv[++i];
        entry.has_value = true;
      }

      if (!help_ && !is_known(entry.name)) {
        throw std::runtime_error("unknown flag '--" + entry.name + "'");
      }
      entries_.push_back(std::move(entry));
    }

    if (help_) {
      print_usage();
    }
  }

  /// @brief Whether `--help` was asked for, in which case the usage has already
  ///        been printed and the program should exit 0.
  bool help() const noexcept { return help_; }

  /// @brief `argv[0]`, for a program that wants to name itself in its own
  ///        output.
  const std::string& program() const noexcept { return program_; }

  /// @brief The usage text this parser was built with.
  const std::string& usage() const noexcept { return usage_; }

  /// @brief Print the usage text to stdout, with a closing newline.
  void print_usage() const {
    std::fputs(usage_.c_str(), stdout);
    if (usage_.empty() || usage_.back() != '\n') {
      std::fputc('\n', stdout);
    }
  }

  /// @brief Whether @p name appeared at all, with or without a value.  This is
  ///        how a bare `--rt` is read.
  bool flag(std::string_view name) const noexcept {
    return find(normalize(name)) != nullptr;
  }

  /**
   * @brief The value given for @p name, or @p fallback when it was absent.
   *
   * Returns by value, which is one allocation per call -- irrelevant at
   * startup, which is the only place this belongs.
   *
   * @throws std::runtime_error when the flag was given without a value.
   */
  std::string get(std::string_view name, std::string_view fallback = {}) const {
    const std::string_view key = normalize(name);
    const Entry* entry = find(key);
    if (entry == nullptr) {
      return std::string(fallback);
    }
    if (!entry->has_value) {
      throw std::runtime_error("missing value for '--" + std::string(key) +
                               "'");
    }
    return entry->value;
  }

  /// @brief `get()` parsed as a signed integer.
  /// @throws std::runtime_error when the text is not an integer, naming the
  ///         flag.
  long get_long(std::string_view name, long fallback) const {
    const std::string_view key = normalize(name);
    const Entry* entry = find(key);
    if (entry == nullptr) {
      return fallback;
    }
    const std::string& text = value_of(key, *entry);
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
      throw bad_number(key, text, "an integer");
    }
    return parsed;
  }

  /// @brief `get()` parsed as a floating-point number.
  /// @throws std::runtime_error when the text is not a number, naming the flag.
  double get_double(std::string_view name, double fallback) const {
    const std::string_view key = normalize(name);
    const Entry* entry = find(key);
    if (entry == nullptr) {
      return fallback;
    }
    const std::string& text = value_of(key, *entry);
    errno = 0;
    char* end = nullptr;
    const double parsed = std::strtod(text.c_str(), &end);
    if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
      throw bad_number(key, text, "a number");
    }
    return parsed;
  }

  /**
   * @brief `get()` parsed as a count.
   *
   * A leading `-` is rejected rather than wrapped around, because
   * `--iterations -1` becoming eighteen quintillion iterations is a long wait
   * for a diagnosis.
   */
  std::size_t get_size(std::string_view name, std::size_t fallback) const {
    const std::string_view key = normalize(name);
    const Entry* entry = find(key);
    if (entry == nullptr) {
      return fallback;
    }
    const std::string& text = value_of(key, *entry);
    if (!text.empty() && text.front() == '-') {
      throw bad_number(key, text, "a non-negative count");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || errno == ERANGE) {
      throw bad_number(key, text, "a non-negative count");
    }
    return static_cast<std::size_t>(parsed);
  }

 private:
  /// One occurrence of a flag on the command line, in the order it appeared.
  struct Entry {
    std::string name;
    std::string value;
    bool has_value = false;
  };

  /// Strip the leading dashes, so the same flag can be written `--cpu` in the
  /// known list and `cpu` at the accessor, or the other way round.
  static std::string_view normalize(std::string_view name) noexcept {
    while (!name.empty() && name.front() == '-') {
      name.remove_prefix(1);
    }
    return name;
  }

  bool is_known(std::string_view name) const noexcept {
    for (const std::string& candidate : known_) {
      if (candidate == name) {
        return true;
      }
    }
    return false;
  }

  /// The last occurrence wins, so a wrapper script can prepend defaults and let
  /// the caller override them.
  const Entry* find(std::string_view name) const noexcept {
    const Entry* found = nullptr;
    for (const Entry& entry : entries_) {
      if (entry.name == name) {
        found = &entry;
      }
    }
    return found;
  }

  static const std::string& value_of(std::string_view name,
                                     const Entry& entry) {
    if (!entry.has_value) {
      throw std::runtime_error("missing value for '--" + std::string(name) +
                               "'");
    }
    return entry.value;
  }

  static std::runtime_error bad_number(std::string_view name,
                                       const std::string& text,
                                       const char* expected) {
    return std::runtime_error("'--" + std::string(name) + "' expects " +
                              expected + ", got '" + text + "'");
  }

  std::string program_;
  std::string usage_;
  std::vector<std::string> known_;
  std::vector<Entry> entries_;
  bool help_ = false;
};

}  // namespace cjfc
