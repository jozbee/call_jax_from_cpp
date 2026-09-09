/**
 * @file cli.hpp
 * @brief The flag parser the examples share.
 *
 * Deliberately tiny: the examples exist to show a call path, and an argument
 * library in the middle of it would be the largest thing on the page.  It
 * understands `--name value` and a bare `--flag`, and refuses anything it was
 * not told about -- a mistyped `--iterationss` falling back to a default is
 * how a benchmark ends up measuring a configuration nobody chose.
 *
 * Everything happens in the constructor.  The accessors are linear scans;
 * resolve them once into locals rather than calling them from a loop.
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
 * list and at every accessor.  Messages spell them back with dashes.
 */
class Cli {
 public:
  /**
   * @brief Parse @p argv against @p known.
   *
   * A token is the previous flag's value unless it begins with `--`, so
   * `--cpu -1` works and `--rt --mlock` is two bare flags; `--name=value` is
   * the unambiguous spelling.  `--help` (or `-h`) prints @p usage, sets
   * `help()` and skips validating the rest; the caller is expected to return.
   *
   * @param argc  As handed to `main`.
   * @param argv  As handed to `main`.
   * @param known Flag names this program accepts, with or without dashes.
   * @param usage Usage text, printed on `--help`.
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

  /// @brief Whether `--help` was asked for; the usage has already been
  ///        printed, and the program should exit 0.
  bool help() const noexcept { return help_; }

  /// @brief Print the usage text to stdout, with a closing newline.
  void print_usage() const {
    std::fputs(usage_.c_str(), stdout);
    if (usage_.empty() || usage_.back() != '\n') {
      std::fputc('\n', stdout);
    }
  }

  /// @brief Whether @p name appeared at all -- how a bare `--rt` is read.
  bool flag(std::string_view name) const noexcept {
    return find(normalize(name)) != nullptr;
  }

  /// @brief The value given for @p name, or @p fallback when it was absent.
  /// @throws std::runtime_error when the flag was given without a value.
  std::string get(std::string_view name, std::string_view fallback = {}) const {
    const std::string* text = value_of(normalize(name));
    return text != nullptr ? *text : std::string(fallback);
  }

  /// @brief `get()` parsed as a signed integer.
  /// @throws std::runtime_error when the text is not an integer, naming the
  ///         flag.
  long get_long(std::string_view name, long fallback) const {
    const std::string_view key = normalize(name);
    const std::string* text = value_of(key);
    if (text == nullptr) {
      return fallback;
    }
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text->c_str(), &end, 10);
    if (end == text->c_str() || *end != '\0' || errno == ERANGE) {
      throw bad_number(key, *text, "an integer");
    }
    return parsed;
  }

  /// @brief `get()` parsed as a count.  A leading `-` is rejected rather than
  ///        wrapped around: `--iterations -1` becoming eighteen quintillion
  ///        iterations is a long wait for a diagnosis.
  /// @throws std::runtime_error when the text is not a non-negative count.
  std::size_t get_size(std::string_view name, std::size_t fallback) const {
    const std::string_view key = normalize(name);
    const std::string* text = value_of(key);
    if (text == nullptr) {
      return fallback;
    }
    if (!text->empty() && text->front() == '-') {
      throw bad_number(key, *text, "a non-negative count");
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text->c_str(), &end, 10);
    if (end == text->c_str() || *end != '\0' || errno == ERANGE) {
      throw bad_number(key, *text, "a non-negative count");
    }
    return static_cast<std::size_t>(parsed);
  }

 private:
  /// One occurrence of a flag, in command-line order.
  struct Entry {
    std::string name;
    std::string value;
    bool has_value = false;
  };

  /// Strip the leading dashes, so `--cpu` and `cpu` name the same flag.
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

  /// The last occurrence wins, so a wrapper script can prepend defaults and
  /// let the caller override them.
  const Entry* find(std::string_view name) const noexcept {
    const Entry* found = nullptr;
    for (const Entry& entry : entries_) {
      if (entry.name == name) {
        found = &entry;
      }
    }
    return found;
  }

  /// The text given for @p name, or nullptr when the flag was absent.
  const std::string* value_of(std::string_view name) const {
    const Entry* entry = find(name);
    if (entry == nullptr) {
      return nullptr;
    }
    if (!entry->has_value) {
      throw std::runtime_error("missing value for '--" + std::string(name) +
                               "'");
    }
    return &entry->value;
  }

  static std::runtime_error bad_number(std::string_view name,
                                       const std::string& text,
                                       const char* expected) {
    return std::runtime_error("'--" + std::string(name) + "' expects " +
                              expected + ", got '" + text + "'");
  }

  std::string usage_;
  std::vector<std::string> known_;
  std::vector<Entry> entries_;
  bool help_ = false;
};

}  // namespace cjfc
