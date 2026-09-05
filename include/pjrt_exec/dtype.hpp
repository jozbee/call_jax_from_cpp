/**
 * @file dtype.hpp
 * @brief The element types this project supports, and their PJRT spellings.
 *
 * XLA has many more element types than a float64 control loop needs, and the
 * ones left out -- F16, BF16, complex, the FP8 families, the sub-byte integers
 * -- have no C++ storage type this API could hand back.  `DType` is therefore
 * a closed set of eleven: exactly those that map onto a C++ scalar, spelled
 * the way NumPy spells them so the JSON sidecar the exporter writes can be
 * read back without a translation table.
 *
 * Conversion runs both ways.  `to_pjrt` is total.  `from_pjrt` is partial and
 * returns `std::nullopt` for everything outside the set, which is how the
 * loader notices an executable it cannot represent instead of handing the
 * caller a pointer to bytes it would misread; `pjrt_type_name` then names the
 * offending type in the error message.
 *
 * Header-only and free of runtime state.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>

#include "pjrt/pjrt_c_api.h"

namespace pjrt {

/**
 * @brief An element type shared by JAX, NumPy, PJRT and C++.
 *
 * `Bool` is one byte holding 0 or 1 -- PJRT's `PRED`, the same storage as
 * NumPy's `bool_` and as C++ `bool` on every platform this builds for.  Any
 * other byte value is *undefined* for XLA rather than merely truthy: it does
 * not normalize the byte, so a stray 2 can make a predicate read as both true
 * and false within one computation.  That is why
 * `FunctionOptions::check_values` audits bool arenas instead of trusting the
 * caller to have written a clean 0 or 1.
 */
enum class DType : std::uint8_t {
  Bool,
  Int8,
  Int16,
  Int32,
  Int64,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
  Float32,
  Float64,
};

/// Bytes one element occupies -- in an arena, and in the sidecar's `nbytes`.
constexpr std::size_t itemsize(DType dtype) noexcept {
  switch (dtype) {
    case DType::Bool:
      return 1;
    case DType::Int8:
      return 1;
    case DType::Int16:
      return 2;
    case DType::Int32:
      return 4;
    case DType::Int64:
      return 8;
    case DType::UInt8:
      return 1;
    case DType::UInt16:
      return 2;
    case DType::UInt32:
      return 4;
    case DType::UInt64:
      return 8;
    case DType::Float32:
      return 4;
    case DType::Float64:
      return 8;
  }
  return 0;  // Unreachable for a valid enumerator; -Wreturn-type wants it.
}

/**
 * @brief The NumPy name: `"bool"`, `"int8"` ... `"float64"`.
 *
 * These are the strings the sidecar carries, so this is also the inverse of
 * `parse_dtype` and the vocabulary of the dtype-mismatch error messages.
 */
constexpr const char* dtype_name(DType dtype) noexcept {
  switch (dtype) {
    case DType::Bool:
      return "bool";
    case DType::Int8:
      return "int8";
    case DType::Int16:
      return "int16";
    case DType::Int32:
      return "int32";
    case DType::Int64:
      return "int64";
    case DType::UInt8:
      return "uint8";
    case DType::UInt16:
      return "uint16";
    case DType::UInt32:
      return "uint32";
    case DType::UInt64:
      return "uint64";
    case DType::Float32:
      return "float32";
    case DType::Float64:
      return "float64";
  }
  return "?";
}

/**
 * @brief Parse a NumPy dtype name, exactly as `dtype_name` writes it.
 *
 * Deliberately strict: no aliases, no `"float"`/`"double"`, no byte-order
 * prefixes.  A sidecar written by this project's exporter always uses the
 * canonical name, so anything else is a sidecar worth rejecting rather than
 * guessing at.
 */
inline std::optional<DType> parse_dtype(std::string_view name) noexcept {
  constexpr DType kAll[] = {DType::Bool,    DType::Int8,   DType::Int16,
                            DType::Int32,   DType::Int64,  DType::UInt8,
                            DType::UInt16,  DType::UInt32, DType::UInt64,
                            DType::Float32, DType::Float64};
  for (const DType dtype : kAll) {
    if (name == std::string_view(dtype_name(dtype))) {
      return dtype;
    }
  }
  return std::nullopt;
}

/// Whether values of this type can be nan or inf, i.e. whether
/// `FunctionOptions::check_values` has anything to test in the arena.
constexpr bool is_floating(DType dtype) noexcept {
  return dtype == DType::Float32 || dtype == DType::Float64;
}

/**
 * @brief The PJRT enumerator for a `DType`.
 *
 * The eleven enumerators involved (`PRED`, `S8`..`S64`, `U8`..`U64`, `F32`,
 * `F64`) hold the same values in PJRT C API 0.90 and 0.114, so an artifact
 * described against one header loads against the other.
 */
constexpr PJRT_Buffer_Type to_pjrt(DType dtype) noexcept {
  switch (dtype) {
    case DType::Bool:
      return PJRT_Buffer_Type_PRED;
    case DType::Int8:
      return PJRT_Buffer_Type_S8;
    case DType::Int16:
      return PJRT_Buffer_Type_S16;
    case DType::Int32:
      return PJRT_Buffer_Type_S32;
    case DType::Int64:
      return PJRT_Buffer_Type_S64;
    case DType::UInt8:
      return PJRT_Buffer_Type_U8;
    case DType::UInt16:
      return PJRT_Buffer_Type_U16;
    case DType::UInt32:
      return PJRT_Buffer_Type_U32;
    case DType::UInt64:
      return PJRT_Buffer_Type_U64;
    case DType::Float32:
      return PJRT_Buffer_Type_F32;
    case DType::Float64:
      return PJRT_Buffer_Type_F64;
  }
  return PJRT_Buffer_Type_INVALID;
}

/**
 * @brief The `DType` for a PJRT enumerator, or `std::nullopt` if unsupported.
 *
 * Everything outside the eleven -- F16, BF16, C64, C128, the F8/F6/F4
 * families, S4/U4/S2/U2/S1/U1, TOKEN, INVALID -- returns `std::nullopt`, which
 * the loader turns into a `LoadError` naming the type.  The `default` here is
 * the conservative answer for element types XLA has not invented yet;
 * `pjrt_type_name` is the exhaustive switch, so re-vendoring a header that
 * adds a type produces a `-Wswitch` warning there and forces a decision.
 */
inline std::optional<DType> from_pjrt(PJRT_Buffer_Type type) noexcept {
  switch (type) {
    case PJRT_Buffer_Type_PRED:
      return DType::Bool;
    case PJRT_Buffer_Type_S8:
      return DType::Int8;
    case PJRT_Buffer_Type_S16:
      return DType::Int16;
    case PJRT_Buffer_Type_S32:
      return DType::Int32;
    case PJRT_Buffer_Type_S64:
      return DType::Int64;
    case PJRT_Buffer_Type_U8:
      return DType::UInt8;
    case PJRT_Buffer_Type_U16:
      return DType::UInt16;
    case PJRT_Buffer_Type_U32:
      return DType::UInt32;
    case PJRT_Buffer_Type_U64:
      return DType::UInt64;
    case PJRT_Buffer_Type_F32:
      return DType::Float32;
    case PJRT_Buffer_Type_F64:
      return DType::Float64;
    default:
      return std::nullopt;
  }
}

/**
 * @brief The XLA spelling of any PJRT element type, supported or not.
 *
 * Exists so that "output 2 has element type BF16, which pjrt_exec does not
 * support" can name what it found.  Covers the whole enum on purpose: when a
 * re-vendored header adds an element type, this switch is where the compiler
 * says so.
 */
inline const char* pjrt_type_name(PJRT_Buffer_Type type) noexcept {
  switch (type) {
    case PJRT_Buffer_Type_INVALID:
      return "INVALID";
    case PJRT_Buffer_Type_PRED:
      return "PRED";
    case PJRT_Buffer_Type_S8:
      return "S8";
    case PJRT_Buffer_Type_S16:
      return "S16";
    case PJRT_Buffer_Type_S32:
      return "S32";
    case PJRT_Buffer_Type_S64:
      return "S64";
    case PJRT_Buffer_Type_U8:
      return "U8";
    case PJRT_Buffer_Type_U16:
      return "U16";
    case PJRT_Buffer_Type_U32:
      return "U32";
    case PJRT_Buffer_Type_U64:
      return "U64";
    case PJRT_Buffer_Type_F16:
      return "F16";
    case PJRT_Buffer_Type_F32:
      return "F32";
    case PJRT_Buffer_Type_F64:
      return "F64";
    case PJRT_Buffer_Type_BF16:
      return "BF16";
    case PJRT_Buffer_Type_C64:
      return "C64";
    case PJRT_Buffer_Type_C128:
      return "C128";
    case PJRT_Buffer_Type_F8E5M2:
      return "F8E5M2";
    case PJRT_Buffer_Type_F8E4M3FN:
      return "F8E4M3FN";
    case PJRT_Buffer_Type_F8E4M3B11FNUZ:
      return "F8E4M3B11FNUZ";
    case PJRT_Buffer_Type_F8E5M2FNUZ:
      return "F8E5M2FNUZ";
    case PJRT_Buffer_Type_F8E4M3FNUZ:
      return "F8E4M3FNUZ";
    case PJRT_Buffer_Type_S4:
      return "S4";
    case PJRT_Buffer_Type_U4:
      return "U4";
    case PJRT_Buffer_Type_TOKEN:
      return "TOKEN";
    case PJRT_Buffer_Type_S2:
      return "S2";
    case PJRT_Buffer_Type_U2:
      return "U2";
    case PJRT_Buffer_Type_F8E4M3:
      return "F8E4M3";
    case PJRT_Buffer_Type_F8E3M4:
      return "F8E3M4";
    case PJRT_Buffer_Type_F8E8M0FNU:
      return "F8E8M0FNU";
    case PJRT_Buffer_Type_F4E2M1FN:
      return "F4E2M1FN";
    case PJRT_Buffer_Type_S1:
      return "S1";
    case PJRT_Buffer_Type_U1:
      return "U1";
    case PJRT_Buffer_Type_F6E2M3FN:
      return "F6E2M3FN";
    case PJRT_Buffer_Type_F6E3M2FN:
      return "F6E3M2FN";
  }
  return "UNKNOWN";
}

namespace detail {

/// Map an integer's width and signedness onto a `DType`.  Split out so the
/// partial specialization below stays a one-liner.
constexpr DType integral_dtype(std::size_t size, bool is_signed) noexcept {
  if (is_signed) {
    return size == 1   ? DType::Int8
           : size == 2 ? DType::Int16
           : size == 4 ? DType::Int32
                       : DType::Int64;
  }
  return size == 1   ? DType::UInt8
         : size == 2 ? DType::UInt16
         : size == 4 ? DType::UInt32
                     : DType::UInt64;
}

}  // namespace detail

/**
 * @brief The `DType` a C++ scalar type stores, for the typed accessors.
 *
 * Declared and never defined, so `Function::input<T>()` with an unsupported
 * `T` fails to compile with the offending type named in the diagnostic
 * ("implicit instantiation of undefined template 'pjrt::dtype_of<std::string>'
 * ") rather than at run time or, worse, not at all.
 *
 * Integers map by width and signedness, so `long`, `long long` and
 * `std::int64_t` all land on `DType::Int64` whatever the platform calls them.
 * Plain `char` is a distinct type from both `signed char` and `unsigned char`
 * and maps by whatever signedness this platform gives it -- x86-64 Linux
 * signed, aarch64 Linux unsigned -- so an artifact's dtype would follow the
 * compiler rather than the artifact.  Write `std::int8_t` / `std::uint8_t` in
 * calling code and let the mismatch check do its job.
 */
template <class T, class = void>
struct dtype_of;

/// `bool` is one byte holding 0 or 1, matching PJRT `PRED`.
template <>
struct dtype_of<bool> {
  static constexpr DType value = DType::Bool;
};

/// IEEE binary32.
template <>
struct dtype_of<float> {
  static constexpr DType value = DType::Float32;
};

/// IEEE binary64, the type the exporter produces with `jax_enable_x64`.
template <>
struct dtype_of<double> {
  static constexpr DType value = DType::Float64;
};

/// Every integer type except `bool`, keyed on width and signedness rather than
/// on spelling, so the typedef a caller happens to use does not matter.
template <class T>
struct dtype_of<
    T, std::enable_if_t<std::is_integral_v<T> &&
                        !std::is_same_v<std::remove_cv_t<T>, bool>>> {
  static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 ||
                    sizeof(T) == 8,
                "no PJRT element type for an integer of this width");
  static constexpr DType value =
      detail::integral_dtype(sizeof(T), std::is_signed_v<T>);
};

/// Shorthand for `dtype_of<T>::value`.
template <class T>
inline constexpr DType dtype_of_v = dtype_of<T>::value;

}  // namespace pjrt
