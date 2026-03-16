#ifndef HLSL_TEST_AUTO_DIFF_ENABLE_IF_H
#define HLSL_TEST_AUTO_DIFF_ENABLE_IF_H

namespace hlsl {
template <bool B, typename T> struct enable_if {};

template <typename T> struct enable_if<true, T> {
  using type = T;
};
} // namespace hlsl
#endif // HLSL_TEST_AUTO_DIFF_ENABLE_IF_H
