#ifndef HLSL_TEST_AUTO_DIFF_TYPE_TRAITS_H
#define HLSL_TEST_AUTO_DIFF_TYPE_TRAITS_H

namespace hlsl {
template <typename T> struct is_arithmetic {
  static const bool value = false;
};

#define __ARITHMETIC_TYPE(type)                                                \
  template <> struct is_arithmetic<type> {                                     \
    static const bool value = true;                                            \
  };

#if __HLSL_ENABLE_16_BIT
__ARITHMETIC_TYPE(uint16_t)
__ARITHMETIC_TYPE(int16_t)
__ARITHMETIC_TYPE(float16_t)
#endif
__ARITHMETIC_TYPE(uint)
__ARITHMETIC_TYPE(int)
__ARITHMETIC_TYPE(uint64_t)
__ARITHMETIC_TYPE(int64_t)
__ARITHMETIC_TYPE(half)
__ARITHMETIC_TYPE(float)
__ARITHMETIC_TYPE(double)

template <typename T> struct vector_traits {
};

template<typename T, int N> struct vector_traits<vector<T, N> > {
  using element_type = T;
  static const int num_elements = N;
};

template <typename T> struct is_vector {
  static const bool value = false;
};

template <typename T, int N> struct is_vector<vector<T, N> > {
  static const bool value = true;
};

template <typename T> struct is_matrix {
  static const bool value = false;
};

template <typename T, int N, int M> struct is_matrix<matrix<T, N, M> > {
  static const bool value = true;
};

template <typename T> struct matrix_traits {
};

template<typename T, int N, int M> struct matrix_traits<matrix<T, N, M> > {
  using element_type = T;
  static const int num_rows = N;
  static const int num_cols = M;
};

template <typename T> struct is_algebraic {
  static const bool value = is_arithmetic<T>::value || is_vector<T>::value ||
                            is_matrix<T>::value;
};

} // namespace hlsl

#endif // HLSL_TEST_AUTO_DIFF_TYPE_TRAITS_H
