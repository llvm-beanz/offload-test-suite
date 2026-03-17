#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

// Shared matrix utility functions for automatic differentiation.
// Provides minor matrix extraction, cofactor matrices, and determinant
// derivatives for square matrices from 1x1 through 4x4.

namespace ad {
namespace __detail {

// ============================================================================
// Minor Matrix Extraction
// ============================================================================
// Extract the (N-1)x(N-1) submatrix by removing the given row and column.

template<typename T>
matrix<T, 2, 2> minor_matrix(matrix<T, 3, 3> m, int row, int col) {
  matrix<T, 2, 2> result;
  int ri = 0;
  [unroll] for (int r = 0; r < 3; r++) {
    if (r == row) continue;
    int ci = 0;
    [unroll] for (int c = 0; c < 3; c++) {
      if (c == col) continue;
      result[ri][ci] = m[r][c];
      ci++;
    }
    ri++;
  }
  return result;
}

template<typename T>
matrix<T, 3, 3> minor_matrix(matrix<T, 4, 4> m, int row, int col) {
  matrix<T, 3, 3> result;
  int ri = 0;
  [unroll] for (int r = 0; r < 4; r++) {
    if (r == row) continue;
    int ci = 0;
    [unroll] for (int c = 0; c < 4; c++) {
      if (c == col) continue;
      result[ri][ci] = m[r][c];
      ci++;
    }
    ri++;
  }
  return result;
}

// ============================================================================
// Cofactor Matrix
// ============================================================================
// C[i][j] = (-1)^(i+j) * det(minor(i,j))
// d(det(M))/dM = cofactor_matrix(M)

template<typename T>
matrix<T, 1, 1> cofactor_matrix(matrix<T, 1, 1> m) {
  matrix<T, 1, 1> c;
  c[0][0] = T(1);
  return c;
}

template<typename T>
matrix<T, 2, 2> cofactor_matrix(matrix<T, 2, 2> m) {
  matrix<T, 2, 2> c;
  c[0][0] =  m[1][1];
  c[0][1] = -m[1][0];
  c[1][0] = -m[0][1];
  c[1][1] =  m[0][0];
  return c;
}

template<typename T>
matrix<T, 3, 3> cofactor_matrix(matrix<T, 3, 3> m) {
  matrix<T, 3, 3> c;
  [unroll] for (int i = 0; i < 3; i++)
    [unroll] for (int j = 0; j < 3; j++)
      c[i][j] = ((i + j) % 2 == 0 ? T(1) : T(-1))
                * determinant(minor_matrix(m, i, j));
  return c;
}

template<typename T>
matrix<T, 4, 4> cofactor_matrix(matrix<T, 4, 4> m) {
  matrix<T, 4, 4> c;
  [unroll] for (int i = 0; i < 4; i++)
    [unroll] for (int j = 0; j < 4; j++)
      c[i][j] = ((i + j) % 2 == 0 ? T(1) : T(-1))
                * determinant(minor_matrix(m, i, j));
  return c;
}

// ============================================================================
// Determinant Derivative (for forward-mode AD)
// ============================================================================
// d/dt det(M(t)) = sum_{i,j} cofactor(i,j) * dM[i][j]

template<typename T>
T det_deriv(matrix<T, 1, 1> m, matrix<T, 1, 1> dm) {
  return dm[0][0];
}

template<typename T>
T det_deriv(matrix<T, 2, 2> m, matrix<T, 2, 2> dm) {
  return  m[1][1] * dm[0][0] - m[1][0] * dm[0][1]
        - m[0][1] * dm[1][0] + m[0][0] * dm[1][1];
}

template<typename T>
T det_deriv(matrix<T, 3, 3> m, matrix<T, 3, 3> dm) {
  T result = T(0);
  [unroll] for (int i = 0; i < 3; i++)
    [unroll] for (int j = 0; j < 3; j++)
      result += ((i + j) % 2 == 0 ? T(1) : T(-1))
                * determinant(minor_matrix(m, i, j)) * dm[i][j];
  return result;
}

template<typename T>
T det_deriv(matrix<T, 4, 4> m, matrix<T, 4, 4> dm) {
  T result = T(0);
  [unroll] for (int i = 0; i < 4; i++)
    [unroll] for (int j = 0; j < 4; j++)
      result += ((i + j) % 2 == 0 ? T(1) : T(-1))
                * determinant(minor_matrix(m, i, j)) * dm[i][j];
  return result;
}

} // namespace __detail
} // namespace ad

#endif // MATRIX_UTILS_H
