#ifndef BACKWARD_AD_HLSL
#define BACKWARD_AD_HLSL

#include "type_traits.h"
#include "enable_if.h"

// This is a supplement for not having `auto`, which would be _really_ nice...
#define AUTO_VAR(var,...) __decltype(__VA_ARGS__) var = __VA_ARGS__

namespace ad {
namespace bwd {
// ============================================================================
// Backward Automatic Differentiation for HLSL - Templated Version
// ============================================================================
// This header provides reverse-mode automatic differentiation using expression
// templates. Backward mode is efficient for functions with many inputs and
// few outputs (like gradients for optimization).

// ============================================================================
// Gradient Context - Manages gradient storage per computation
// ============================================================================

template<typename T, int MaxVars = 64>
struct GradientContext
{
    T gradients[MaxVars];
    int variable_count;

    void reset()
    {
        for (int i = 0; i < variable_count; i++)
        {
            gradients[i] = T(0);
        }
        variable_count = 0;
    }

    void zeroGradients()
    {
        for (int i = 0; i < variable_count; i++)
        {
            gradients[i] = (T)0;
        }
    }

    int allocateVariable()
    {
        int id = variable_count;
        variable_count++;
        return id;
    }
};

// ============================================================================
// Variable Class - Represents Input Variables
// ============================================================================

template<typename T>
struct Variable
{
    using ValueType = T;
    T value;
    int id;

    // Get the current gradient for this variable from context
    T gradient(inout GradientContext<T> context)
    {
        return context.gradients[id];
    }

    // Reset gradient to zero in context
    void zeroGradient(inout GradientContext<T> context)
    {
        context.gradients[id] = (T)0;
    }
};

namespace __detail {
  template<typename T>
  typename hlsl::enable_if<!hlsl::is_algebraic<T>::value, typename T::ValueType>::type forward(inout T var)
  {
      return var.forward();
  }

  template<typename T>
  typename hlsl::enable_if<hlsl::is_algebraic<T>::value, T>::type forward(T var)
  {
      return var;
  }

  template<typename T>
  typename hlsl::enable_if<!hlsl::is_algebraic<T>::value, void>::type backward(inout T var, inout GradientContext<typename T::ValueType> context, typename T::ValueType gradient)
  {
      var.backward(context, gradient);
  }

  template<typename T>
  typename hlsl::enable_if<hlsl::is_algebraic<T>::value, void>::type backward(T var, inout GradientContext<T> context, T gradient)
  {
    return; // No gradient on algebraic types.
  }

  // Cofactor matrix helpers for determinant gradient (sizes 1x1 through 4x4)
  // The cofactor matrix C[i][j] = (-1)^(i+j) * det(minor(i,j)).
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
    c[0][0] =  determinant(matrix<T,2,2>(m[1][1], m[1][2], m[2][1], m[2][2]));
    c[0][1] = -determinant(matrix<T,2,2>(m[1][0], m[1][2], m[2][0], m[2][2]));
    c[0][2] =  determinant(matrix<T,2,2>(m[1][0], m[1][1], m[2][0], m[2][1]));
    c[1][0] = -determinant(matrix<T,2,2>(m[0][1], m[0][2], m[2][1], m[2][2]));
    c[1][1] =  determinant(matrix<T,2,2>(m[0][0], m[0][2], m[2][0], m[2][2]));
    c[1][2] = -determinant(matrix<T,2,2>(m[0][0], m[0][1], m[2][0], m[2][1]));
    c[2][0] =  determinant(matrix<T,2,2>(m[0][1], m[0][2], m[1][1], m[1][2]));
    c[2][1] = -determinant(matrix<T,2,2>(m[0][0], m[0][2], m[1][0], m[1][2]));
    c[2][2] =  determinant(matrix<T,2,2>(m[0][0], m[0][1], m[1][0], m[1][1]));
    return c;
  }

  template<typename T>
  matrix<T, 4, 4> cofactor_matrix(matrix<T, 4, 4> m) {
    matrix<T, 4, 4> c;
    c[0][0] =  determinant(matrix<T,3,3>(m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]));
    c[0][1] = -determinant(matrix<T,3,3>(m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]));
    c[0][2] =  determinant(matrix<T,3,3>(m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]));
    c[0][3] = -determinant(matrix<T,3,3>(m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]));
    c[1][0] = -determinant(matrix<T,3,3>(m[0][1], m[0][2], m[0][3], m[2][1], m[2][2], m[2][3], m[3][1], m[3][2], m[3][3]));
    c[1][1] =  determinant(matrix<T,3,3>(m[0][0], m[0][2], m[0][3], m[2][0], m[2][2], m[2][3], m[3][0], m[3][2], m[3][3]));
    c[1][2] = -determinant(matrix<T,3,3>(m[0][0], m[0][1], m[0][3], m[2][0], m[2][1], m[2][3], m[3][0], m[3][1], m[3][3]));
    c[1][3] =  determinant(matrix<T,3,3>(m[0][0], m[0][1], m[0][2], m[2][0], m[2][1], m[2][2], m[3][0], m[3][1], m[3][2]));
    c[2][0] =  determinant(matrix<T,3,3>(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[3][1], m[3][2], m[3][3]));
    c[2][1] = -determinant(matrix<T,3,3>(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[3][0], m[3][2], m[3][3]));
    c[2][2] =  determinant(matrix<T,3,3>(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[3][0], m[3][1], m[3][3]));
    c[2][3] = -determinant(matrix<T,3,3>(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[3][0], m[3][1], m[3][2]));
    c[3][0] = -determinant(matrix<T,3,3>(m[0][1], m[0][2], m[0][3], m[1][1], m[1][2], m[1][3], m[2][1], m[2][2], m[2][3]));
    c[3][1] =  determinant(matrix<T,3,3>(m[0][0], m[0][2], m[0][3], m[1][0], m[1][2], m[1][3], m[2][0], m[2][2], m[2][3]));
    c[3][2] = -determinant(matrix<T,3,3>(m[0][0], m[0][1], m[0][3], m[1][0], m[1][1], m[1][3], m[2][0], m[2][1], m[2][3]));
    c[3][3] =  determinant(matrix<T,3,3>(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]));
    return c;
  }
}


// ============================================================================
// Variable Expression (Leaf Node)
// ============================================================================

template<typename T>
struct VariableExpr
{
  using ValueType = T;
    Variable<T> var;

    T forward()
    {
        return var.value;
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // Accumulate gradient for this variable
        context.gradients[var.id] += gradient;
    }
};

// ============================================================================
// Addition Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackAddExpr
{
  using ValueType = T;
    L left;
    R right;

    T forward()
    {
        return __detail::forward(left) + __detail::forward(right);
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left + right)/dleft = 1, d(left + right)/dright = 1
        __detail::backward(left, context, gradient);
        __detail::backward(right, context, gradient);
    }
};

// ============================================================================
// Subtraction Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackSubExpr
{
  using ValueType = T;
    L left;
    R right;

    T forward()
    {
        return __detail::forward(left) - __detail::forward(right);
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left - right)/dleft = 1, d(left - right)/dright = -1
        __detail::backward(left, context, gradient);
        __detail::backward(right, context, -gradient);
    }
};

// ============================================================================
// Multiplication Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackMulExpr
{
  using ValueType = T;
    L left;
    R right;
    T left_val;
    T right_val;

    T forward()
    {
        left_val = __detail::forward(left);
        right_val = __detail::forward(right);
        return left_val * right_val;
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left * right)/dleft = right, d(left * right)/dright = left
        __detail::backward(left, context, gradient * right_val);
        __detail::backward(right, context, gradient * left_val);
    }
};

// ============================================================================
// Division Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackDivExpr
{
  using ValueType = T;
    L left;
    R right;
    T left_val;
    T right_val;

    T forward()
    {
        left_val = __detail::forward(left);
        right_val = __detail::forward(right);
        return left_val / right_val;
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left / right)/dleft = 1/right, d(left / right)/dright = -left/right^2
        __detail::backward(left, context, gradient / right_val);
        __detail::backward(right, context, -gradient * left_val / (right_val * right_val));
    }
};

// ============================================================================
// Power Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackPowExpr
{
  using ValueType = T;
    L base;
    R exponent;
    T base_val;
    T exp_val;
    T result_val;

    T forward()
    {
        base_val = __detail::forward(base);
        exp_val = __detail::forward(exponent);
        result_val = pow(base_val, exp_val);
        return result_val;
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(base^exp)/dbase = exp * base^(exp-1)
        // d(base^exp)/dexp = base^exp * log(base)
        __detail::backward(base, context, gradient * exp_val * pow(base_val, exp_val - T(1)));
        __detail::backward(exponent, context, gradient * result_val * log(base_val));
    }
};

// ============================================================================
// Negation Expression
// ============================================================================

template<typename T, typename E>
struct BackNegExpr
{
  using ValueType = T;
    E expr;

    T forward()
    {
        return -__detail::forward(expr);
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(-expr)/dexpr = -1
        __detail::backward(expr, context, -gradient);
    }
};

// ============================================================================
// Trigonometric Functions
// ============================================================================

template<typename T, typename E>
struct BackSinExpr
{
  using ValueType = T;
    E expr;
    T expr_val;

    T forward()
    {
        expr_val = __detail::forward(expr);
        return sin(expr_val);
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(sin(x))/dx = cos(x)
        __detail::backward(expr, context, gradient * cos(expr_val));
    }
};

template<typename T, typename E>
struct BackCosExpr
{
  using ValueType = T;
    E expr;
    T expr_val;

    T forward()
    {
        expr_val = __detail::forward(expr);
        return cos(expr_val);
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(cos(x))/dx = -sin(x)
        __detail::backward(expr, context, gradient * (-sin(expr_val)));
    }
};

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

template<typename T, typename E>
struct BackExpExpr
{
  using ValueType = T;
    E expr;
    T result_val;

    T forward()
    {
        T expr_val = __detail::forward(expr);
        result_val = exp(expr_val);
        return result_val;
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(exp(x))/dx = exp(x)
        __detail::backward(expr, context, gradient * result_val);
    }
};

template<typename T, typename E>
struct BackLogExpr
{
  using ValueType = T;
    E expr;
    T expr_val;

    T forward()
    {
        expr_val = __detail::forward(expr);
        return log(expr_val);
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(log(x))/dx = 1/x
        __detail::backward(expr, context, gradient / expr_val);
    }
};

// ============================================================================
// Square Root Expression
// ============================================================================

template<typename T, typename E>
struct BackSqrtExpr
{
  using ValueType = T;
    E expr;
    T expr_val;
    T result_val;

    T forward()
    {
        expr_val = __detail::forward(expr);
        result_val = sqrt(expr_val);
        return result_val;
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(sqrt(x))/dx = 1/(2*sqrt(x))
        __detail::backward(expr, context, gradient / (T(2) * result_val));
    }
};

// ============================================================================
// High-Level API Functions
// ============================================================================

// Create a variable
template<typename T>
Variable<T> variable(inout GradientContext<T> context, T value)
{
    Variable<T> var;
    var.value = value;
    var.id = context.allocateVariable();
    return var;
}

// Create a constant (just returns the typed value)
template<typename T>
T constant(T value)
{
    return value;
}

// Helper function to create VariableExpr
template<typename T>
VariableExpr<T> makeVariableExpr(Variable<T> var)
{
    VariableExpr<T> expr;
    expr.var = var;
    return expr;
}

// Binary operations - using macros to reduce duplication
#define MAKE_BINARY_OP_BACK(ExprType, funcName) \
template<typename T, typename L, typename R> \
ExprType<T, L, R> funcName(L left, R right) \
{ \
    ExprType<T, L, R> expr; \
    expr.left = left; \
    expr.right = right; \
    return expr; \
}

MAKE_BINARY_OP_BACK(BackAddExpr, add)
MAKE_BINARY_OP_BACK(BackSubExpr, subtract)
MAKE_BINARY_OP_BACK(BackMulExpr, multiply)
MAKE_BINARY_OP_BACK(BackDivExpr, divide)

template<typename T, typename L, typename R>
BackPowExpr<T, L, R> power(L base, R exponent)
{
    BackPowExpr<T, L, R> expr;
    expr.base = base;
    expr.exponent = exponent;
    return expr;
}

// Unary operations - using macro to reduce duplication
#define MAKE_UNARY_OP_BACK(ExprType, funcName) \
template<typename T, typename E> \
ExprType<T, E> funcName(E expr) \
{ \
    ExprType<T, E> result; \
    result.expr = expr; \
    return result; \
}

MAKE_UNARY_OP_BACK(BackNegExpr, negate)
MAKE_UNARY_OP_BACK(BackSinExpr, sinExpr)
MAKE_UNARY_OP_BACK(BackCosExpr, cosExpr)
MAKE_UNARY_OP_BACK(BackExpExpr, expExpr)
MAKE_UNARY_OP_BACK(BackLogExpr, logExpr)
MAKE_UNARY_OP_BACK(BackSqrtExpr, sqrtExpr)

// ============================================================================
// Computation Functions - Macro to Reduce Duplication
// ============================================================================

// Macro to generate compute_gradients functions
#define MAKE_COMPUTE_GRADIENTS_BINARY(ExprType) \
template<typename T, typename L, typename R> \
T compute_gradients(inout GradientContext<T> context, ExprType<T, L, R> expr) \
{ \
    context.zeroGradients(); \
    T result = expr.forward(); \
    expr.backward(context, T(1)); \
    return result; \
}

#define MAKE_COMPUTE_GRADIENTS_UNARY(ExprType) \
template<typename T, typename E> \
T compute_gradients(inout GradientContext<T> context, ExprType<T, E> expr) \
{ \
    context.zeroGradients(); \
    T result = expr.forward(); \
    expr.backward(context, T(1)); \
    return result; \
}

// Generate compute_gradients for all expression types
template<typename T>
T compute_gradients(inout GradientContext<T> context, VariableExpr<T> var_expr)
{
    context.zeroGradients();
    T result = var_expr.forward();
    var_expr.backward(context, T(1));
    return result;
}

MAKE_COMPUTE_GRADIENTS_BINARY(BackAddExpr)
MAKE_COMPUTE_GRADIENTS_BINARY(BackSubExpr)
MAKE_COMPUTE_GRADIENTS_BINARY(BackMulExpr)
MAKE_COMPUTE_GRADIENTS_BINARY(BackDivExpr)
MAKE_COMPUTE_GRADIENTS_BINARY(BackPowExpr)

MAKE_COMPUTE_GRADIENTS_UNARY(BackNegExpr)
MAKE_COMPUTE_GRADIENTS_UNARY(BackSinExpr)
MAKE_COMPUTE_GRADIENTS_UNARY(BackCosExpr)
MAKE_COMPUTE_GRADIENTS_UNARY(BackExpExpr)
MAKE_COMPUTE_GRADIENTS_UNARY(BackLogExpr)
MAKE_COMPUTE_GRADIENTS_UNARY(BackSqrtExpr)

// ============================================================================
// Vector and Matrix Support for Backward AD
// ============================================================================

// ============================================================================
// Vector Operations
// ============================================================================

// Vector Dot Product
template<typename T, typename L, typename R>
struct BackDotExpr
{
  using ValueType = T;
    L left;
    R right;
    T left_val;
    T right_val;

    using ElementType = typename hlsl::vector_traits<T>::element_type;

    ElementType forward()
    {
        left_val = __detail::forward(left);
        right_val = __detail::forward(right);
        return dot(left_val, right_val);
    }

    void backward(inout GradientContext<T> context, ElementType gradient)
    {
        // d(dot(u,v))/du = v, d(dot(u,v))/dv = u
        __detail::backward(left, context, right_val * gradient);
        __detail::backward(right, context, left_val * gradient);
    }
};

// Vector Cross Product (3D only)
template<typename T, typename L, typename R>
struct BackCrossExpr
{
  using ValueType = T;
    L left;
    R right;
    T left_val;
    T right_val;

    T forward()
    {
        left_val = __detail::forward(left);
        right_val = __detail::forward(right);
        return cross(left_val, right_val);
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(cross(u,v))/du = cross(gradient, v), d(cross(u,v))/dv = cross(u, gradient)
        __detail::backward(left, context, cross(gradient, right_val));
        __detail::backward(right, context, cross(left_val, gradient));
    }
};

// Vector Length
template<typename T, typename E>
struct BackLengthExpr
{
  using ElementType = typename hlsl::vector_traits<T>::element_type;
  using ValueType = ElementType;
    E expr;
    T expr_val;
    ElementType result_val;

    ElementType forward()
    {
        expr_val = __detail::forward(expr);
        result_val = length(expr_val);
        return result_val;
    }

    void backward(inout GradientContext<T> context, ElementType gradient)
    {
        // d(|v|)/dv = v / |v|
        __detail::backward(expr, context, (expr_val / result_val) * gradient);
    }
};

// Vector Normalize
template<typename T, typename E>
struct BackNormalizeExpr
{
  using ElementType = typename hlsl::vector_traits<T>::element_type;
  using ValueType = T;
    E expr;
    T expr_val;
    T result_val;
    ElementType length_val;

    T forward()
    {
        expr_val = __detail::forward(expr);
        length_val = length(expr_val);
        result_val = normalize(expr_val);
        return result_val;
    }

    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(normalize(v))/dv = (I - normalize(v) * normalize(v)^T) / |v|
        // Simplified: (gradient * length - result * dot(gradient, result)) / length
        ElementType dot_grad_result = dot(gradient, result_val);
        T back_grad = (gradient * length_val - result_val * dot_grad_result) / length_val;
        __detail::backward(expr, context, back_grad);
    }
};

// ============================================================================
// Matrix Operations
// ============================================================================

// Matrix-Vector Multiplication
template<typename M, typename V, typename L, typename R>
struct BackMatVecMulExpr
{
  using ElementType = typename hlsl::matrix_traits<M>::element_type;
  static const int Rows = hlsl::matrix_traits<M>::num_rows;
  static const int Cols = hlsl::matrix_traits<M>::num_columns;
  using ValueType = V;
    L left;   // Matrix
    R right;  // Vector
    M left_val;
    vector<ElementType, Cols> right_val;

    V forward()
    {
        left_val = __detail::forward(left);
        right_val = __detail::forward(right);
        return mul(left_val, right_val);
    }

    void backward(inout GradientContext<V> context, V gradient)
    {
        // d(A*v)/dA = v * gradient^T, d(A*v)/dv = A^T * gradient
        // For matrix gradient, we need a different context type - simplified here
        __detail::backward(right, context, mul(transpose(left_val), gradient));
    }
};

// Matrix Determinant (square matrices 1x1 through 4x4)
template<typename M, typename E>
struct BackDetExpr
{
  using ElementType = typename hlsl::matrix_traits<M>::element_type;
  using ValueType = ElementType;
    E expr;
    M expr_val;

    ElementType forward()
    {
        expr_val = __detail::forward(expr);
        return determinant(expr_val);
    }

    void backward(inout GradientContext<M> context, ElementType gradient)
    {
        // d(det(M))/dM = cofactor_matrix(M)
        __detail::backward(expr, context, __detail::cofactor_matrix(expr_val) * gradient);
    }
};

MAKE_BINARY_OP_BACK(BackDotExpr, dotProduct)
MAKE_BINARY_OP_BACK(BackCrossExpr, crossProduct)

// Macro for vector unary operations
#define MAKE_VECTOR_UNARY_OP_BACK(ExprType, funcName) \
template<typename T, typename E> \
ExprType<T, E> funcName(E expr) \
{ \
    ExprType<T, E> result; \
    result.expr = expr; \
    return result; \
}

MAKE_VECTOR_UNARY_OP_BACK(BackLengthExpr, lengthExpr)
MAKE_VECTOR_UNARY_OP_BACK(BackNormalizeExpr, normalizeExpr)

template<typename M, typename V, typename L, typename R>
BackMatVecMulExpr<M, V, L, R> matVecMul(L left, R right)
{
    BackMatVecMulExpr<M, V, L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename M, typename E>
BackDetExpr<M, E> determinantExpr(E expr)
{
    BackDetExpr<M, E> result;
    result.expr = expr;
    return result;
}

// ============================================================================
// Vector/Matrix Compute Gradients Functions - Reduced with Macros
// ============================================================================

// Note: Vector/matrix compute_gradients have different context and seed types
// so they need individual implementations, but can share some patterns

template<typename T, typename L, typename R>
typename hlsl::vector_traits<T>::element_type compute_gradients(inout GradientContext<T> context, BackDotExpr<T, L, R> expr)
{
    using ElementType = typename hlsl::vector_traits<T>::element_type;
    context.zeroGradients();
    ElementType result = expr.forward();
    expr.backward(context, (ElementType)1);
    return result;
}

template<typename T, typename L, typename R>
T compute_gradients(inout GradientContext<T> context, BackCrossExpr<T, L, R> expr)
{
    using ElementType = typename hlsl::vector_traits<T>::element_type;
    context.zeroGradients();
    T result = expr.forward();
    T seed = (T)1;
    expr.backward(context, seed);
    return result;
}

// Macro for vector unary compute_gradients with scalar return type
#define MAKE_VECTOR_COMPUTE_GRADIENTS_SCALAR(ExprType) \
template<typename T, typename E> \
typename hlsl::vector_traits<T>::element_type compute_gradients(inout GradientContext<T> context, ExprType<T, E> expr) \
{ \
    using ElementType = typename hlsl::vector_traits<T>::element_type; \
    context.zeroGradients(); \
    ElementType result = expr.forward(); \
    expr.backward(context, (ElementType)1); \
    return result; \
}

MAKE_VECTOR_COMPUTE_GRADIENTS_SCALAR(BackLengthExpr)

// Macro for vector unary compute_gradients with vector return type
#define MAKE_VECTOR_COMPUTE_GRADIENTS_VECTOR(ExprType) \
template<typename T, typename E> \
T compute_gradients(inout GradientContext<T> context, ExprType<T, E> expr) \
{ \
    using ElementType = typename hlsl::vector_traits<T>::element_type; \
    static const int N = hlsl::vector_traits<T>::num_elements; \
    context.zeroGradients(); \
    T result = expr.forward(); \
    T seed = (T)0; \
    if (N >= 1) seed[0] = (ElementType)1; \
    expr.backward(context, seed); \
    return result; \
}

MAKE_VECTOR_COMPUTE_GRADIENTS_VECTOR(BackNormalizeExpr)

template<typename M, typename V, typename L, typename R>
V compute_gradients(inout GradientContext<V> context, BackMatVecMulExpr<M, V, L, R> expr)
{
    using ElementType = typename hlsl::vector_traits<V>::element_type;
    static const int N = hlsl::vector_traits<V>::num_elements;
    context.zeroGradients();
    V result = expr.forward();
    V seed = (V)0;
    if (N >= 1) seed[0] = (ElementType)1;
    expr.backward(context, seed);
    return result;
}

template<typename M, typename E>
typename hlsl::matrix_traits<M>::element_type compute_gradients(inout GradientContext<M> context, BackDetExpr<M, E> expr)
{
    using ElementType = typename hlsl::matrix_traits<M>::element_type;
    context.zeroGradients();
    ElementType result = expr.forward();
    expr.backward(context, (ElementType)1);
    return result;
}

// Additional overload for matrix-vector multiplication with matrix context
template<typename M, typename V, typename L, typename R>
V compute_gradients(inout GradientContext<M> context, BackMatVecMulExpr<M, V, L, R> expr)
{
    using ElementType = typename hlsl::vector_traits<V>::element_type;
    static const int N = hlsl::vector_traits<V>::num_elements;
    context.zeroGradients();
    V result = expr.forward();
    V seed = (V)0;
    if (N >= 1) seed[0] = (ElementType)1;
    expr.backward(context, seed);
    return result;
}

} // namespace bwd
} // namespace ad

#endif // BACKWARD_AD_HLSL
