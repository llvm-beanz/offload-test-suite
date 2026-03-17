#ifndef FORWARD_AD_HLSL
#define FORWARD_AD_HLSL

#include "type_traits.h"
#include "enable_if.h"
#include "matrix_utils.h"

namespace ad {
namespace fwd {
// Forward Automatic Differentiation for HLSL
// This header provides Value numbers for automatic differentiation with templated types

// ============================================================================
// Value Number Structure - templated for different types
// ============================================================================

template<typename T>
struct Value
{
    T value;
    T derivative;

    // Static creation functions
    static Value<T> Create(T v, T d)
    {
        Value<T> result;
        result.value = v;
        result.derivative = d;
        return result;
    }

    static Value<T> CreateValue(T v)
    {
        Value<T> result;
        result.value = v;
        result.derivative = (T)0;
        return result;
    }

    static Value<T> CreateZero()
    {
        Value<T> result;
        result.value = (T)0;
        result.derivative = (T)0;
        return result;
    }

    // Member operators
    Value<T> operator+(Value<T> other)
    {
        return Value<T>::Create(value + other.value, derivative + other.derivative);
    }

    Value<T> operator+(T other)
    {
        return Value<T>::Create(value + other, derivative);
    }

    Value<T> operator-(Value<T> other)
    {
        return Value<T>::Create(value - other.value, derivative - other.derivative);
    }

    Value<T> operator-(T other)
    {
        return Value<T>::Create(value - other, derivative);
    }

    Value<T> operator*(Value<T> other)
    {
        return Value<T>::Create(value * other.value,
                       derivative * other.value + value * other.derivative);
    }

    Value<T> operator*(T other)
    {
        return Value<T>::Create(value * other, derivative * other);
    }

    Value<T> operator/(Value<T> other)
    {
        T denom = other.value * other.value;
        return Value<T>::Create(value / other.value,
                       (derivative * other.value - value * other.derivative) / denom);
    }

    Value<T> operator/(T other)
    {
        return Value<T>::Create(value / other, derivative / other);
    }

    // Unary operator overloading does not work on DXC:
    // https://github.com/microsoft/DirectXShaderCompiler/issues/7944
    // This code will work with Clang so it is left here.
  #if !__hlsl_dx_compiler
    Value<T> operator-()
    {
        return Value<T>::Create(-value, -derivative);
    }
  #endif
};

#if !__hlsl_dx_compiler
// Non-member operator overloading does not work in DXC, so these are only
// available when using Clang.
template<typename T>
Value<T> operator+(T lhs, Value<T> rhs)
{
    return Value<T>::Create(lhs + rhs.value, rhs.derivative);
}

template<typename T>
Value<T> operator-(T lhs, Value<T> rhs)
{
    return Value<T>::Create(lhs - rhs.value, -rhs.derivative);
}

template<typename T>
Value<T> operator*(T lhs, Value<T> rhs)
{
    return Value<T>::Create(lhs * rhs.value, lhs * rhs.derivative);
}

template<typename T>
Value<T> operator/(T lhs, Value<T> rhs)
{
    return Value<T>::Create(lhs / rhs.value, -lhs * rhs.derivative / (rhs.value * rhs.value));
}
#endif

// ============================================================================
// Helper Functions for Value Extraction
// ============================================================================

namespace __detail {

// These wrappers allow the templates below to expressions or values
// interchangably.
// Extract Value from templated expression types
template <typename T>
typename T::ResultType getValue(T value)
{
    return value.eval();
}


// Extract Value from Value (identity)
template<typename T>
Value<T> getValue(Value<T> d)
{
    return d;
}
} // namespace __detail


// ============================================================================
// Binary Operation Expression Templates
// ============================================================================

// Addition Expression
template<typename T, typename L, typename R>
struct AddExpr
{
    using ResultType = Value<T>;
    L left;
    R right;

    Value<T> eval()
    {
        Value<T> l_val = __detail::getValue(left);
        Value<T> r_val = __detail::getValue(right);
        return Value<T>::Create(l_val.value + r_val.value,
                   l_val.derivative + r_val.derivative);
    }
};

// Subtraction Expression
template<typename T, typename L, typename R>
struct SubExpr
{
    using ResultType = Value<T>;
    L left;
    R right;

    Value<T> eval()
    {
        Value<T> l_val = __detail::getValue(left);
        Value<T> r_val = __detail::getValue(right);
        return Value<T>::Create(l_val.value - r_val.value,
                   l_val.derivative - r_val.derivative);
    }
};

// Multiplication Expression
template<typename T, typename L, typename R>
struct MulExpr
{
    using ResultType = Value<T>;
    L left;
    R right;

    Value<T> eval()
    {
        Value<T> l_val = __detail::getValue(left);
        Value<T> r_val = __detail::getValue(right);
        // Product rule: (f*g)' = f'*g + f*g'
        return Value<T>::Create(l_val.value * r_val.value,
                   l_val.derivative * r_val.value + l_val.value * r_val.derivative);
    }
};

// Division Expression
template<typename T, typename L, typename R>
struct DivExpr
{
    using ResultType = Value<T>;
    L left;
    R right;

    Value<T> eval()
    {
        Value<T> l_val = __detail::getValue(left);
        Value<T> r_val = __detail::getValue(right);
        // Quotient rule: (f/g)' = (f'*g - f*g') / g^2
        T denom = r_val.value * r_val.value;
        return Value<T>::Create(l_val.value / r_val.value,
                   (l_val.derivative * r_val.value - l_val.value * r_val.derivative) / denom);
    }
};

// Power Expression
template<typename T, typename L, typename R>
struct PowExpr
{
    using ResultType = Value<T>;
    L left;
    R right;

    Value<T> eval()
    {
        Value<T> b_val = __detail::getValue(left);
        Value<T> e_val = __detail::getValue(right);
        // Power rule: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        T pow_val = pow(b_val.value, e_val.value);
        T deriv = pow_val * (e_val.derivative * log(b_val.value) +
                            e_val.value * b_val.derivative / b_val.value);
        return Value<T>::Create(pow_val, deriv);
    }
};

// ============================================================================
// Unary Operation Expression Templates
// ============================================================================

// Negation Expression
template<typename T, typename E>
struct NegExpr
{
    using ResultType = Value<T>;
    E expr;

    Value<T> eval()
    {
        Value<T> val = __detail::getValue(expr);
        return Value<T>::Create(-val.value, -val.derivative);
    }
};

// Sine Expression
template<typename T, typename E>
struct SinExpr
{
    using ResultType = Value<T>;
    E expr;

    Value<T> eval()
    {
        Value<T> val = __detail::getValue(expr);
        // d/dx[sin(x)] = cos(x) * x'
        return Value<T>::Create(sin(val.value), cos(val.value) * val.derivative);
    }
};

// Cosine Expression
template<typename T, typename E>
struct CosExpr
{
    using ResultType = Value<T>;
    E expr;

    Value<T> eval()
    {
        Value<T> val = __detail::getValue(expr);
        // d/dx[cos(x)] = -sin(x) * x'
        return Value<T>::Create(cos(val.value), -sin(val.value) * val.derivative);
    }
};

// Exponential Expression
template<typename T, typename E>
struct ExpExpr
{
    using ResultType = Value<T>;
    E expr;

    Value<T> eval()
    {
        Value<T> val = __detail::getValue(expr);
        // d/dx[exp(x)] = exp(x) * x'
        T exp_val = exp(val.value);
        return Value<T>::Create(exp_val, exp_val * val.derivative);
    }
};

// Natural Logarithm Expression
template<typename T, typename E>
struct LogExpr
{
    using ResultType = Value<T>;
    E expr;

    Value<T> eval()
    {
        Value<T> val = __detail::getValue(expr);
        // d/dx[log(x)] = x' / x
        return Value<T>::Create(log(val.value), val.derivative / val.value);
    }
};

// Square Root Expression
template<typename T, typename E>
struct SqrtExpr
{
    using ResultType = Value<T>;
    E expr;

    Value<T> eval()
    {
        Value<T> val = __detail::getValue(expr);
        // d/dx[sqrt(x)] = x' / (2 * sqrt(x))
        T sqrt_val = sqrt(val.value);
        return Value<T>::Create(sqrt_val, val.derivative / ((T)2 * sqrt_val));
    }
};

// ============================================================================
// Named Functions for Operations with Type Support
// ============================================================================

// Templated operation functions - using macro to reduce duplication
#define MAKE_BINARY_OP(opName, ExprType) \
template<typename T, typename L, typename R> \
ExprType<T, L, R> make##ExprType(L left, R right) \
{ \
    ExprType<T, L, R> result; \
    result.left = left; \
    result.right = right; \
    return result; \
} \
namespace __detail { \
template<typename T, typename L, typename R> \
ExprType<T, L, R> opName(L left, R right) \
{ \
    return make##ExprType<T>(left, right); \
} \
} /* namespace __detail */ \
template<typename T> \
Value<T> opName(Value<T> left, Value<T> right) \
{ \
    return __detail::opName<T>(left, right).eval(); \
} \
template<typename T, typename E> \
typename hlsl::enable_if<hlsl::is_arithmetic<E>::value, Value<T> >::type \
opName(Value<T> left, E right) \
{ \
    return __detail::opName<T>(left, right).eval(); \
} \
template<typename T, typename E> \
typename hlsl::enable_if<hlsl::is_arithmetic<E>::value, Value<T> >::type opName(E left, Value<T> right) \
{ \
    return __detail::opName<T>(left, right).eval(); \
}

MAKE_BINARY_OP(add, AddExpr)
MAKE_BINARY_OP(subtract, SubExpr)
MAKE_BINARY_OP(multiply, MulExpr)
MAKE_BINARY_OP(divide, DivExpr)
MAKE_BINARY_OP(pow, PowExpr)

// Unary operation functions - using macro to reduce duplication
#define MAKE_UNARY_OP(opName, ExprType) \
template<typename T, typename E> \
ExprType<T, E> make##ExprType(E expr) \
{ \
    ExprType<T, E> result; \
    result.expr = expr; \
    return result; \
} \
namespace __detail { \
template<typename T, typename E> \
ExprType<T, E> opName(E expr) \
{ \
    return make##ExprType<T>(expr); \
} \
} /* namespace __detail */ \
template<typename T> \
Value<T> opName(Value<T> arg) \
{ \
    return __detail::opName<T>(arg).eval(); \
} \
template<typename T, typename E> \
typename hlsl::enable_if<hlsl::is_arithmetic<T>::value, Value<T> >::type \
opName(T arg) \
{ \
    return __detail::opName<T>(arg).eval(); \
}

MAKE_UNARY_OP(negate, NegExpr)
MAKE_UNARY_OP(sin, SinExpr)
MAKE_UNARY_OP(cos, CosExpr)
MAKE_UNARY_OP(exp, ExpExpr)
MAKE_UNARY_OP(log, LogExpr)
MAKE_UNARY_OP(sqrt, SqrtExpr)

// ============================================================================
// Utility Functions
// ============================================================================

template<typename T>
Value<T> variable(T value)
{
    return Value<T>::Create(value, (T)1);
}

// Scalar constant function
template<typename T>
Value<T> constant(T value)
{
    return Value<T>::Create(value, (T)0);
}

// ============================================================================
// Vector Operations
// ============================================================================

// Dot Product Expression
template<typename T, typename L, typename R>
struct DotExpr
{
    //_Static_assert(typename hlsl::is_vector<T>::value, "Dot product is only defined for vectors");
    using ElementType = typename hlsl::vector_traits<T>::element_type;
    using ResultType = Value<ElementType>;
    L left;
    R right;

    Value<ElementType> eval()
    {
        Value<T> l_val = __detail::getValue(left);
        Value<T> r_val = __detail::getValue(right);

        // Dot product: d/dx[dot(u,v)] = dot(u',v) + dot(u,v')
        ElementType val = dot(l_val.value, r_val.value);
        ElementType deriv = dot(l_val.derivative, r_val.value) + dot(l_val.value, r_val.derivative);

        return Value<ElementType>::Create(val, deriv);
    }
};

// Cross Product Expression (for 3D vectors only)
template<typename T, typename L, typename R>
struct CrossExpr
{
    //_Static_assert(typename hlsl::is_vector<T>::value, "Cross product is only defined for vectors");
    //_Static_assert(typename hlsl::vector_traits<T>::num_elements == 3, "Cross product is only defined for 3D vectors");
    using ElementType = typename hlsl::vector_traits<T>::element_type;
    using ResultType = Value<vector<ElementType, 3> >;
    L left;
    R right;

    Value<vector<ElementType, 3> > eval()
    {
        Value<vector<ElementType, 3> > l_val = __detail::getValue(left);
        Value<vector<ElementType, 3> > r_val = __detail::getValue(right);

        // Cross product: d/dx[cross(u,v)] = cross(u',v) + cross(u,v')
        vector<ElementType, 3> val = cross(l_val.value, r_val.value);
        vector<ElementType, 3> deriv = cross(l_val.derivative, r_val.value) + cross(l_val.value, r_val.derivative);

        return Value<vector<ElementType, 3> >::Create(val, deriv);
    }
};

// Vector Length Expression
template<typename T, typename E>
struct LengthExpr
{
    using ElementType = typename hlsl::vector_traits<T>::element_type;
    using ResultType = Value<ElementType>;
    E expr;

    Value<ElementType> eval()
    {
        Value<T> val = __detail::getValue(expr);

        // Length: d/dx[|v|] = dot(v, v') / |v|
        ElementType len = length(val.value);
        ElementType deriv = dot(val.value, val.derivative) / len;

        return Value<ElementType>::Create(len, deriv);
    }
};

// Vector Normalize Expression
template<typename T, typename E>
struct NormalizeExpr
{
    using ElementType = typename hlsl::vector_traits<T>::element_type;
    using ResultType = Value<T>;
    E expr;

    Value<T> eval()
    {
        Value<T> val = __detail::getValue(expr);

        // Normalize: d/dx[normalize(v)] = (v' * |v| - v * (dot(v,v') / |v|)) / |v|²
        ElementType len = length(val.value);
        T norm_val = normalize(val.value);
        ElementType dot_deriv = dot(val.value, val.derivative);
        T deriv = (val.derivative * len - val.value * (dot_deriv / len)) / (len * len);

        return Value<T>::Create(norm_val, deriv);
    }
};

// ============================================================================
// Matrix Operations
// ============================================================================

// Matrix Multiplication Expression
template<typename T, int N, int K, int M, typename L, typename R>
struct MatMulExpr
{
    using ResultType = Value<vector<T, N> >;
    L left;   // Matrix NxK
    R right;  // Matrix KxM or Vector K

    // Returns either Value<matrix<T,N,M> > for matrix*matrix or Value<vector<T,N> > for matrix*vector
    // We'll use a specific implementation that works with common cases
    Value<vector<T, N> > eval()  // Assuming matrix-vector multiplication for now
    {
        Value<matrix<T, N, K> > l_val = __detail::getValue(left);
        Value<vector<T, K> > r_val = __detail::getValue(right);

        // Matrix-vector multiplication: d/dx[A*v] = A'*v + A*v'
        vector<T, N> val = mul(l_val.value, r_val.value);
        vector<T, N> deriv = mul(l_val.derivative, r_val.value) + mul(l_val.value, r_val.derivative);

        return Value<vector<T, N> >::Create(val, deriv);
    }
};

// Matrix Transpose Expression
template<typename T, int N, int M, typename E>
struct TransposeExpr
{
    using ResultType = Value<matrix<T, M, N> >;
    E expr;

    Value<matrix<T, M, N> > eval()  // Transpose flips dimensions
    {
        Value<matrix<T, N, M> > val = __detail::getValue(expr);

        // Transpose: d/dx[transpose(M)] = transpose(M')
        matrix<T, M, N> val_t = transpose(val.value);
        matrix<T, M, N> deriv_t = transpose(val.derivative);

        return Value<matrix<T, M, N> >::Create(val_t, deriv_t);
    }
};

// Matrix Determinant Expression (supports 1x1 through 4x4)
template<typename T, int N, typename E>
struct DetExpr
{
    using ResultType = Value<T>;
    E expr;

    Value<T> eval()
    {
        Value<matrix<T, N, N> > val = __detail::getValue(expr);

        T det_val = determinant(val.value);
        T deriv = ad::__detail::det_deriv(val.value, val.derivative);

        return Value<T>::Create(det_val, deriv);
    }
};


// ============================================================================
// Vector and Matrix Operation Functions
// ============================================================================


#define MAKE_VECTOR_BINARY_OP(opName, ExprType) \
template<typename T, typename L, typename R> \
ExprType<T, L, R> make##ExprType(L left, R right) \
{ \
    ExprType<T, L, R> result; \
    result.left = left; \
    result.right = right; \
    return result; \
} \
namespace __detail { \
template<typename T, typename L, typename R> \
ExprType<T, L, R> opName(L left, R right) \
{ \
    return make##ExprType<T>(left, right); \
} \
} /* namespace __detail */ \
template<typename T> \
typename hlsl::enable_if<hlsl::is_vector<T>::value, typename ExprType<T, Value<T>, Value<T> >::ResultType>::type opName(Value<T> left, Value<T> right) \
{ \
    return __detail::opName<T>(left, right).eval(); \
} \
template<typename T, typename E> \
typename hlsl::enable_if<hlsl::is_vector<E>::value, typename ExprType<T, Value<T>, E >::ResultType>::type \
opName(Value<T> left, E right) \
{ \
    return __detail::opName<T>(left, right).eval(); \
} \
template<typename T, typename E> \
typename hlsl::enable_if<hlsl::is_vector<E>::value, typename ExprType<T, E, Value<T> >::ResultType>::type opName(E left, Value<T> right) \
{ \
    return __detail::opName<T>(left, right).eval(); \
}

MAKE_VECTOR_BINARY_OP(dot, DotExpr)
MAKE_VECTOR_BINARY_OP(cross, CrossExpr)

// Generic unary vector expression factory
#define MAKE_VECTOR_UNARY_OP(opName, ExprType) \
template<typename T, typename E> \
ExprType<T, E> make##ExprType(E expr) \
{ \
    ExprType<T, E> result; \
    result.expr = expr; \
    return result; \
} \
namespace __detail { \
template<typename T, typename E> \
ExprType<T, E> opName(E arg) \
{ \
    return make##ExprType<T>(arg); \
} \
} /* namespace __detail */ \
template<typename T> \
typename hlsl::enable_if<hlsl::is_vector<T>::value, typename ExprType<T, Value<T> >::ResultType>::type opName(Value<T> arg) \
{ \
    return __detail::opName<T>(arg).eval(); \
} \
template<typename T> \
typename hlsl::enable_if<hlsl::is_vector<T>::value, typename ExprType<T, T>::ResultType>::type \
pName(T arg) \
{ \
    return __detail::opName<T>(arg).eval(); \
}

MAKE_VECTOR_UNARY_OP(length, LengthExpr)
MAKE_VECTOR_UNARY_OP(normalize, NormalizeExpr)

// Matrix operations
template<typename T, int N, int K, int M, typename L, typename R>
MatMulExpr<T, N, K, M, L, R> matMul(L left, R right)
{
    MatMulExpr<T, N, K, M, L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

// Special cases with different template parameters
template<typename T, int N, int M, typename E>
TransposeExpr<T, N, M, E> transposeExpr(E expr)
{
    TransposeExpr<T, N, M, E> result;
    result.expr = expr;
    return result;
}

template<typename T, int N, typename E>
DetExpr<T, N, E> determinantExpr(E expr)
{
    DetExpr<T, N, E> result;
    result.expr = expr;
    return result;
}

// ============================================================================
// Component-wise Vector Operations
// ============================================================================

// Component access for vectors (returns scalar Value)
template<typename T, int N>
Value<T> getComponent(Value<vector<T, N> > vec, int index)
{
    return Value<T>::Create(vec.value[index], vec.derivative[index]);
}

// Convenience functions for common components - reduce duplication with macro
#define MAKE_COMPONENT_ACCESSOR(ComponentName, Index) \
template<typename T, int N> \
Value<T> get##ComponentName(Value<vector<T, N> > vec) \
{ \
    return getComponent(vec, Index); \
}

MAKE_COMPONENT_ACCESSOR(X, 0)
MAKE_COMPONENT_ACCESSOR(Y, 1)
MAKE_COMPONENT_ACCESSOR(Z, 2)
MAKE_COMPONENT_ACCESSOR(W, 3)

// Specific vector construction functions
template<typename T>
Value<vector<T, 2> > makeVector(Value<T> x, Value<T> y)
{
    return Value<vector<T, 2> >::Create(vector<T, 2>(x.value, y.value),
                                 vector<T, 2>(x.derivative, y.derivative));
}

template<typename T>
Value<vector<T, 3> > makeVector(Value<T> x, Value<T> y, Value<T> z)
{
    return Value<vector<T, 3> >::Create(vector<T, 3>(x.value, y.value, z.value),
                                 vector<T, 3>(x.derivative, y.derivative, z.derivative));
}

template<typename T>
Value<vector<T, 4> > makeVector(Value<T> x, Value<T> y, Value<T> z, Value<T> w)
{
    return Value<vector<T, 4> >::Create(vector<T, 4>(x.value, y.value, z.value, w.value),
                                 vector<T, 4>(x.derivative, y.derivative, z.derivative, w.derivative));
}

} // namespace fwd
} // namespace ad
#endif // FORWARD_AD_HLSL
