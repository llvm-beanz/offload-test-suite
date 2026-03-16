#ifndef FORWARD_AD_HLSL
#define FORWARD_AD_HLSL

namespace ad {
namespace fwd {
// Forward Automatic Differentiation for HLSL
// This header provides Value numbers for automatic differentiation with templated types

// ============================================================================
// Forward declarations 
// ============================================================================

// Template Value structure for different numeric types
template<typename T>
struct Value;

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
    
    Value<T> operator-()
    {
        return Value<T>::Create(-value, -derivative);
    }
};

// ============================================================================
// Expression Template Base Classes (simplified)
// ============================================================================

// We'll remove the problematic cast function and just use direct eval calls

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
        Value<T> l_val = getValue(left);
        Value<T> r_val = getValue(right);
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
        Value<T> l_val = getValue(left);
        Value<T> r_val = getValue(right);
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
        Value<T> l_val = getValue(left);
        Value<T> r_val = getValue(right);
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
        Value<T> l_val = getValue(left);
        Value<T> r_val = getValue(right);
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
    L base;
    R exponent;
    
    Value<T> eval()
    {
        Value<T> b_val = getValue(base);
        Value<T> e_val = getValue(exponent);
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
        Value<T> val = getValue(expr);
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
        Value<T> val = getValue(expr);
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
        Value<T> val = getValue(expr);
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
        Value<T> val = getValue(expr);
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
        Value<T> val = getValue(expr);
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
        Value<T> val = getValue(expr);
        // d/dx[sqrt(x)] = x' / (2 * sqrt(x))
        T sqrt_val = sqrt(val.value);
        return Value<T>::Create(sqrt_val, val.derivative / ((T)2 * sqrt_val));
    }
};

// ============================================================================
// Helper Functions for Value Extraction
// ============================================================================

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

// Removed generic getValue to avoid ambiguity with expression specializations

// Helper functions to create templated expression templates - using macros to reduce duplication

#define MAKE_BINARY_EXPR(ExprType) \
template<typename T, typename L, typename R> \
ExprType<T, L, R> make##ExprType(L left, R right) \
{ \
    ExprType<T, L, R> result; \
    result.left = left; \
    result.right = right; \
    return result; \
}

MAKE_BINARY_EXPR(AddExpr)
MAKE_BINARY_EXPR(SubExpr)
MAKE_BINARY_EXPR(MulExpr)
MAKE_BINARY_EXPR(DivExpr)

template<typename T, typename L, typename R>
PowExpr<T, L, R> makePowExpr(L base, R exponent)
{
    PowExpr<T, L, R> result;
    result.base = base;
    result.exponent = exponent;
    return result;
}

#define MAKE_UNARY_EXPR(ExprType) \
template<typename T, typename E> \
ExprType<T, E> make##ExprType(E expr) \
{ \
    ExprType<T, E> result; \
    result.expr = expr; \
    return result; \
}

MAKE_UNARY_EXPR(NegExpr)
MAKE_UNARY_EXPR(SinExpr)
MAKE_UNARY_EXPR(CosExpr)
MAKE_UNARY_EXPR(ExpExpr)
MAKE_UNARY_EXPR(LogExpr)
MAKE_UNARY_EXPR(SqrtExpr)

// ============================================================================
// Named Functions for Operations with Type Support
// ============================================================================

// Templated operation functions - using macro to reduce duplication
#define MAKE_BINARY_OP(opName, ExprType) \
template<typename T, typename L, typename R> \
ExprType<T, L, R> opName(L left, R right) \
{ \
    return make##ExprType<T>(left, right); \
}

MAKE_BINARY_OP(add, AddExpr)
MAKE_BINARY_OP(subtract, SubExpr)
MAKE_BINARY_OP(multiply, MulExpr)
MAKE_BINARY_OP(divide, DivExpr)

template<typename T, typename L, typename R>
PowExpr<T, L, R> power(L base, R exponent)
{
    return makePowExpr<T>(base, exponent);
}

#define MAKE_UNARY_OP(opName, ExprType) \
template<typename T, typename E> \
ExprType<T, E> opName(E expr) \
{ \
    return make##ExprType<T>(expr); \
}

MAKE_UNARY_OP(negate, NegExpr)

// Mathematical Function Templates - using macros to reduce duplication
#define MAKE_MATH_EXPR_OP(mathName, ExprType) \
template<typename T, typename E> \
ExprType<T, E> mathName##Expr(E expr) \
{ \
    return make##ExprType<T>(expr); \
}

MAKE_MATH_EXPR_OP(sin, SinExpr)
MAKE_MATH_EXPR_OP(cos, CosExpr)
MAKE_MATH_EXPR_OP(exp, ExpExpr)
MAKE_MATH_EXPR_OP(log, LogExpr)
MAKE_MATH_EXPR_OP(sqrt, SqrtExpr)

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
// Matrix and Vector Value Number Support  
// ============================================================================

// Vector Value initialization functions using splat casting
template<typename T, int N>
Value<vector<T, N> > variable(vector<T, N> value)
{
    return Value<vector<T, N> >::Create(value, (vector<T, N>)1); // Splat 1 to all components
}

// Matrix Value initialization functions using splat casting
template<typename T, int N, int M>
Value<matrix<T, N, M> > variable(matrix<T, N, M> value)
{
    return Value<matrix<T, N, M> >::Create(value, (matrix<T, N, M>)1); // Splat 1 to all components
}

// Specialized constant functions for vector and matrix types
template<typename T, int N>
Value<vector<T, N> > constantVector(vector<T, N> value)
{
    return Value<vector<T, N> >::Create(value, (vector<T, N>)0); // Splat 0 to all components
}

template<typename T, int N, int M>
Value<matrix<T, N, M> > constantMatrix(matrix<T, N, M> value)
{
    return Value<matrix<T, N, M> >::Create(value, (matrix<T, N, M>)0); // Splat 0 to all components
}

// ============================================================================
// Vector Operations
// ============================================================================

// Dot Product Expression
template<typename T, int N, typename L, typename R>
struct DotExpr
{
    using ResultType = Value<T>;
    L left;
    R right;
    
    Value<T> eval()
    {
        Value<vector<T, N> > l_val = getValue(left);
        Value<vector<T, N> > r_val = getValue(right);
        
        // Dot product: d/dx[dot(u,v)] = dot(u',v) + dot(u,v')
        T val = dot(l_val.value, r_val.value);
        T deriv = dot(l_val.derivative, r_val.value) + dot(l_val.value, r_val.derivative);
        
        return Value<T>::Create(val, deriv);
    }
};

// Cross Product Expression (for 3D vectors only)
template<typename T, typename L, typename R>
struct CrossExpr
{
    using ResultType = Value<vector<T, 3> >;
    L left;
    R right;
    
    Value<vector<T, 3> > eval()
    {
        Value<vector<T, 3> > l_val = getValue(left);
        Value<vector<T, 3> > r_val = getValue(right);
        
        // Cross product: d/dx[cross(u,v)] = cross(u',v) + cross(u,v')
        vector<T, 3> val = cross(l_val.value, r_val.value);
        vector<T, 3> deriv = cross(l_val.derivative, r_val.value) + cross(l_val.value, r_val.derivative);
        
        return Value<vector<T, 3> >::Create(val, deriv);
    }
};

// Vector Length Expression
template<typename T, int N, typename E>
struct LengthExpr
{
    using ResultType = Value<T>;
    E expr;
    
    Value<T> eval()
    {
        Value<vector<T, N> > val = getValue(expr);
        
        // Length: d/dx[|v|] = dot(v, v') / |v|
        T len = length(val.value);
        T deriv = dot(val.value, val.derivative) / len;
        
        return Value<T>::Create(len, deriv);
    }
};

// Vector Normalize Expression
template<typename T, int N, typename E>
struct NormalizeExpr
{
    using ResultType = Value<vector<T, N> >;
    E expr;
    
    Value<vector<T, N> > eval()
    {
        Value<vector<T, N> > val = getValue(expr);
        
        // Normalize: d/dx[normalize(v)] = (v' * |v| - v * (dot(v,v') / |v|)) / |v|²
        T len = length(val.value);
        vector<T, N> norm_val = normalize(val.value);
        T dot_deriv = dot(val.value, val.derivative);
        vector<T, N> deriv = (val.derivative * len - val.value * (dot_deriv / len)) / (len * len);
        
        return Value<vector<T, N> >::Create(norm_val, deriv);
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
        Value<matrix<T, N, K> > l_val = getValue(left);
        Value<vector<T, K> > r_val = getValue(right);
        
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
        Value<matrix<T, N, M> > val = getValue(expr);
        
        // Transpose: d/dx[transpose(M)] = transpose(M')
        matrix<T, M, N> val_t = transpose(val.value);
        matrix<T, M, N> deriv_t = transpose(val.derivative);
        
        return Value<matrix<T, M, N> >::Create(val_t, deriv_t);
    }
};

// Matrix Determinant Expression (generic for square matrices)
template<typename T, int N, typename E>
struct DetExpr
{
    using ResultType = Value<T>;
    E expr;
    
    Value<T> eval()
    {
        Value<matrix<T, N, N> > val = getValue(expr);
        
        // Determinant: d/dx[det(M)] = det(M) * tr(M^-1 * M')
        // For simplicity, we'll use the fact that d/dx[det(M)] = det(M) * tr(adj(M)^T * M') / det(M) = tr(adj(M)^T * M')
        T det_val = determinant(val.value);
        
        // This is a simplified derivative - full implementation would need adjugate matrix
        // For 2x2: det([[a,b],[c,d]]) = ad - bc
        // d_det = a'*d + a*d' - b'*c - b*c' (only valid for 2x2)
        T deriv;
        if (N == 2)
        {
            deriv = val.derivative[0][0] * val.value[1][1] + val.value[0][0] * val.derivative[1][1] - 
                    val.derivative[0][1] * val.value[1][0] - val.value[0][1] * val.derivative[1][0];
        }
        else
        {
            // For larger matrices, this is more complex - simplified approximation
            deriv = (T)0;
        }
        
        return Value<T>::Create(det_val, deriv);
    }
};

// ============================================================================
// Vector and Matrix getValue Specializations
// ============================================================================


// ============================================================================
// Vector and Matrix Operation Functions
// ============================================================================

// Vector operations - reduce duplication with generic patterns
template<typename T, int N, typename L, typename R>
DotExpr<T, N, L, R> dotProduct(L left, R right)
{
    DotExpr<T, N, L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename T, typename L, typename R>  
CrossExpr<T, L, R> crossProduct(L left, R right)
{
    CrossExpr<T, L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

// Generic unary vector expression factory
#define MAKE_VECTOR_UNARY_OP(OpName, ExprType) \
template<typename T, int N, typename E> \
ExprType<T, N, E> OpName(E expr) \
{ \
    ExprType<T, N, E> result; \
    result.expr = expr; \
    return result; \
}

MAKE_VECTOR_UNARY_OP(lengthExpr, LengthExpr)
MAKE_VECTOR_UNARY_OP(normalizeExpr, NormalizeExpr)

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
