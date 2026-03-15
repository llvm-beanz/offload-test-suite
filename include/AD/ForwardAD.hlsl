#ifndef FORWARD_AD_HLSL
#define FORWARD_AD_HLSL

// Forward Automatic Differentiation for HLSL
// This header provides dual numbers for automatic differentiation with templated types

// ============================================================================
// Forward declarations 
// ============================================================================

// Template dual structure for different numeric types
template<typename T>
struct Dual;

// ============================================================================
// Dual Number Structure - templated for different types
// ============================================================================

template<typename T>
struct Dual
{
    T value;
    T derivative;
    
    // Static creation functions
    static Dual<T> Create(T v, T d)
    {
        Dual<T> result;
        result.value = v;
        result.derivative = d;
        return result;
    }
    
    static Dual<T> CreateValue(T v)
    {
        Dual<T> result;
        result.value = v;
        result.derivative = (T)0;
        return result;
    }
    
    static Dual<T> CreateZero()
    {
        Dual<T> result;
        result.value = (T)0;
        result.derivative = (T)0;
        return result;
    }
    
    // Member operators
    Dual<T> operator+(Dual<T> other)
    {
        return Dual<T>::Create(value + other.value, derivative + other.derivative);
    }
    
    Dual<T> operator+(T other)
    {
        return Dual<T>::Create(value + other, derivative);
    }
    
    Dual<T> operator-(Dual<T> other)
    {
        return Dual<T>::Create(value - other.value, derivative - other.derivative);
    }
    
    Dual<T> operator-(T other)
    {
        return Dual<T>::Create(value - other, derivative);
    }
    
    Dual<T> operator*(Dual<T> other)
    {
        return Dual<T>::Create(value * other.value, 
                       derivative * other.value + value * other.derivative);
    }
    
    Dual<T> operator*(T other)
    {
        return Dual<T>::Create(value * other, derivative * other);
    }
    
    Dual<T> operator/(Dual<T> other)
    {
        T denom = other.value * other.value;
        return Dual<T>::Create(value / other.value,
                       (derivative * other.value - value * other.derivative) / denom);
    }
    
    Dual<T> operator/(T other)
    {
        return Dual<T>::Create(value / other, derivative / other);
    }
    
    Dual<T> operator-()
    {
        return Dual<T>::Create(-value, -derivative);
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
    using ResultType = Dual<T>;
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<T> l_val = getValue(left);
        Dual<T> r_val = getValue(right);
        return Dual<T>::Create(l_val.value + r_val.value, 
                   l_val.derivative + r_val.derivative);
    }
};

// Subtraction Expression
template<typename T, typename L, typename R>
struct SubExpr
{
    using ResultType = Dual<T>;
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<T> l_val = getValue(left);
        Dual<T> r_val = getValue(right);
        return Dual<T>::Create(l_val.value - r_val.value, 
                   l_val.derivative - r_val.derivative);
    }
};

// Multiplication Expression
template<typename T, typename L, typename R>
struct MulExpr
{
    using ResultType = Dual<T>;
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<T> l_val = getValue(left);
        Dual<T> r_val = getValue(right);
        // Product rule: (f*g)' = f'*g + f*g'
        return Dual<T>::Create(l_val.value * r_val.value,
                   l_val.derivative * r_val.value + l_val.value * r_val.derivative);
    }
};

// Division Expression
template<typename T, typename L, typename R>
struct DivExpr
{
    using ResultType = Dual<T>;
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<T> l_val = getValue(left);
        Dual<T> r_val = getValue(right);
        // Quotient rule: (f/g)' = (f'*g - f*g') / g^2
        T denom = r_val.value * r_val.value;
        return Dual<T>::Create(l_val.value / r_val.value,
                   (l_val.derivative * r_val.value - l_val.value * r_val.derivative) / denom);
    }
};

// Power Expression
template<typename T, typename L, typename R>
struct PowExpr
{
    using ResultType = Dual<T>;
    L base;
    R exponent;
    
    Dual<T> eval()
    {
        Dual<T> b_val = getValue(base);
        Dual<T> e_val = getValue(exponent);
        // Power rule: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        T pow_val = pow(b_val.value, e_val.value);
        T deriv = pow_val * (e_val.derivative * log(b_val.value) + 
                            e_val.value * b_val.derivative / b_val.value);
        return Dual<T>::Create(pow_val, deriv);
    }
};

// ============================================================================
// Unary Operation Expression Templates
// ============================================================================

// Negation Expression
template<typename T, typename E>
struct NegExpr
{
    using ResultType = Dual<T>;
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        return Dual<T>::Create(-val.value, -val.derivative);
    }
};

// Sine Expression
template<typename T, typename E>
struct SinExpr
{
    using ResultType = Dual<T>;
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[sin(x)] = cos(x) * x'
        return Dual<T>::Create(sin(val.value), cos(val.value) * val.derivative);
    }
};

// Cosine Expression
template<typename T, typename E>
struct CosExpr
{
    using ResultType = Dual<T>;
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[cos(x)] = -sin(x) * x'
        return Dual<T>::Create(cos(val.value), -sin(val.value) * val.derivative);
    }
};

// Exponential Expression
template<typename T, typename E>
struct ExpExpr
{
    using ResultType = Dual<T>;
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[exp(x)] = exp(x) * x'
        T exp_val = exp(val.value);
        return Dual<T>::Create(exp_val, exp_val * val.derivative);
    }
};

// Natural Logarithm Expression
template<typename T, typename E>
struct LogExpr
{
    using ResultType = Dual<T>;
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[log(x)] = x' / x
        return Dual<T>::Create(log(val.value), val.derivative / val.value);
    }
};

// Square Root Expression
template<typename T, typename E>
struct SqrtExpr
{
    using ResultType = Dual<T>;
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[sqrt(x)] = x' / (2 * sqrt(x))
        T sqrt_val = sqrt(val.value);
        return Dual<T>::Create(sqrt_val, val.derivative / ((T)2 * sqrt_val));
    }
};

// ============================================================================
// Helper Functions for Value Extraction
// ============================================================================

// Extract Dual from templated expression types
template <typename T>
typename T::ResultType getValue(T value)
{
    return value.eval();
}


// Extract Dual from Dual (identity)
template<typename T>
Dual<T> getValue(Dual<T> d)
{
    return d;
}

// Removed generic getValue to avoid ambiguity with expression specializations

// Helper functions to create templated expression templates
template<typename T, typename L, typename R>
AddExpr<T, L, R> makeAddExpr(L left, R right)
{
    AddExpr<T, L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename T, typename L, typename R>
SubExpr<T, L, R> makeSubExpr(L left, R right)
{
    SubExpr<T, L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename T, typename L, typename R>
MulExpr<T, L, R> makeMulExpr(L left, R right)
{
    MulExpr<T, L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename T, typename L, typename R>
DivExpr<T, L, R> makeDivExpr(L left, R right)
{
    DivExpr<T, L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename T, typename L, typename R>
PowExpr<T, L, R> makePowExpr(L base, R exponent)
{
    PowExpr<T, L, R> result;
    result.base = base;
    result.exponent = exponent;
    return result;
}

template<typename T, typename E>
NegExpr<T, E> makeNegExpr(E expr)
{
    NegExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
SinExpr<T, E> makeSinExpr(E expr)
{
    SinExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
CosExpr<T, E> makeCosExpr(E expr)
{
    CosExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
ExpExpr<T, E> makeExpExpr(E expr)
{
    ExpExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
LogExpr<T, E> makeLogExpr(E expr)
{
    LogExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
SqrtExpr<T, E> makeSqrtExpr(E expr)
{
    SqrtExpr<T, E> result;
    result.expr = expr;
    return result;
}

// ============================================================================
// Named Functions for Operations with Type Support
// ============================================================================

// Templated operation functions
template<typename T, typename L, typename R>
AddExpr<T, L, R> add(L left, R right)
{
    return makeAddExpr<T>(left, right);
}

template<typename T, typename L, typename R>
SubExpr<T, L, R> subtract(L left, R right)
{
    return makeSubExpr<T>(left, right);
}

template<typename T, typename L, typename R>
MulExpr<T, L, R> multiply(L left, R right)
{
    return makeMulExpr<T>(left, right);
}

template<typename T, typename L, typename R>
DivExpr<T, L, R> divide(L left, R right)
{
    return makeDivExpr<T>(left, right);
}

template<typename T, typename L, typename R>
PowExpr<T, L, R> power(L base, R exponent)
{
    return makePowExpr<T>(base, exponent);
}

template<typename T, typename E>
NegExpr<T, E> negate(E expr)
{
    return makeNegExpr<T>(expr);
}

// Mathematical Function Templates
template<typename T, typename E>
SinExpr<T, E> sinExpr(E expr)
{
    return makeSinExpr<T>(expr);
}

template<typename T>
Dual<T> sinDual(Dual<T> d)
{
    return Dual<T>::Create(sin(d.value), cos(d.value) * d.derivative);
}

// Additional math functions with templates
template<typename T, typename E>
CosExpr<T, E> cosExpr(E expr)
{
    return makeCosExpr<T>(expr);
}

template<typename T>
Dual<T> cosDual(Dual<T> d)
{
    return Dual<T>::Create(cos(d.value), -sin(d.value) * d.derivative);
}

template<typename T, typename E>
ExpExpr<T, E> expExpr(E expr)
{
    return makeExpExpr<T>(expr);
}

template<typename T>
Dual<T> expDual(Dual<T> d)
{
    T exp_val = exp(d.value);
    return Dual<T>::Create(exp_val, exp_val * d.derivative);
}

template<typename T, typename E>
LogExpr<T, E> logExpr(E expr)
{
    return makeLogExpr<T>(expr);
}

template<typename T>
Dual<T> logDual(Dual<T> d)
{
    return Dual<T>::Create(log(d.value), d.derivative / d.value);
}

template<typename T, typename E>
SqrtExpr<T, E> sqrtExpr(E expr)
{
    return makeSqrtExpr<T>(expr);
}

template<typename T>
Dual<T> sqrtDual(Dual<T> d)
{
    T sqrt_val = sqrt(d.value);
    return Dual<T>::Create(sqrt_val, d.derivative / ((T)2 * sqrt_val));
}

// ============================================================================
// Utility Functions
// ============================================================================

template<typename T>
Dual<T> variable(T value)
{
    return Dual<T>::Create(value, (T)1);
}

// Scalar constant function  
template<typename T>
Dual<T> constant(T value)
{
    return Dual<T>::Create(value, (T)0);
}

// ============================================================================
// Matrix and Vector Dual Number Support  
// ============================================================================

// Vector dual initialization functions using splat casting
template<typename T, int N>
Dual<vector<T, N> > variable(vector<T, N> value)
{
    return Dual<vector<T, N> >::Create(value, (vector<T, N>)1); // Splat 1 to all components
}

// Matrix dual initialization functions using splat casting
template<typename T, int N, int M>
Dual<matrix<T, N, M> > variable(matrix<T, N, M> value)
{
    return Dual<matrix<T, N, M> >::Create(value, (matrix<T, N, M>)1); // Splat 1 to all components
}

// Specialized constant functions for vector and matrix types
template<typename T, int N>
Dual<vector<T, N> > constantVector(vector<T, N> value)
{
    return Dual<vector<T, N> >::Create(value, (vector<T, N>)0); // Splat 0 to all components
}

template<typename T, int N, int M>
Dual<matrix<T, N, M> > constantMatrix(matrix<T, N, M> value)
{
    return Dual<matrix<T, N, M> >::Create(value, (matrix<T, N, M>)0); // Splat 0 to all components
}

// ============================================================================
// Vector Operations
// ============================================================================

// Dot Product Expression
template<typename T, int N, typename L, typename R>
struct DotExpr
{
    using ResultType = Dual<T>;
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<vector<T, N> > l_val = getValue(left);
        Dual<vector<T, N> > r_val = getValue(right);
        
        // Dot product: d/dx[dot(u,v)] = dot(u',v) + dot(u,v')
        T val = dot(l_val.value, r_val.value);
        T deriv = dot(l_val.derivative, r_val.value) + dot(l_val.value, r_val.derivative);
        
        return Dual<T>::Create(val, deriv);
    }
};

// Cross Product Expression (for 3D vectors only)
template<typename T, typename L, typename R>
struct CrossExpr
{
    using ResultType = Dual<vector<T, 3> >;
    L left;
    R right;
    
    Dual<vector<T, 3> > eval()
    {
        Dual<vector<T, 3> > l_val = getValue(left);
        Dual<vector<T, 3> > r_val = getValue(right);
        
        // Cross product: d/dx[cross(u,v)] = cross(u',v) + cross(u,v')
        vector<T, 3> val = cross(l_val.value, r_val.value);
        vector<T, 3> deriv = cross(l_val.derivative, r_val.value) + cross(l_val.value, r_val.derivative);
        
        return Dual<vector<T, 3> >::Create(val, deriv);
    }
};

// Vector Length Expression
template<typename T, int N, typename E>
struct LengthExpr
{
    using ResultType = Dual<T>;
    E expr;
    
    Dual<T> eval()
    {
        Dual<vector<T, N> > val = getValue(expr);
        
        // Length: d/dx[|v|] = dot(v, v') / |v|
        T len = length(val.value);
        T deriv = dot(val.value, val.derivative) / len;
        
        return Dual<T>::Create(len, deriv);
    }
};

// Vector Normalize Expression
template<typename T, int N, typename E>
struct NormalizeExpr
{
    using ResultType = Dual<vector<T, N> >;
    E expr;
    
    Dual<vector<T, N> > eval()
    {
        Dual<vector<T, N> > val = getValue(expr);
        
        // Normalize: d/dx[normalize(v)] = (v' * |v| - v * (dot(v,v') / |v|)) / |v|²
        T len = length(val.value);
        vector<T, N> norm_val = normalize(val.value);
        T dot_deriv = dot(val.value, val.derivative);
        vector<T, N> deriv = (val.derivative * len - val.value * (dot_deriv / len)) / (len * len);
        
        return Dual<vector<T, N> >::Create(norm_val, deriv);
    }
};

// ============================================================================
// Matrix Operations
// ============================================================================

// Matrix Multiplication Expression
template<typename T, int N, int K, int M, typename L, typename R>
struct MatMulExpr
{
    using ResultType = Dual<vector<T, N> >;
    L left;   // Matrix NxK
    R right;  // Matrix KxM or Vector K
    
    // Returns either Dual<matrix<T,N,M> > for matrix*matrix or Dual<vector<T,N> > for matrix*vector
    // We'll use a specific implementation that works with common cases
    Dual<vector<T, N> > eval()  // Assuming matrix-vector multiplication for now
    {
        Dual<matrix<T, N, K> > l_val = getValue(left);
        Dual<vector<T, K> > r_val = getValue(right);
        
        // Matrix-vector multiplication: d/dx[A*v] = A'*v + A*v'
        vector<T, N> val = mul(l_val.value, r_val.value);
        vector<T, N> deriv = mul(l_val.derivative, r_val.value) + mul(l_val.value, r_val.derivative);
        
        return Dual<vector<T, N> >::Create(val, deriv);
    }
};

// Matrix Transpose Expression
template<typename T, int N, int M, typename E>
struct TransposeExpr
{
    using ResultType = Dual<matrix<T, M, N> >;
    E expr;
    
    Dual<matrix<T, M, N> > eval()  // Transpose flips dimensions
    {
        Dual<matrix<T, N, M> > val = getValue(expr);
        
        // Transpose: d/dx[transpose(M)] = transpose(M')
        matrix<T, M, N> val_t = transpose(val.value);
        matrix<T, M, N> deriv_t = transpose(val.derivative);
        
        return Dual<matrix<T, M, N> >::Create(val_t, deriv_t);
    }
};

// Matrix Determinant Expression (generic for square matrices)
template<typename T, int N, typename E>
struct DetExpr
{
    using ResultType = Dual<T>;
    E expr;
    
    Dual<T> eval()
    {
        Dual<matrix<T, N, N> > val = getValue(expr);
        
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
        
        return Dual<T>::Create(det_val, deriv);
    }
};

// ============================================================================
// Vector and Matrix getValue Specializations
// ============================================================================


// ============================================================================
// Vector and Matrix Operation Functions
// ============================================================================

// Vector operations
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

template<typename T, int N, typename E>
LengthExpr<T, N, E> lengthExpr(E expr)
{
    LengthExpr<T, N, E> result;
    result.expr = expr;
    return result;
}

template<typename T, int N, typename E>
NormalizeExpr<T, N, E> normalizeExpr(E expr)
{
    NormalizeExpr<T, N, E> result;
    result.expr = expr;
    return result;
}

// Matrix operations
template<typename T, int N, int K, int M, typename L, typename R>
MatMulExpr<T, N, K, M, L, R> matMul(L left, R right)
{
    MatMulExpr<T, N, K, M, L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

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

// Component access for vectors (returns scalar dual) 
template<typename T, int N>
Dual<T> getComponent(Dual<vector<T, N> > vec, int index)
{
    return Dual<T>::Create(vec.value[index], vec.derivative[index]);
}

// Convenience functions for common components
template<typename T, int N>
Dual<T> getX(Dual<vector<T, N> > vec)
{
    return getComponent(vec, 0);
}

template<typename T, int N>
Dual<T> getY(Dual<vector<T, N> > vec)
{
    return getComponent(vec, 1);
}

template<typename T, int N>
Dual<T> getZ(Dual<vector<T, N> > vec)
{
    return getComponent(vec, 2);
}

template<typename T, int N>
Dual<T> getW(Dual<vector<T, N> > vec)
{
    return getComponent(vec, 3);
}

// Specific vector construction functions
template<typename T>
Dual<vector<T, 2> > makeVector2(Dual<T> x, Dual<T> y)
{
    return Dual<vector<T, 2> >::Create(vector<T, 2>(x.value, y.value), 
                                 vector<T, 2>(x.derivative, y.derivative));
}

template<typename T>
Dual<vector<T, 3> > makeVector3(Dual<T> x, Dual<T> y, Dual<T> z)
{
    return Dual<vector<T, 3> >::Create(vector<T, 3>(x.value, y.value, z.value), 
                                 vector<T, 3>(x.derivative, y.derivative, z.derivative));
}

template<typename T>
Dual<vector<T, 4> > makeVector4(Dual<T> x, Dual<T> y, Dual<T> z, Dual<T> w)
{
    return Dual<vector<T, 4> >::Create(vector<T, 4>(x.value, y.value, z.value, w.value), 
                                 vector<T, 4>(x.derivative, y.derivative, z.derivative, w.derivative));
}

#endif // FORWARD_AD_HLSL