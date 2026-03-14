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

// Initialization functions for templated types
template<typename T> Dual<T> makeDual(T v, T d);
template<typename T> Dual<T> makeDualValue(T v);
template<typename T> Dual<T> makeDualZero();

// ============================================================================
// Dual Number Structure - templated for different types
// ============================================================================

template<typename T>
struct Dual
{
    T value;
    T derivative;
    
    // Member operators
    Dual<T> operator+(Dual<T> other)
    {
        return makeDual<T>(value + other.value, derivative + other.derivative);
    }
    
    Dual<T> operator+(T other)
    {
        return makeDual<T>(value + other, derivative);
    }
    
    Dual<T> operator-(Dual<T> other)
    {
        return makeDual<T>(value - other.value, derivative - other.derivative);
    }
    
    Dual<T> operator-(T other)
    {
        return makeDual<T>(value - other, derivative);
    }
    
    Dual<T> operator*(Dual<T> other)
    {
        return makeDual<T>(value * other.value, 
                       derivative * other.value + value * other.derivative);
    }
    
    Dual<T> operator*(T other)
    {
        return makeDual<T>(value * other, derivative * other);
    }
    
    Dual<T> operator/(Dual<T> other)
    {
        T denom = other.value * other.value;
        return makeDual<T>(value / other.value,
                       (derivative * other.value - value * other.derivative) / denom);
    }
    
    Dual<T> operator/(T other)
    {
        return makeDual<T>(value / other, derivative / other);
    }
    
    Dual<T> operator-()
    {
        return makeDual<T>(-value, -derivative);
    }
};

// Initialization functions (HLSL doesn't have constructors)
template<typename T>
Dual<T> makeDual(T v, T d)
{
    Dual<T> result;
    result.value = v;
    result.derivative = d;
    return result;
}

template<typename T>
Dual<T> makeDualValue(T v)
{
    Dual<T> result;
    result.value = v;
    result.derivative = T(0);
    return result;
}

template<typename T>
Dual<T> makeDualZero()
{
    Dual<T> result;
    result.value = T(0);
    result.derivative = T(0);
    return result;
}

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
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<T> l_val = getValue(left);
        Dual<T> r_val = getValue(right);
        return makeDual(l_val.value + r_val.value, 
                   l_val.derivative + r_val.derivative);
    }
};

// Subtraction Expression
template<typename T, typename L, typename R>
struct SubExpr
{
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<T> l_val = getValue(left);
        Dual<T> r_val = getValue(right);
        return makeDual(l_val.value - r_val.value, 
                   l_val.derivative - r_val.derivative);
    }
};

// Multiplication Expression
template<typename T, typename L, typename R>
struct MulExpr
{
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<T> l_val = getValue(left);
        Dual<T> r_val = getValue(right);
        // Product rule: (f*g)' = f'*g + f*g'
        return makeDual(l_val.value * r_val.value,
                   l_val.derivative * r_val.value + l_val.value * r_val.derivative);
    }
};

// Division Expression
template<typename T, typename L, typename R>
struct DivExpr
{
    L left;
    R right;
    
    Dual<T> eval()
    {
        Dual<T> l_val = getValue(left);
        Dual<T> r_val = getValue(right);
        // Quotient rule: (f/g)' = (f'*g - f*g') / g^2
        T denom = r_val.value * r_val.value;
        return makeDual<T>(l_val.value / r_val.value,
                   (l_val.derivative * r_val.value - l_val.value * r_val.derivative) / denom);
    }
};

// Power Expression
template<typename T, typename L, typename R>
struct PowExpr
{
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
        return makeDual<T>(pow_val, deriv);
    }
};

// ============================================================================
// Unary Operation Expression Templates
// ============================================================================

// Negation Expression
template<typename T, typename E>
struct NegExpr
{
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        return makeDual<T>(-val.value, -val.derivative);
    }
};

// Sine Expression
template<typename T, typename E>
struct SinExpr
{
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[sin(x)] = cos(x) * x'
        return makeDual<T>(sin(val.value), cos(val.value) * val.derivative);
    }
};

// Cosine Expression
template<typename T, typename E>
struct CosExpr
{
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[cos(x)] = -sin(x) * x'
        return makeDual<T>(cos(val.value), -sin(val.value) * val.derivative);
    }
};

// Exponential Expression
template<typename T, typename E>
struct ExpExpr
{
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[exp(x)] = exp(x) * x'
        T exp_val = exp(val.value);
        return makeDual<T>(exp_val, exp_val * val.derivative);
    }
};

// Natural Logarithm Expression
template<typename T, typename E>
struct LogExpr
{
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[log(x)] = x' / x
        return makeDual<T>(log(val.value), val.derivative / val.value);
    }
};

// Square Root Expression
template<typename T, typename E>
struct SqrtExpr
{
    E expr;
    
    Dual<T> eval()
    {
        Dual<T> val = getValue(expr);
        // d/dx[sqrt(x)] = x' / (2 * sqrt(x))
        T sqrt_val = sqrt(val.value);
        return makeDual<T>(sqrt_val, val.derivative / (T(2) * sqrt_val));
    }
};

// ============================================================================
// Helper Functions for Value Extraction
// ============================================================================

// Extract Dual from templated expression types
template<typename T, typename L, typename R>
Dual<T> getValue(AddExpr<T, L, R> expr)
{
    return expr.eval();
}

template<typename T, typename L, typename R>
Dual<T> getValue(SubExpr<T, L, R> expr)
{
    return expr.eval();
}

template<typename T, typename L, typename R>
Dual<T> getValue(MulExpr<T, L, R> expr)
{
    return expr.eval();
}

template<typename T, typename L, typename R>
Dual<T> getValue(DivExpr<T, L, R> expr)
{
    return expr.eval();
}

template<typename T, typename L, typename R>
Dual<T> getValue(PowExpr<T, L, R> expr)
{
    return expr.eval();
}

template<typename T, typename E>
Dual<T> getValue(NegExpr<T, E> expr)
{
    return expr.eval();
}

template<typename T, typename E>
Dual<T> getValue(SinExpr<T, E> expr)
{
    return expr.eval();
}

template<typename T, typename E>
Dual<T> getValue(CosExpr<T, E> expr)
{
    return expr.eval();
}

template<typename T, typename E>
Dual<T> getValue(ExpExpr<T, E> expr)
{
    return expr.eval();
}

template<typename T, typename E>
Dual<T> getValue(LogExpr<T, E> expr)
{
    return expr.eval();
}

template<typename T, typename E>
Dual<T> getValue(SqrtExpr<T, E> expr)
{
    return expr.eval();
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
    return makeDual<T>(sin(d.value), cos(d.value) * d.derivative);
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
    return makeDual<T>(cos(d.value), -sin(d.value) * d.derivative);
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
    return makeDual<T>(exp_val, exp_val * d.derivative);
}

template<typename T, typename E>
LogExpr<T, E> logExpr(E expr)
{
    return makeLogExpr<T>(expr);
}

template<typename T>
Dual<T> logDual(Dual<T> d)
{
    return makeDual<T>(log(d.value), d.derivative / d.value);
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
    return makeDual<T>(sqrt_val, d.derivative / (T(2) * sqrt_val));
}

// ============================================================================
// Utility Functions
// ============================================================================

template<typename T>
Dual<T> variable(T value)
{
    return makeDual<T>(value, T(1));
}

template<typename T>
Dual<T> constant(T value)
{
    return makeDual<T>(value, T(0));
}

#endif // FORWARD_AD_HLSL