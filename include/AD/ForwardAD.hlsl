#ifndef FORWARD_AD_HLSL
#define FORWARD_AD_HLSL

// Forward Automatic Differentiation for HLSL
// This header provides dual numbers for automatic differentiation

// ============================================================================
// Forward declarations and helper functions
// ============================================================================
struct Dual;

// Initialization functions (defined first since they're used in Dual struct)
Dual makeDual(float v, float d);
Dual makeDualValue(float v);
Dual makeDualZero();

// ============================================================================
// Dual Number Structure - holds value and derivative
// ============================================================================
struct Dual
{
    float value;
    float derivative;
    
    // Member operators
    Dual operator+(Dual other)
    {
        return makeDual(value + other.value, derivative + other.derivative);
    }
    
    Dual operator+(float other)
    {
        return makeDual(value + other, derivative);
    }
    
    Dual operator-(Dual other)
    {
        return makeDual(value - other.value, derivative - other.derivative);
    }
    
    Dual operator-(float other)
    {
        return makeDual(value - other, derivative);
    }
    
    Dual operator*(Dual other)
    {
        return makeDual(value * other.value, 
                       derivative * other.value + value * other.derivative);
    }
    
    Dual operator*(float other)
    {
        return makeDual(value * other, derivative * other);
    }
    
    Dual operator/(Dual other)
    {
        float denom = other.value * other.value;
        return makeDual(value / other.value,
                       (derivative * other.value - value * other.derivative) / denom);
    }
    
    Dual operator/(float other)
    {
        return makeDual(value / other, derivative / other);
    }
    
    Dual operator-()
    {
        return makeDual(-value, -derivative);
    }
};

// Initialization functions (HLSL doesn't have constructors)
Dual makeDual(float v, float d)
{
    Dual result;
    result.value = v;
    result.derivative = d;
    return result;
}

Dual makeDualValue(float v)
{
    Dual result;
    result.value = v;
    result.derivative = 0.0f;
    return result;
}

Dual makeDualZero()
{
    Dual result;
    result.value = 0.0f;
    result.derivative = 0.0f;
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
template<typename L, typename R>
struct AddExpr
{
    L left;
    R right;
    
    Dual eval()
    {
        Dual l_val = getValue(left);
        Dual r_val = getValue(right);
        return makeDual(l_val.value + r_val.value, 
                   l_val.derivative + r_val.derivative);
    }
};

// Subtraction Expression
template<typename L, typename R>
struct SubExpr
{
    L left;
    R right;
    
    Dual eval()
    {
        Dual l_val = getValue(left);
        Dual r_val = getValue(right);
        return makeDual(l_val.value - r_val.value, 
                   l_val.derivative - r_val.derivative);
    }
};

// Multiplication Expression
template<typename L, typename R>
struct MulExpr
{
    L left;
    R right;
    
    Dual eval()
    {
        Dual l_val = getValue(left);
        Dual r_val = getValue(right);
        // Product rule: (f*g)' = f'*g + f*g'
        return makeDual(l_val.value * r_val.value,
                   l_val.derivative * r_val.value + l_val.value * r_val.derivative);
    }
};

// Division Expression
template<typename L, typename R>
struct DivExpr
{
    L left;
    R right;
    
    Dual eval()
    {
        Dual l_val = getValue(left);
        Dual r_val = getValue(right);
        // Quotient rule: (f/g)' = (f'*g - f*g') / g^2
        float denom = r_val.value * r_val.value;
        return makeDual(l_val.value / r_val.value,
                   (l_val.derivative * r_val.value - l_val.value * r_val.derivative) / denom);
    }
};

// Power Expression
template<typename L, typename R>
struct PowExpr
{
    L base;
    R exponent;
    
    Dual eval()
    {
        Dual b_val = getValue(base);
        Dual e_val = getValue(exponent);
        // Power rule: (f^g)' = f^g * (g' * ln(f) + g * f'/f)
        float pow_val = pow(b_val.value, e_val.value);
        float deriv = pow_val * (e_val.derivative * log(b_val.value) + 
                                e_val.value * b_val.derivative / b_val.value);
        return makeDual(pow_val, deriv);
    }
};

// ============================================================================
// Unary Operation Expression Templates
// ============================================================================

// Negation Expression
template<typename E>
struct NegExpr
{
    E expr;
    
    Dual eval()
    {
        Dual val = getValue(expr);
        return makeDual(-val.value, -val.derivative);
    }
};

// Sine Expression
template<typename E>
struct SinExpr
{
    E expr;
    
    Dual eval()
    {
        Dual val = getValue(expr);
        // d/dx[sin(x)] = cos(x) * x'
        return makeDual(sin(val.value), cos(val.value) * val.derivative);
    }
};

// Cosine Expression
template<typename E>
struct CosExpr
{
    E expr;
    
    Dual eval()
    {
        Dual val = getValue(expr);
        // d/dx[cos(x)] = -sin(x) * x'
        return makeDual(cos(val.value), -sin(val.value) * val.derivative);
    }
};

// Exponential Expression
template<typename E>
struct ExpExpr
{
    E expr;
    
    Dual eval()
    {
        Dual val = getValue(expr);
        // d/dx[exp(x)] = exp(x) * x'
        float exp_val = exp(val.value);
        return makeDual(exp_val, exp_val * val.derivative);
    }
};

// Natural Logarithm Expression
template<typename E>
struct LogExpr
{
    E expr;
    
    Dual eval()
    {
        Dual val = getValue(expr);
        // d/dx[log(x)] = x' / x
        return makeDual(log(val.value), val.derivative / val.value);
    }
};

// Square Root Expression
template<typename E>
struct SqrtExpr
{
    E expr;
    
    Dual eval()
    {
        Dual val = getValue(expr);
        // d/dx[sqrt(x)] = x' / (2 * sqrt(x))
        float sqrt_val = sqrt(val.value);
        return makeDual(sqrt_val, val.derivative / (2.0f * sqrt_val));
    }
};

// ============================================================================
// Helper Functions for Value Extraction
// ============================================================================

// Extract Dual from Expression
// Extract Dual from expression types
template<typename L, typename R>
Dual getValue(AddExpr<L, R> expr)
{
    return expr.eval();
}

template<typename L, typename R>
Dual getValue(SubExpr<L, R> expr)
{
    return expr.eval();
}

template<typename L, typename R>
Dual getValue(MulExpr<L, R> expr)
{
    return expr.eval();
}

template<typename L, typename R>
Dual getValue(DivExpr<L, R> expr)
{
    return expr.eval();
}

template<typename L, typename R>
Dual getValue(PowExpr<L, R> expr)
{
    return expr.eval();
}

template<typename E>
Dual getValue(NegExpr<E> expr)
{
    return expr.eval();
}

template<typename E>
Dual getValue(SinExpr<E> expr)
{
    return expr.eval();
}

template<typename E>
Dual getValue(CosExpr<E> expr)
{
    return expr.eval();
}

template<typename E>
Dual getValue(ExpExpr<E> expr)
{
    return expr.eval();
}

template<typename E>
Dual getValue(LogExpr<E> expr)
{
    return expr.eval();
}

template<typename E>
Dual getValue(SqrtExpr<E> expr)
{
    return expr.eval();
}

// Extract Dual from Dual (identity)
Dual getValue(Dual d)
{
    return d;
}

// Extract Dual from float (zero derivative)
Dual getValue(float f)
{
    return makeDual(f, 0.0f);
}

// Helper functions to create expression templates
template<typename L, typename R>
AddExpr<L, R> makeAddExpr(L left, R right)
{
    AddExpr<L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename L, typename R>
SubExpr<L, R> makeSubExpr(L left, R right)
{
    SubExpr<L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename L, typename R>
MulExpr<L, R> makeMulExpr(L left, R right)
{
    MulExpr<L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename L, typename R>
DivExpr<L, R> makeDivExpr(L left, R right)
{
    DivExpr<L, R> result;
    result.left = left;
    result.right = right;
    return result;
}

template<typename L, typename R>
PowExpr<L, R> makePowExpr(L base, R exponent)
{
    PowExpr<L, R> result;
    result.base = base;
    result.exponent = exponent;
    return result;
}

template<typename E>
NegExpr<E> makeNegExpr(E expr)
{
    NegExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
SinExpr<E> makeSinExpr(E expr)
{
    SinExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
CosExpr<E> makeCosExpr(E expr)
{
    CosExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
ExpExpr<E> makeExpExpr(E expr)
{
    ExpExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
LogExpr<E> makeLogExpr(E expr)
{
    LogExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
SqrtExpr<E> makeSqrtExpr(E expr)
{
    SqrtExpr<E> result;
    result.expr = expr;
    return result;
}

// ============================================================================
// Named Functions for Operations (HLSL doesn't support global operators)
// ============================================================================

// Addition functions
template<typename L, typename R>
AddExpr<L, R> add(L left, R right)
{
    return makeAddExpr(left, right);
}

template<typename L>
AddExpr<L, Dual> add(L left, float right)
{
    return makeAddExpr(left, makeDualValue(right));
}

template<typename R>
AddExpr<Dual, R> add(float left, R right)
{
    return makeAddExpr(makeDualValue(left), right);
}

Dual add(float left, Dual right)
{
    return makeDual(left + right.value, right.derivative);
}

// Subtraction functions
template<typename L, typename R>
SubExpr<L, R> subtract(L left, R right)
{
    return makeSubExpr(left, right);
}

template<typename L>
SubExpr<L, Dual> subtract(L left, float right)
{
    return makeSubExpr(left, makeDualValue(right));
}

template<typename R>
SubExpr<Dual, R> subtract(float left, R right)
{
    return makeSubExpr(makeDualValue(left), right);
}

Dual subtract(float left, Dual right)
{
    return makeDual(left - right.value, -right.derivative);
}

// Multiplication functions
template<typename L, typename R>
MulExpr<L, R> multiply(L left, R right)
{
    return makeMulExpr(left, right);
}

template<typename L>
MulExpr<L, Dual> multiply(L left, float right)
{
    return makeMulExpr(left, makeDualValue(right));
}

template<typename R>
MulExpr<Dual, R> multiply(float left, R right)
{
    return makeMulExpr(makeDualValue(left), right);
}

Dual multiply(float left, Dual right)
{
    return makeDual(left * right.value, left * right.derivative);
}

// Division functions
template<typename L, typename R>
DivExpr<L, R> divide(L left, R right)
{
    return makeDivExpr(left, right);
}

template<typename L>
DivExpr<L, Dual> divide(L left, float right)
{
    return makeDivExpr(left, makeDualValue(right));
}

template<typename R>
DivExpr<Dual, R> divide(float left, R right)
{
    return makeDivExpr(makeDualValue(left), right);
}

Dual divide(float left, Dual right)
{
    float denom = right.value * right.value;
    return makeDual(left / right.value, -left * right.derivative / denom);
}

// Negation functions
template<typename E>
NegExpr<E> negate(E expr)
{
    return makeNegExpr(expr);
}

// Power functions (replacing pow which might conflict with built-in)
template<typename L, typename R>
PowExpr<L, R> power(L base, R exponent)
{
    return makePowExpr(base, exponent);
}

// Mathematical Function Overloads
// ============================================================================

// Trigonometric functions
template<typename E>
SinExpr<E> sinExpr(E expr)
{
    return makeSinExpr(expr);
}

Dual sinDual(Dual d)
{
    return makeDual(sin(d.value), cos(d.value) * d.derivative);
}

template<typename E>
CosExpr<E> cosExpr(E expr)
{
    return makeCosExpr(expr);
}

Dual cosDual(Dual d)
{
    return makeDual(cos(d.value), -sin(d.value) * d.derivative);
}

Dual tanDual(Dual d)
{
    float cos_val = cos(d.value);
    float sec_sq = 1.0f / (cos_val * cos_val);
    return makeDual(tan(d.value), sec_sq * d.derivative);
}

// Exponential and logarithmic functions
template<typename E>
ExpExpr<E> expExpr(E expr)
{
    return makeExpExpr(expr);
}

Dual expDual(Dual d)
{
    float exp_val = exp(d.value);
    return makeDual(exp_val, exp_val * d.derivative);
}

template<typename E>
LogExpr<E> logExpr(E expr)
{
    return makeLogExpr(expr);
}

Dual logDual(Dual d)
{
    return makeDual(log(d.value), d.derivative / d.value);
}

// Square root
template<typename E>
SqrtExpr<E> sqrtExpr(E expr)
{
    return makeSqrtExpr(expr);
}

Dual sqrtDual(Dual d)
{
    float sqrt_val = sqrt(d.value);
    return makeDual(sqrt_val, d.derivative / (2.0f * sqrt_val));
}

// Absolute value
Dual absDual(Dual d)
{
    float sign_val = (d.value >= 0.0f) ? 1.0f : -1.0f;
    return makeDual(abs(d.value), sign_val * d.derivative);
}

// ============================================================================
// Utility Functions
// ============================================================================

// Create a variable with derivative 1 (for differentiation w.r.t. this variable)
Dual variable(float value)
{
    return makeDual(value, 1.0f);
}

// Create a constant with derivative 0
Dual constant(float value)
{
    return makeDual(value, 0.0f);
}

// Extract just the value from a dual number or expression
template<typename E>
float value(E expr)
{
    return getValue(expr).value;
}

// Extract just the derivative from a dual number or expression
template<typename E>
float derivative(E expr)
{
    return getValue(expr).derivative;
}

// ============================================================================
// Example Usage and Test Functions
// ============================================================================

/*
Example usage:

// Define a variable x with value 2.0 and derivative 1.0 (differentiating w.r.t. x)
Dual x = variable(2.0f);

// Define some constants
Dual a = constant(3.0f);
Dual b = constant(1.5f);

// Build a complex expression: f(x) = a * x^2 + b * sin(x) + exp(x/2)
// Note: Using named functions instead of operators for HLSL compatibility
DualSquaredExpr x_squared = multiply(x, x);
PowExpr<Dual, Dual> x_squared_alt = power(x, constant(2.0f));
MulExpr<Dual, PowExpr<Dual, Dual>> ax_squared = multiply(a, x_squared_alt);
SinExpr<Dual> sin_x = sinExpr(x);
MulExpr<Dual, SinExpr<Dual>> b_sin_x = multiply(b, sin_x);
DivExpr<Dual, Dual> x_div_2 = divide(x, constant(2.0f));
ExpExpr<DivExpr<Dual, Dual>> exp_x_div_2 = expExpr(x_div_2);
AddExpr<MulExpr<Dual, PowExpr<Dual, Dual>>, MulExpr<Dual, SinExpr<Dual>>> part1 = add(ax_squared, b_sin_x);
AddExpr<AddExpr<...>, ExpExpr<...>> expr = add(part1, exp_x_div_2);

// Evaluate the expression to get both value and derivative
Dual result = expr.eval();
float f_val = result.value;      // Function value at x=2.0
float f_deriv = result.derivative; // Derivative at x=2.0

// Or use helper functions
float val = value(expr);         // Same as result.value
float deriv = derivative(expr);  // Same as result.derivative
*/

#endif // FORWARD_AD_HLSL