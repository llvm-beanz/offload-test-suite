#ifndef BACKWARD_AD_HLSL
#define BACKWARD_AD_HLSL

// ============================================================================
// Backward Automatic Differentiation for HLSL - Templated Version
// ============================================================================
// This header provides reverse-mode automatic differentiation using expression 
// templates. Backward mode is efficient for functions with many inputs and 
// few outputs (like gradients for optimization).

// Maximum number of variables supported (for gradient storage)
#define MAX_VARIABLES 64

// Global gradient storage - template specialized per type
#define DECLARE_GRADIENT_STORAGE(T) \
    static T g_gradients_##T[MAX_VARIABLES]; \
    static int g_variable_count_##T = 0;

// Declare storage for common types
DECLARE_GRADIENT_STORAGE(float)
DECLARE_GRADIENT_STORAGE(double)
DECLARE_GRADIENT_STORAGE(half)
DECLARE_GRADIENT_STORAGE(int)

// ============================================================================
// Variable Class - Represents Input Variables
// ============================================================================

template<typename T>
struct Variable
{
    T value;
    int id;
    
    // Note: HLSL doesn't support constructors, so we use init functions
    static Variable makeVariable(T val);
    
    // Get the current gradient for this variable
    T gradient();
    
    // Reset gradient to zero
    void zeroGradient();
};

// Template specializations for Variable methods
#define SPECIALIZE_VARIABLE(T) \
template<> \
Variable<T> Variable<T>::makeVariable(T val) \
{ \
    Variable<T> var; \
    var.value = val; \
    var.id = g_variable_count_##T; \
    g_variable_count_##T++; \
    return var; \
} \
template<> \
T Variable<T>::gradient() \
{ \
    return g_gradients_##T[id]; \
} \
template<> \
void Variable<T>::zeroGradient() \
{ \
    g_gradients_##T[id] = T(0); \
}

SPECIALIZE_VARIABLE(float)
SPECIALIZE_VARIABLE(double)
SPECIALIZE_VARIABLE(half)
SPECIALIZE_VARIABLE(int)

// ============================================================================
// Variable Expression (Leaf Node)
// ============================================================================

template<typename T>
struct VariableExpr
{
    Variable<T> var;
    
    T forward()
    {
        return var.value;
    }
    
    void backward(T gradient)
    {
        // Accumulate gradient for this variable - template specialized
        // Will be defined per type below
    }
};

// Template specializations for VariableExpr backward method
#define SPECIALIZE_VARIABLE_EXPR(T) \
template<> \
void VariableExpr<T>::backward(T gradient) \
{ \
    g_gradients_##T[var.id] += gradient; \
}

SPECIALIZE_VARIABLE_EXPR(float)
SPECIALIZE_VARIABLE_EXPR(double)
SPECIALIZE_VARIABLE_EXPR(half)
SPECIALIZE_VARIABLE_EXPR(int)

// ============================================================================
// Addition Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackAddExpr
{
    L left;
    R right;
    
    T forward()
    {
        return left.forward() + right.forward();
    }
    
    void backward(T gradient)
    {
        // d(left + right)/dleft = 1, d(left + right)/dright = 1
        left.backward(gradient);
        right.backward(gradient);
    }
};

// ============================================================================
// Scalar-Expression Operations (for constants like 2.0f + expr)
// ============================================================================

// Scalar + Expression
template<typename T, typename R>
struct BackAddExpr<T, T, R>
{
    T left;        // scalar constant
    R right;       // expression
    
    T forward()
    {
        return left + right.forward();
    }
    
    void backward(T gradient)
    {
        // d(scalar + right)/dright = 1
        // No gradient for scalar constant
        right.backward(gradient);
    }
};

// Expression + Scalar
template<typename T, typename L>
struct BackAddExpr<T, L, T>
{
    L left;        // expression
    T right;       // scalar constant
    
    T forward()
    {
        return left.forward() + right;
    }
    
    void backward(T gradient)
    {
        // d(left + scalar)/dleft = 1
        // No gradient for scalar constant
        left.backward(gradient);
    }
};

// ============================================================================
// Subtraction Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackSubExpr
{
    L left;
    R right;
    
    T forward()
    {
        return left.forward() - right.forward();
    }
    
    void backward(T gradient)
    {
        // d(left - right)/dleft = 1, d(left - right)/dright = -1
        left.backward(gradient);
        right.backward(-gradient);
    }
};

// ============================================================================  
// Scalar-Expression Subtraction
// ============================================================================

// Scalar - Expression
template<typename T, typename R>
struct BackSubExpr<T, T, R>
{
    T left;        // scalar constant
    R right;       // expression
    
    T forward()
    {
        return left - right.forward();
    }
    
    void backward(T gradient)
    {
        // d(scalar - right)/dright = -1
        // No gradient for scalar constant  
        right.backward(-gradient);
    }
};

// Expression - Scalar
template<typename T, typename L>
struct BackSubExpr<T, L, T>
{
    L left;        // expression
    T right;       // scalar constant
    
    T forward()
    {
        return left.forward() - right;
    }
    
    void backward(T gradient)
    {
        // d(left - scalar)/dleft = 1
        // No gradient for scalar constant
        left.backward(gradient);
    }
};

// ============================================================================
// Multiplication Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackMulExpr
{
    L left;
    R right;
    T left_val;
    T right_val;
    
    T forward()
    {
        left_val = left.forward();
        right_val = right.forward();
        return left_val * right_val;
    }
    
    void backward(T gradient)
    {
        // d(left * right)/dleft = right, d(left * right)/dright = left
        left.backward(gradient * right_val);
        right.backward(gradient * left_val);
    }
};

// ============================================================================
// Scalar-Expression Multiplication (for constants like 2.0f * expr)
// ============================================================================

template<typename T, typename R>
struct BackMulExpr<T, T, R>
{
    T left;        // scalar constant
    R right;       // expression
    T left_val;
    T right_val;
    
    T forward()
    {
        left_val = left;  // scalar value
        right_val = right.forward();  // expression value
        return left_val * right_val;
    }
    
    void backward(T gradient)
    {
        // d(scalar * right)/dright = scalar
        // No gradient for scalar constant
        right.backward(gradient * left_val);
    }
};

// Expression-Scalar Multiplication (for expressions like expr * 2.0f)
template<typename T, typename L>
struct BackMulExpr<T, L, T>
{
    L left;        // expression
    T right;       // scalar constant
    T left_val;
    T right_val;
    
    T forward()
    {
        left_val = left.forward();  // expression value
        right_val = right;  // scalar value
        return left_val * right_val;
    }
    
    void backward(T gradient)
    {
        // d(left * scalar)/dleft = scalar
        // No gradient for scalar constant
        left.backward(gradient * right_val);
    }
};

// ============================================================================
// Division Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackDivExpr
{
    L left;
    R right;
    T left_val;
    T right_val;
    
    T forward()
    {
        left_val = left.forward();
        right_val = right.forward();
        return left_val / right_val;
    }
    
    void backward(T gradient)
    {
        // d(left / right)/dleft = 1/right, d(left / right)/dright = -left/right^2
        left.backward(gradient / right_val);
        right.backward(-gradient * left_val / (right_val * right_val));
    }
};

// ============================================================================
// Scalar-Expression Division
// ============================================================================

// Scalar / Expression
template<typename T, typename R>
struct BackDivExpr<T, T, R>
{
    T left;        // scalar constant
    R right;       // expression
    T left_val;
    T right_val;
    
    T forward()
    {
        left_val = left;  // scalar value
        right_val = right.forward();  // expression value
        return left_val / right_val;
    }
    
    void backward(T gradient)
    {
        // d(scalar / right)/dright = -scalar/right^2
        // No gradient for scalar constant
        right.backward(-gradient * left_val / (right_val * right_val));
    }
};

// Expression / Scalar
template<typename T, typename L>
struct BackDivExpr<T, L, T>
{
    L left;        // expression
    T right;       // scalar constant
    T left_val;
    T right_val;
    
    T forward()
    {
        left_val = left.forward();  // expression value
        right_val = right;  // scalar value
        return left_val / right_val;
    }
    
    void backward(T gradient)
    {
        // d(left / scalar)/dleft = 1/scalar
        // No gradient for scalar constant
        left.backward(gradient / right_val);
    }
};

// ============================================================================
// Power Expression
// ============================================================================

template<typename T, typename L, typename R>
struct BackPowExpr
{
    L base;
    R exponent;
    T base_val;
    T exp_val;
    T result_val;
    
    T forward()
    {
        base_val = base.forward();
        exp_val = exponent.forward();
        result_val = pow(base_val, exp_val);
        return result_val;
    }
    
    void backward(T gradient)
    {
        // d(base^exp)/dbase = exp * base^(exp-1)
        // d(base^exp)/dexp = base^exp * log(base)
        base.backward(gradient * exp_val * pow(base_val, exp_val - T(1)));
        exponent.backward(gradient * result_val * log(base_val));
    }
};

// ============================================================================
// Negation Expression
// ============================================================================

template<typename T, typename E>
struct BackNegExpr
{
    E expr;
    
    T forward()
    {
        return -expr.forward();
    }
    
    void backward(T gradient)
    {
        // d(-expr)/dexpr = -1
        expr.backward(-gradient);
    }
};

// ============================================================================
// Trigonometric Functions
// ============================================================================

template<typename T, typename E>
struct BackSinExpr
{
    E expr;
    T expr_val;
    
    T forward()
    {
        expr_val = expr.forward();
        return sin(expr_val);
    }
    
    void backward(T gradient)
    {
        // d(sin(x))/dx = cos(x)
        expr.backward(gradient * cos(expr_val));
    }
};

template<typename T, typename E>
struct BackCosExpr
{
    E expr;
    T expr_val;
    
    T forward()
    {
        expr_val = expr.forward();
        return cos(expr_val);
    }
    
    void backward(T gradient)
    {
        // d(cos(x))/dx = -sin(x)
        expr.backward(gradient * (-sin(expr_val)));
    }
};

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

template<typename T, typename E>
struct BackExpExpr
{
    E expr;
    T result_val;
    
    T forward()
    {
        T expr_val = expr.forward();
        result_val = exp(expr_val);
        return result_val;
    }
    
    void backward(T gradient)
    {
        // d(exp(x))/dx = exp(x)
        expr.backward(gradient * result_val);
    }
};

template<typename T, typename E>
struct BackLogExpr
{
    E expr;
    T expr_val;
    
    T forward()
    {
        expr_val = expr.forward();
        return log(expr_val);
    }
    
    void backward(T gradient)
    {
        // d(log(x))/dx = 1/x
        expr.backward(gradient / expr_val);
    }
};

// ============================================================================
// Square Root Expression
// ============================================================================

template<typename T, typename E>
struct BackSqrtExpr
{
    E expr;
    T expr_val;
    T result_val;
    
    T forward()
    {
        expr_val = expr.forward();
        result_val = sqrt(expr_val);
        return result_val;
    }
    
    void backward(T gradient)
    {
        // d(sqrt(x))/dx = 1/(2*sqrt(x))
        expr.backward(gradient / (T(2) * result_val));
    }
};

// ============================================================================
// High-Level API Functions
// ============================================================================

// Create a variable
template<typename T>
Variable<T> variable(T value)
{
    return Variable<T>::makeVariable(value);
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

// Binary operations
template<typename T, typename L, typename R>
BackAddExpr<T, L, R> add(L left, R right)
{
    BackAddExpr<T, L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename T, typename L, typename R>
BackSubExpr<T, L, R> subtract(L left, R right)
{
    BackSubExpr<T, L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename T, typename L, typename R>
BackMulExpr<T, L, R> multiply(L left, R right)
{
    BackMulExpr<T, L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename T, typename L, typename R>
BackDivExpr<T, L, R> divide(L left, R right)
{
    BackDivExpr<T, L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename T, typename L, typename R>
BackPowExpr<T, L, R> power(L base, R exponent)
{
    BackPowExpr<T, L, R> expr;
    expr.base = base;
    expr.exponent = exponent;
    return expr;
}

// Unary operations
template<typename T, typename E>
BackNegExpr<T, E> negate(E expr)
{
    BackNegExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
BackSinExpr<T, E> sinExpr(E expr)
{
    BackSinExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
BackCosExpr<T, E> cosExpr(E expr)
{
    BackCosExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
BackExpExpr<T, E> expExpr(E expr)
{
    BackExpExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
BackLogExpr<T, E> logExpr(E expr)
{
    BackLogExpr<T, E> result;
    result.expr = expr;
    return result;
}

template<typename T, typename E>
BackSqrtExpr<T, E> sqrtExpr(E expr)
{
    BackSqrtExpr<T, E> result;
    result.expr = expr;
    return result;
}

// ============================================================================
// Computation Functions
// ============================================================================

// Template specializations for reset functions
#define SPECIALIZE_RESET(T) \
template<> \
void reset_variables<T>() \
{ \
    for (int i = 0; i < g_variable_count_##T; i++) \
    { \
        g_gradients_##T[i] = T(0); \
    } \
    g_variable_count_##T = 0; \
}

// Reset all variables for a specific type
template<typename T>
void reset_variables();

SPECIALIZE_RESET(float)
SPECIALIZE_RESET(double)
SPECIALIZE_RESET(half)
SPECIALIZE_RESET(int)

// Template specializations for compute_gradients functions
#define SPECIALIZE_COMPUTE_GRADIENTS(T) \
template<> \
T compute_gradients<T>(VariableExpr<T> var_expr) \
{ \
    for (int i = 0; i < g_variable_count_##T; i++) \
    { \
        g_gradients_##T[i] = T(0); \
    } \
    T result = var_expr.forward(); \
    var_expr.backward(T(1)); \
    return result; \
}

// Combined forward and backward pass for different expression types
template<typename T>
T compute_gradients(VariableExpr<T> var_expr);

template<typename T, typename L, typename R>
T compute_gradients(BackAddExpr<T, L, R> expr)
{
    // Zero all gradients for type T - simplified for float for now
    for (int i = 0; i < g_variable_count_float; i++)
    {
        g_gradients_float[i] = 0.0f;
    }
    
    // Forward pass
    T result = expr.forward();
    
    // Backward pass (starting with gradient = 1.0 for the output)
    expr.backward(T(1));
    
    return result;
}

template<typename T, typename L, typename R>
T compute_gradients(BackSubExpr<T, L, R> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename L, typename R>
T compute_gradients(BackMulExpr<T, L, R> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename L, typename R>
T compute_gradients(BackDivExpr<T, L, R> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename L, typename R>  
T compute_gradients(BackPowExpr<T, L, R> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename E>
T compute_gradients(BackNegExpr<T, E> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename E>
T compute_gradients(BackSinExpr<T, E> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename E>
T compute_gradients(BackCosExpr<T, E> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename E>
T compute_gradients(BackExpExpr<T, E> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename E>
T compute_gradients(BackLogExpr<T, E> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

template<typename T, typename E>
T compute_gradients(BackSqrtExpr<T, E> expr)
{
    for (int i = 0; i < g_variable_count_float; i++) { g_gradients_float[i] = 0.0f; }
    T result = expr.forward();
    expr.backward(T(1));
    return result;
}

SPECIALIZE_COMPUTE_GRADIENTS(float)
SPECIALIZE_COMPUTE_GRADIENTS(double)
SPECIALIZE_COMPUTE_GRADIENTS(half)
SPECIALIZE_COMPUTE_GRADIENTS(int)

#endif // BACKWARD_AD_HLSL