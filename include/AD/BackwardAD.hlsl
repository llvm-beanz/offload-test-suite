#ifndef BACKWARD_AD_HLSL
#define BACKWARD_AD_HLSL

// ============================================================================
// Backward Automatic Differentiation for HLSL
// ============================================================================
// This header provides reverse-mode automatic differentiation using expression 
// templates. Backward mode is efficient for functions with many inputs and 
// few outputs (like gradients for optimization).

// Forward declarations
struct Variable;
template<typename E> struct BackwardExpr;

// Maximum number of variables supported (for gradient storage)
#define MAX_VARIABLES 64

// Global gradient storage
static float g_gradients[MAX_VARIABLES];
static int g_variable_count = 0;

// ============================================================================
// Variable Class - Represents Input Variables
// ============================================================================

struct Variable
{
    float value;
    int id;
    
    // Note: HLSL doesn't support constructors, so we use init functions
    static Variable makeVariable(float val)
    {
        Variable var;
        var.value = val;
        var.id = g_variable_count;
        g_variable_count++;
        return var;
    }
    
    // Get the current gradient for this variable
    float gradient()
    {
        return g_gradients[id];
    }
    
    // Reset gradient to zero
    void zeroGradient()
    {
        g_gradients[id] = 0.0f;
    }
};

// ============================================================================
// Base Expression Template
// ============================================================================

template<typename Derived>
struct BackwardExpr
{
    // Forward evaluation - compute the value
    float forward()
    {
        // HLSL doesn't support static_cast, so we use direct function calls
        // This will be specialized by derived classes
        return 0.0f; // Default implementation
    }
    
    // Backward pass - propagate gradients
    void backward(float gradient)
    {
        // HLSL doesn't support static_cast, so we use direct function calls
        // This will be specialized by derived classes
    }
};

// ============================================================================
// Variable Expression (Leaf Node)
// ============================================================================

struct VariableExpr
{
    Variable var;
    
    float forward()
    {
        return var.value;
    }
    
    void backward(float gradient)
    {
        // Accumulate gradient for this variable
        g_gradients[var.id] += gradient;
    }
};

// ============================================================================
// Addition Expression
// ============================================================================

template<typename L, typename R>
struct BackAddExpr
{
    L left;
    R right;
    
    float forward()
    {
        return getForward(left) + getForward(right);
    }
    
    void backward(float gradient)
    {
        // d(left + right)/dleft = 1, d(left + right)/dright = 1
        propagateBackward(left, gradient);
        propagateBackward(right, gradient);
    }
};

// ============================================================================
// Subtraction Expression
// ============================================================================

template<typename L, typename R>
struct BackSubExpr
{
    L left;
    R right;
    
    float forward()
    {
        return getForward(left) - getForward(right);
    }
    
    void backward(float gradient)
    {
        // d(left - right)/dleft = 1, d(left - right)/dright = -1
        propagateBackward(left, gradient);
        propagateBackward(right, -gradient);
    }
};

// ============================================================================
// Multiplication Expression
// ============================================================================

template<typename L, typename R>
struct BackMulExpr
{
    L left;
    R right;
    float left_val;
    float right_val;
    
    float forward()
    {
        left_val = getForward(left);
        right_val = getForward(right);
        return left_val * right_val;
    }
    
    void backward(float gradient)
    {
        // d(left * right)/dleft = right, d(left * right)/dright = left
        propagateBackward(left, gradient * right_val);
        propagateBackward(right, gradient * left_val);
    }
};

// ============================================================================
// Division Expression
// ============================================================================

template<typename L, typename R>
struct BackDivExpr
{
    L left;
    R right;
    float left_val;
    float right_val;
    
    float forward()
    {
        left_val = getForward(left);
        right_val = getForward(right);
        return left_val / right_val;
    }
    
    void backward(float gradient)
    {
        // d(left / right)/dleft = 1/right, d(left / right)/dright = -left/right^2
        propagateBackward(left, gradient / right_val);
        propagateBackward(right, -gradient * left_val / (right_val * right_val));
    }
};

// ============================================================================
// Power Expression
// ============================================================================

template<typename L, typename R>
struct BackPowExpr
{
    L base;
    R exponent;
    float base_val;
    float exp_val;
    float result_val;
    
    float forward()
    {
        base_val = getForward(base);
        exp_val = getForward(exponent);
        result_val = pow(base_val, exp_val);
        return result_val;
    }
    
    void backward(float gradient)
    {
        // d(base^exp)/dbase = exp * base^(exp-1)
        // d(base^exp)/dexp = base^exp * log(base)
        propagateBackward(base, gradient * exp_val * pow(base_val, exp_val - 1.0f));
        propagateBackward(exponent, gradient * result_val * log(base_val));
    }
};

// ============================================================================
// Negation Expression
// ============================================================================

template<typename E>
struct BackNegExpr
{
    E expr;
    
    float forward()
    {
        return -getForward(expr);
    }
    
    void backward(float gradient)
    {
        // d(-expr)/dexpr = -1
        propagateBackward(expr, -gradient);
    }
};

// ============================================================================
// Trigonometric Functions
// ============================================================================

template<typename E>
struct BackSinExpr
{
    E expr;
    float expr_val;
    
    float forward()
    {
        expr_val = getForward(expr);
        return sin(expr_val);
    }
    
    void backward(float gradient)
    {
        // d(sin(x))/dx = cos(x)
        propagateBackward(expr, gradient * cos(expr_val));
    }
};

template<typename E>
struct BackCosExpr
{
    E expr;
    float expr_val;
    
    float forward()
    {
        expr_val = getForward(expr);
        return cos(expr_val);
    }
    
    void backward(float gradient)
    {
        // d(cos(x))/dx = -sin(x)
        propagateBackward(expr, gradient * (-sin(expr_val)));
    }
};

// ============================================================================
// Exponential and Logarithmic Functions
// ============================================================================

template<typename E>
struct BackExpExpr
{
    E expr;
    float result_val;
    
    float forward()
    {
        float expr_val = getForward(expr);
        result_val = exp(expr_val);
        return result_val;
    }
    
    void backward(float gradient)
    {
        // d(exp(x))/dx = exp(x)
        propagateBackward(expr, gradient * result_val);
    }
};

template<typename E>
struct BackLogExpr
{
    E expr;
    float expr_val;
    
    float forward()
    {
        expr_val = getForward(expr);
        return log(expr_val);
    }
    
    void backward(float gradient)
    {
        // d(log(x))/dx = 1/x
        propagateBackward(expr, gradient / expr_val);
    }
};

// ============================================================================
// Square Root Expression
// ============================================================================

template<typename E>
struct BackSqrtExpr
{
    E expr;
    float expr_val;
    float result_val;
    
    float forward()
    {
        expr_val = getForward(expr);
        result_val = sqrt(expr_val);
        return result_val;
    }
    
    void backward(float gradient)
    {
        // d(sqrt(x))/dx = 1/(2*sqrt(x))
        propagateBackward(expr, gradient / (2.0f * result_val));
    }
};

// ============================================================================
// Helper Functions for Forward Pass
// =============================================================================

// Get forward value from variable
float getForward(VariableExpr var_expr)
{
    return var_expr.forward();
}

// Get forward value from different expression types
template<typename L, typename R>
float getForward(BackAddExpr<L, R> expr)
{
    return expr.forward();
}

template<typename L, typename R>
float getForward(BackSubExpr<L, R> expr)
{
    return expr.forward();
}

template<typename L, typename R>
float getForward(BackMulExpr<L, R> expr)
{
    return expr.forward();
}

template<typename L, typename R>
float getForward(BackDivExpr<L, R> expr)
{
    return expr.forward();
}

template<typename L, typename R>
float getForward(BackPowExpr<L, R> expr)
{
    return expr.forward();
}

template<typename E>
float getForward(BackNegExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float getForward(BackSinExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float getForward(BackCosExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float getForward(BackExpExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float getForward(BackLogExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float getForward(BackSqrtExpr<E> expr)
{
    return expr.forward();
}

// Get forward value from float constant
float getForward(float val)
{
    return val;
}

// ============================================================================
// Helper Functions for Backward Pass
// ============================================================================

// Propagate gradient to variable
void propagateBackward(VariableExpr var_expr, float gradient)
{
    var_expr.backward(gradient);
}

// Propagate gradient to different expression types
template<typename L, typename R>
void propagateBackward(BackAddExpr<L, R> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename L, typename R>
void propagateBackward(BackSubExpr<L, R> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename L, typename R>
void propagateBackward(BackMulExpr<L, R> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename L, typename R>
void propagateBackward(BackDivExpr<L, R> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename L, typename R>
void propagateBackward(BackPowExpr<L, R> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename E>
void propagateBackward(BackNegExpr<E> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename E>
void propagateBackward(BackSinExpr<E> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename E>
void propagateBackward(BackCosExpr<E> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename E>
void propagateBackward(BackExpExpr<E> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename E>
void propagateBackward(BackLogExpr<E> expr, float gradient)
{
    expr.backward(gradient);
}

template<typename E>
void propagateBackward(BackSqrtExpr<E> expr, float gradient)
{
    expr.backward(gradient);
}

// Propagate gradient to float constant (no-op)
void propagateBackward(float val, float gradient)
{
    // Constants don't accumulate gradients
}

// ============================================================================
// Expression Factory Functions
// ============================================================================

// Create variable expression
VariableExpr makeVariableExpr(Variable var)
{
    VariableExpr expr;
    expr.var = var;
    return expr;
}

// Create binary operation expressions
template<typename L, typename R>
BackAddExpr<L, R> makeBackAddExpr(L left, R right)
{
    BackAddExpr<L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename L, typename R>
BackSubExpr<L, R> makeBackSubExpr(L left, R right)
{
    BackSubExpr<L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename L, typename R>
BackMulExpr<L, R> makeBackMulExpr(L left, R right)
{
    BackMulExpr<L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename L, typename R>
BackDivExpr<L, R> makeBackDivExpr(L left, R right)
{
    BackDivExpr<L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename L, typename R>
BackPowExpr<L, R> makeBackPowExpr(L base, R exponent)
{
    BackPowExpr<L, R> expr;
    expr.base = base;
    expr.exponent = exponent;
    return expr;
}

// Create unary operation expressions
template<typename E>
BackNegExpr<E> makeBackNegExpr(E expr)
{
    BackNegExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
BackSinExpr<E> makeBackSinExpr(E expr)
{
    BackSinExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
BackCosExpr<E> makeBackCosExpr(E expr)
{
    BackCosExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
BackExpExpr<E> makeBackExpExpr(E expr)
{
    BackExpExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
BackLogExpr<E> makeBackLogExpr(E expr)
{
    BackLogExpr<E> result;
    result.expr = expr;
    return result;
}

template<typename E>
BackSqrtExpr<E> makeBackSqrtExpr(E expr)
{
    BackSqrtExpr<E> result;
    result.expr = expr;
    return result;
}

// ============================================================================
// High-Level API Functions
// ============================================================================

// Create a variable
Variable variable(float value)
{
    return Variable::makeVariable(value);
}

// Create a constant (just returns the float value)
float constant(float value)
{
    return value;
}

// Binary operations
template<typename L, typename R>
BackAddExpr<L, R> add(L left, R right)
{
    return makeBackAddExpr(left, right);
}

template<typename L, typename R>
BackSubExpr<L, R> subtract(L left, R right)
{
    return makeBackSubExpr(left, right);
}

template<typename L, typename R>
BackMulExpr<L, R> multiply(L left, R right)
{
    return makeBackMulExpr(left, right);
}

template<typename L, typename R>
BackDivExpr<L, R> divide(L left, R right)
{
    return makeBackDivExpr(left, right);
}

template<typename L, typename R>
BackPowExpr<L, R> power(L base, R exponent)
{
    return makeBackPowExpr(base, exponent);
}

// Unary operations
template<typename E>
BackNegExpr<E> negate(E expr)
{
    return makeBackNegExpr(expr);
}

template<typename E>
BackSinExpr<E> sinExpr(E expr)
{
    return makeBackSinExpr(expr);
}

template<typename E>
BackCosExpr<E> cosExpr(E expr)
{
    return makeBackCosExpr(expr);
}

template<typename E>
BackExpExpr<E> expExpr(E expr)
{
    return makeBackExpExpr(expr);
}

template<typename E>
BackLogExpr<E> logExpr(E expr)
{
    return makeBackLogExpr(expr);
}

template<typename E>
BackSqrtExpr<E> sqrtExpr(E expr)
{
    return makeBackSqrtExpr(expr);
}

// ============================================================================
// Computation Functions
// ============================================================================

// Forward pass: compute function value
float forward(VariableExpr var_expr)
{
    return var_expr.forward();
}

template<typename L, typename R>
float forward(BackAddExpr<L, R> expr)
{
    return expr.forward();
}

template<typename L, typename R>
float forward(BackSubExpr<L, R> expr)
{
    return expr.forward();
}

template<typename L, typename R>
float forward(BackMulExpr<L, R> expr)
{
    return expr.forward();
}

template<typename L, typename R>
float forward(BackDivExpr<L, R> expr)
{
    return expr.forward();
}

template<typename L, typename R>
float forward(BackPowExpr<L, R> expr)
{
    return expr.forward();
}

template<typename E>
float forward(BackNegExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float forward(BackSinExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float forward(BackCosExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float forward(BackExpExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float forward(BackLogExpr<E> expr)
{
    return expr.forward();
}

template<typename E>
float forward(BackSqrtExpr<E> expr)
{
    return expr.forward();
}

float forward(float val)
{
    return val;
}

// Backward pass: compute gradients
void backward(VariableExpr var_expr, float gradient = 1.0f)
{
    var_expr.backward(gradient);
}

template<typename L, typename R>
void backward(BackAddExpr<L, R> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename L, typename R>
void backward(BackSubExpr<L, R> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename L, typename R>
void backward(BackMulExpr<L, R> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename L, typename R>
void backward(BackDivExpr<L, R> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename L, typename R>
void backward(BackPowExpr<L, R> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename E>
void backward(BackNegExpr<E> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename E>
void backward(BackSinExpr<E> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename E>
void backward(BackCosExpr<E> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename E>
void backward(BackExpExpr<E> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename E>
void backward(BackLogExpr<E> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

template<typename E>
void backward(BackSqrtExpr<E> expr, float gradient = 1.0f)
{
    expr.backward(gradient);
}

// Combined forward and backward pass
template<typename E>
float compute_gradients(BackwardExpr<E> expr)
{
    // Zero all gradients
    for (int i = 0; i < g_variable_count; i++)
    {
        g_gradients[i] = 0.0f;
    }
    
    // Forward pass
    float result = expr.forward();
    
    // Backward pass (starting with gradient = 1.0 for the output)
    expr.backward(1.0f);
    
    return result;
}

float compute_gradients(VariableExpr var_expr)
{
    // Zero all gradients
    for (int i = 0; i < g_variable_count; i++)
    {
        g_gradients[i] = 0.0f;
    }
    
    // Forward pass
    float result = var_expr.forward();
    
    // Backward pass
    var_expr.backward(1.0f);
    
    return result;
}

// Reset the variable counter (for new computations)
void reset_variables()
{
    g_variable_count = 0;
    for (int i = 0; i < MAX_VARIABLES; i++)
    {
        g_gradients[i] = 0.0f;
    }
}

#endif // BACKWARD_AD_HLSL