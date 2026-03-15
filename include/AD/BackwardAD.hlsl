#ifndef BACKWARD_AD_HLSL
#define BACKWARD_AD_HLSL

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
        context.gradients[id] = T(0);
    }
};



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
    L left;
    R right;
    
    T forward()
    {
        return left.forward() + right.forward();
    }
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left + right)/dleft = 1, d(left + right)/dright = 1
        left.backward(context, gradient);
        right.backward(context, gradient);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(scalar + right)/dright = 1
        // No gradient for scalar constant
        right.backward(context, gradient);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left + scalar)/dleft = 1
        // No gradient for scalar constant
        left.backward(context, gradient);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left - right)/dleft = 1, d(left - right)/dright = -1
        left.backward(context, gradient);
        right.backward(context, -gradient);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(scalar - right)/dright = -1
        // No gradient for scalar constant  
        right.backward(context, -gradient);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left - scalar)/dleft = 1
        // No gradient for scalar constant
        left.backward(context, gradient);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left * right)/dleft = right, d(left * right)/dright = left
        left.backward(context, gradient * right_val);
        right.backward(context, gradient * left_val);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(scalar * right)/dright = scalar
        // No gradient for scalar constant
        right.backward(context, gradient * left_val);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left * scalar)/dleft = scalar
        // No gradient for scalar constant
        left.backward(context, gradient * right_val);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left / right)/dleft = 1/right, d(left / right)/dright = -left/right^2
        left.backward(context, gradient / right_val);
        right.backward(context, -gradient * left_val / (right_val * right_val));
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(scalar / right)/dright = -scalar/right^2
        // No gradient for scalar constant
        right.backward(context, -gradient * left_val / (right_val * right_val));
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(left / scalar)/dleft = 1/scalar
        // No gradient for scalar constant
        left.backward(context, gradient / right_val);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(base^exp)/dbase = exp * base^(exp-1)
        // d(base^exp)/dexp = base^exp * log(base)
        base.backward(context, gradient * exp_val * pow(base_val, exp_val - T(1)));
        exponent.backward(context, gradient * result_val * log(base_val));
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(-expr)/dexpr = -1
        expr.backward(context, -gradient);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(sin(x))/dx = cos(x)
        expr.backward(context, gradient * cos(expr_val));
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(cos(x))/dx = -sin(x)
        expr.backward(context, gradient * (-sin(expr_val)));
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(exp(x))/dx = exp(x)
        expr.backward(context, gradient * result_val);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(log(x))/dx = 1/x
        expr.backward(context, gradient / expr_val);
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
    
    void backward(inout GradientContext<T> context, T gradient)
    {
        // d(sqrt(x))/dx = 1/(2*sqrt(x))
        expr.backward(context, gradient / (T(2) * result_val));
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
#define MAKE_BINARY_BACK_EXPR(ExprType) \
template<typename T, typename L, typename R> \
ExprType<T, L, R> ExprType##LowerCase(L left, R right) \
{ \
    ExprType<T, L, R> expr; \
    expr.left = left; \
    expr.right = right; \
    return expr; \
}

// Helper to convert ExprType to lowercase function name
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
template<typename T, int N, typename L, typename R>
struct BackDotExpr
{
    L left;
    R right;
    vector<T, N> left_val;
    vector<T, N> right_val;
    
    T forward()
    {
        left_val = left.forward();
        right_val = right.forward();
        return dot(left_val, right_val);
    }
    
    void backward(inout GradientContext<vector<T, N> > context, T gradient)
    {
        // d(dot(u,v))/du = v, d(dot(u,v))/dv = u
        left.backward(context, right_val * gradient);
        right.backward(context, left_val * gradient);
    }
};

// Vector Cross Product (3D only)
template<typename T, typename L, typename R>
struct BackCrossExpr
{
    L left;
    R right;
    vector<T, 3> left_val;
    vector<T, 3> right_val;
    
    vector<T, 3> forward()
    {
        left_val = left.forward();
        right_val = right.forward();
        return cross(left_val, right_val);
    }
    
    void backward(inout GradientContext<vector<T, 3> > context, vector<T, 3> gradient)
    {
        // d(cross(u,v))/du = cross(gradient, v), d(cross(u,v))/dv = cross(u, gradient)
        left.backward(context, cross(gradient, right_val));
        right.backward(context, cross(left_val, gradient));
    }
};

// Vector Length
template<typename T, int N, typename E>
struct BackLengthExpr
{
    E expr;
    vector<T, N> expr_val;
    T result_val;
    
    T forward()
    {
        expr_val = expr.forward();
        result_val = length(expr_val);
        return result_val;
    }
    
    void backward(inout GradientContext<vector<T, N> > context, T gradient)
    {
        // d(|v|)/dv = v / |v|
        expr.backward(context, (expr_val / result_val) * gradient);
    }
};

// Vector Normalize
template<typename T, int N, typename E>
struct BackNormalizeExpr
{
    E expr;
    vector<T, N> expr_val;
    vector<T, N> result_val;
    T length_val;
    
    vector<T, N> forward()
    {
        expr_val = expr.forward();
        length_val = length(expr_val);
        result_val = normalize(expr_val);
        return result_val;
    }
    
    void backward(inout GradientContext<vector<T, N> > context, vector<T, N> gradient)
    {
        // d(normalize(v))/dv = (I - normalize(v) * normalize(v)^T) / |v|
        // Simplified: (gradient * length - result * dot(gradient, result)) / length
        T dot_grad_result = dot(gradient, result_val);
        vector<T, N> back_grad = (gradient * length_val - result_val * dot_grad_result) / length_val;
        expr.backward(context, back_grad);
    }
};

// ============================================================================
// Matrix Operations
// ============================================================================

// Matrix-Vector Multiplication
template<typename T, int N, int K, typename L, typename R>
struct BackMatVecMulExpr
{
    L left;   // Matrix NxK
    R right;  // Vector K
    matrix<T, N, K> left_val;
    vector<T, K> right_val;
    
    vector<T, N> forward()
    {
        left_val = left.forward();
        right_val = right.forward();
        return mul(left_val, right_val);
    }
    
    void backward(inout GradientContext<vector<T, N> > context, vector<T, N> gradient)
    {
        // d(A*v)/dA = v * gradient^T, d(A*v)/dv = A^T * gradient
        // For matrix gradient, we need a different context type - simplified here
        right.backward(context, mul(transpose(left_val), gradient));
    }
};

// Matrix Determinant (2x2 only for simplicity)
template<typename T, typename E>
struct BackDet2x2Expr
{
    E expr;
    matrix<T, 2, 2> expr_val;
    
    T forward()
    {
        expr_val = expr.forward();
        return determinant(expr_val);
    }
    
    void backward(inout GradientContext<matrix<T, 2, 2> > context, T gradient)
    {
        // d(det(M))/dM = adj(M)^T where adj is adjugate matrix
        // For 2x2: adj([[a,b],[c,d]]) = [[d,-b],[-c,a]]
        matrix<T, 2, 2> adj_matrix;
        adj_matrix[0][0] = expr_val[1][1];  // d
        adj_matrix[0][1] = -expr_val[0][1]; // -b
        adj_matrix[1][0] = -expr_val[1][0]; // -c  
        adj_matrix[1][1] = expr_val[0][0];  // a
        
        expr.backward(context, adj_matrix * gradient);
    }
};

// ============================================================================
// Vector/Matrix Variable Support
// ============================================================================

// Create vector variable
template<typename T, int N>
Variable<vector<T, N> > variableVector(inout GradientContext<vector<T, N> > context, vector<T, N> value)
{
    Variable<vector<T, N> > var;
    var.value = value;
    var.id = context.allocateVariable();
    return var;
}

// Create matrix variable
template<typename T, int N, int M>
Variable<matrix<T, N, M> > variableMatrix(inout GradientContext<matrix<T, N, M> > context, matrix<T, N, M> value)
{
    Variable<matrix<T, N, M> > var;
    var.value = value;
    var.id = context.allocateVariable();
    return var;
}

// ============================================================================
// Vector/Matrix Expression Creation Functions - Reduced Duplication
// ============================================================================

template<typename T, int N, typename L, typename R>
BackDotExpr<T, N, L, R> dotProduct(L left, R right)
{
    BackDotExpr<T, N, L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename T, typename L, typename R>
BackCrossExpr<T, L, R> crossProduct(L left, R right)
{
    BackCrossExpr<T, L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

// Macro for vector unary operations
#define MAKE_VECTOR_UNARY_OP_BACK(ExprType, funcName) \
template<typename T, int N, typename E> \
ExprType<T, N, E> funcName(E expr) \
{ \
    ExprType<T, N, E> result; \
    result.expr = expr; \
    return result; \
}

MAKE_VECTOR_UNARY_OP_BACK(BackLengthExpr, lengthExpr)
MAKE_VECTOR_UNARY_OP_BACK(BackNormalizeExpr, normalizeExpr)

template<typename T, int N, int K, typename L, typename R>
BackMatVecMulExpr<T, N, K, L, R> matVecMul(L left, R right)
{
    BackMatVecMulExpr<T, N, K, L, R> expr;
    expr.left = left;
    expr.right = right;
    return expr;
}

template<typename T, typename E>
BackDet2x2Expr<T, E> determinantExpr(E expr)
{
    BackDet2x2Expr<T, E> result;
    result.expr = expr;
    return result;
}

// ============================================================================
// Vector/Matrix Compute Gradients Functions - Reduced with Macros
// ============================================================================

// Note: Vector/matrix compute_gradients have different context and seed types
// so they need individual implementations, but can share some patterns

template<typename T, int N, typename L, typename R>
T compute_gradients(inout GradientContext<vector<T, N> > context, BackDotExpr<T, N, L, R> expr)
{
    context.zeroGradients();
    T result = expr.forward();
    expr.backward(context, (T)1);
    return result;
}

template<typename T, typename L, typename R>
vector<T, 3> compute_gradients(inout GradientContext<vector<T, 3> > context, BackCrossExpr<T, L, R> expr)
{
    context.zeroGradients();
    vector<T, 3> result = expr.forward();
    vector<T, 3> seed = { (T)1, (T)1, (T)1 };
    expr.backward(context, seed);
    return result;
}

// Macro for vector unary compute_gradients with T return type
#define MAKE_VECTOR_COMPUTE_GRADIENTS_SCALAR(ExprType) \
template<typename T, int N, typename E> \
T compute_gradients(inout GradientContext<vector<T, N> > context, ExprType<T, N, E> expr) \
{ \
    context.zeroGradients(); \
    T result = expr.forward(); \
    expr.backward(context, (T)1); \
    return result; \
}

MAKE_VECTOR_COMPUTE_GRADIENTS_SCALAR(BackLengthExpr)

// Macro for vector unary compute_gradients with vector return type
#define MAKE_VECTOR_COMPUTE_GRADIENTS_VECTOR(ExprType) \
template<typename T, int N, typename E> \
vector<T, N> compute_gradients(inout GradientContext<vector<T, N> > context, ExprType<T, N, E> expr) \
{ \
    context.zeroGradients(); \
    vector<T, N> result = expr.forward(); \
    vector<T, N> seed = (vector<T, N>)0; \
    if (N >= 1) seed[0] = (T)1; \
    expr.backward(context, seed); \
    return result; \
}

MAKE_VECTOR_COMPUTE_GRADIENTS_VECTOR(BackNormalizeExpr)

template<typename T, int N, int K, typename L, typename R>
vector<T, N> compute_gradients(inout GradientContext<vector<T, N> > context, BackMatVecMulExpr<T, N, K, L, R> expr)
{
    context.zeroGradients();
    vector<T, N> result = expr.forward();
    vector<T, N> seed = (vector<T, N>)0;
    if (N >= 1) seed[0] = (T)1;
    expr.backward(context, seed);
    return result;
}

template<typename T, typename E>
T compute_gradients(inout GradientContext<matrix<T, 2, 2> > context, BackDet2x2Expr<T, E> expr)
{
    context.zeroGradients();
    T result = expr.forward();
    expr.backward(context, (T)1);
    return result;
}

// Additional overload for matrix-vector multiplication with matrix context
template<typename T, int N, int K, typename L, typename R>
vector<T, N> compute_gradients(inout GradientContext<matrix<T, N, K> > context, BackMatVecMulExpr<T, N, K, L, R> expr)
{
    context.zeroGradients();
    vector<T, N> result = expr.forward();
    vector<T, N> seed = (vector<T, N>)0;
    if (N >= 1) seed[0] = (T)1;
    expr.backward(context, seed);
    return result;
}

#endif // BACKWARD_AD_HLSL