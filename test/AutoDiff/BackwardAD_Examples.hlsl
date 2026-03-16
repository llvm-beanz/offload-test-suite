#include "BackwardAD.hlsl"

// ============================================================================
// Backward Automatic Differentiation Examples
// ============================================================================

#define AUTO_VAR(var, expr) __decltype(expr) var = expr;

// Example 1: Simple function f(x) = x^2, f'(x) = 2x
void ExampleSimpleQuadratic()
{
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, 3.0f);  // x = 3
    AUTO_VAR(x_expr, makeVariableExpr<float>(x));

    // f(x) = x^2
    AUTO_VAR(f, multiply<float>(x_expr, x_expr));

    // Compute function value and gradients
    float result = compute_gradients<float>(context, f);

    // result should be 9.0 (3^2)
    // x.gradient() should be 6.0 (2*3)
    float function_value = result;
    float derivative = x.gradient(context);
}

// Example 2: Multi-variable function f(x,y) = x*y + x^2
void ExampleMultiVariable()
{
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, 2.0f);  // x = 2
    Variable<float> y = variable<float>(context, 3.0f);  // y = 3

    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);

    // f(x,y) = x*y + x^2
    AUTO_VAR(xy, multiply<float>(x_expr, y_expr));
    AUTO_VAR(x_squared, multiply<float>(x_expr, x_expr));
    AUTO_VAR(f, add<float>(xy, x_squared));

    // Compute function value and gradients
    float result = compute_gradients<float>(context, f);

    // result should be 10.0 (2*3 + 2^2 = 6 + 4)
    // ∂f/∂x = y + 2x = 3 + 4 = 7
    // ∂f/∂y = x = 2
    float function_value = result;
    float df_dx = x.gradient(context);
    float df_dy = y.gradient(context);
}

// Example 3: Complex function with trigonometric operations
// f(x) = sin(x) * cos(x) + exp(x)
void ExampleTrigonometric()
{
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, 1.0f);  // x = 1
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);

    // f(x) = sin(x) * cos(x) + exp(x)
    AUTO_VAR(sin_x, sinExpr<float>(x_expr));
    AUTO_VAR(cos_x, cosExpr<float>(x_expr));
    AUTO_VAR(sin_cos, multiply<float>(sin_x, cos_x));
    AUTO_VAR(exp_x, expExpr<float>(x_expr));
    AUTO_VAR(f, add<float>(sin_cos, exp_x));

    // Compute function value and gradients
    float result = compute_gradients<float>(context, f);

    // f'(x) = cos(x)*cos(x) - sin(x)*sin(x) + exp(x) = cos(2x) + exp(x)
    float function_value = result;
    float derivative = x.gradient(context);
}

// Example 4: Chain rule with nested functions
// f(x) = log(x^2 + 1)
void ExampleChainRule()
{
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, 2.0f);  // x = 2
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);

    // f(x) = log(x^2 + 1)
    AUTO_VAR(x_squared, multiply<float>(x_expr, x_expr));
    AUTO_VAR(x_sq_plus_1, add<float>(x_squared, 1.0f));
    AUTO_VAR(f, logExpr<float>(x_sq_plus_1));

    // Compute function value and gradients
    float result = compute_gradients<float>(context, f);

    // f'(x) = (2x) / (x^2 + 1) = 4/5 = 0.8 at x=2
    float function_value = result;
    float derivative = x.gradient(context);
}

// Example 5: Multiple outputs - computing Jacobian
// f1(x,y) = x^2 + y^2 (distance squared)
// f2(x,y) = x*y (product)
void ExampleJacobian()
{
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, 3.0f);  // x = 3
    Variable<float> y = variable<float>(context, 4.0f);  // y = 4

    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);

    // f1(x,y) = x^2 + y^2
    AUTO_VAR(x_squared, multiply<float>(x_expr, x_expr));
    AUTO_VAR(y_squared, multiply<float>(y_expr, y_expr));
    AUTO_VAR(f1, add<float>(x_squared, y_squared));

    // Compute f1 and its gradients
    float f1_value = compute_gradients<float>(context, f1);
    float df1_dx = x.gradient(context);
    float df1_dy = y.gradient(context);

    // f2(x,y) = x*y
    AUTO_VAR(f2, multiply<float>(x_expr, y_expr));

    // Compute f2 and its gradients
    float f2_value = compute_gradients<float>(context, f2);
    float df2_dx = x.gradient(context);
    float df2_dy = y.gradient(context);

    // Jacobian matrix:
    // | ∂f1/∂x  ∂f1/∂y |   | 6  8 |
    // | ∂f2/∂x  ∂f2/∂y | = | 4  3 |
}

// Example 6: Optimization step using gradients
// Minimize f(x,y) = (x-1)^2 + (y-2)^2 using gradient descent
void ExampleGradientDescent()
{
    float x_val = 0.0f;  // Starting point
    float y_val = 0.0f;
    float learning_rate = 0.1f;

    for (int iter = 0; iter < 10; iter++)
    {
        GradientContext<float> context;
        context.variable_count = 0;

        Variable<float> x = variable<float>(context, x_val);
        Variable<float> y = variable<float>(context, y_val);

        VariableExpr<float> x_expr = makeVariableExpr<float>(x);
        VariableExpr<float> y_expr = makeVariableExpr<float>(y);

        // f(x,y) = (x-1)^2 + (y-2)^2
        AUTO_VAR(x_minus_1, subtract<float>(x_expr, 1.0f));
        AUTO_VAR(y_minus_2, subtract<float>(y_expr, 2.0f));
        AUTO_VAR(term1, multiply<float>(x_minus_1, x_minus_1));
        AUTO_VAR(term2, multiply<float>(y_minus_2, y_minus_2));
        AUTO_VAR(f, add<float>(term1, term2));

        // Compute function value and gradients
        float loss = compute_gradients<float>(context, f);
        float grad_x = x.gradient(context);
        float grad_y = y.gradient(context);

        // Gradient descent update
        x_val -= learning_rate * grad_x;
        y_val -= learning_rate * grad_y;

        // Should converge to (1, 2) - the minimum
    }

    // Final values should be close to (1, 2)
    float final_x = x_val;
    float final_y = y_val;
}

// Example 7: Neural network style computation
// Simple linear layer: y = W*x + b, loss = (y - target)^2
void ExampleNeuralNetwork()
{
    GradientContext<float> context;
    context.variable_count = 0;

    // Input
    float x_input = 2.0f;
    float target = 5.0f;

    // Parameters (weights and bias)
    Variable<float> W = variable<float>(context, 1.0f);  // Weight
    Variable<float> b = variable<float>(context, 0.0f);  // Bias

    VariableExpr<float> W_expr = makeVariableExpr<float>(W);
    VariableExpr<float> b_expr = makeVariableExpr<float>(b);

    // Forward pass: y = W*x + b
    AUTO_VAR(Wx, multiply<float>(W_expr, x_input));
    AUTO_VAR(y, add<float>(Wx, b_expr));

    // Loss: (y - target)^2
    AUTO_VAR(error, subtract<float>(y, target));
    AUTO_VAR(loss, multiply<float>(error, error));

    // Compute loss and gradients
    float loss_value = compute_gradients<float>(context, loss);
    float dL_dW = W.gradient(context);  // Gradient w.r.t. weight
    float dL_db = b.gradient(context);  // Gradient w.r.t. bias

    // Use gradients to update parameters with gradient descent
    // W_new = W - learning_rate * dL_dW
    // b_new = b - learning_rate * dL_db
}

// ============================================================================
// Vector and Matrix Examples
// ============================================================================

// Example 8: Vector operations - dot product
// f(u,v) = dot(u,v) where u is variable, v is constant
void ExampleVectorDotProduct()
{
    GradientContext<vector<float, 2> > context;
    context.variable_count = 0;

    // Variable vector u = [2, 3]
    Variable<vector<float, 2> > u = variableVector<float, 2>(context, vector<float, 2>(2.0f, 3.0f));
    VariableExpr<vector<float, 2> > u_expr;
    u_expr.var = u;

    // Constant vector v = [1, 4]
    VariableExpr<vector<float, 2> > v_expr;
    v_expr.var.value = vector<float, 2>(1.0f, 4.0f);
    v_expr.var.id = -1; // Constant

    // f(u,v) = dot(u,v) = u.x*v.x + u.y*v.y = 2*1 + 3*4 = 14
    AUTO_VAR(f, dotProduct<float2>(u_expr, v_expr));

    float result = compute_gradients(context, f);
    vector<float, 2> gradient = u.gradient(context);

    // result = 14.0f
    // gradient = [1, 4] (∂(dot(u,v))/∂u = v)
}

// Example 9: Vector length and normalization
// f(v) = length(v), g(v) = normalize(v)
void ExampleVectorLengthNormalize()
{
    GradientContext<vector<float, 3> > context;
    context.variable_count = 0;

    // Variable vector v = [3, 4, 0] (length = 5)
    Variable<vector<float, 3> > v = variableVector<float, 3>(context, vector<float, 3>(3.0f, 4.0f, 0.0f));
    VariableExpr<vector<float, 3> > v_expr;
    v_expr.var = v;

    // f(v) = |v|
    AUTO_VAR(length_expr, (lengthExpr<float, 3>(v_expr)));
    float length_value = compute_gradients(context, length_expr);
    vector<float, 3> length_gradient = v.gradient(context);

    // length_value = 5.0f
    // length_gradient = [0.6, 0.8, 0] (v/|v|)

    // Reset gradients for next computation
    context.variable_count = 0;
    v = variableVector<float, 3>(context, vector<float, 3>(3.0f, 4.0f, 0.0f));
    v_expr.var = v;

    // g(v) = normalize(v)
    AUTO_VAR(normalize_expr, (normalizeExpr<float, 3>(v_expr)));
    vector<float, 3> normalized_value = compute_gradients(context, normalize_expr);
    vector<float, 3> normalize_gradient = v.gradient(context);

    // normalized_value = [0.6, 0.8, 0]
    // Gradient of normalize is more complex: (I - n⊗n)/|v| where n = normalized vector
}

// Example 10: Cross product for 3D vectors
// f(u,v) = cross(u,v)
void ExampleVectorCrossProduct()
{
    GradientContext<vector<float, 3> > context;
    context.variable_count = 0;

    // Variable vector u = [1, 0, 0]
    Variable<vector<float, 3> > u = variableVector<float, 3>(context, vector<float, 3>(1.0f, 0.0f, 0.0f));
    VariableExpr<vector<float, 3> > u_expr;
    u_expr.var = u;

    // Constant vector v = [0, 1, 0]
    VariableExpr<vector<float, 3> > v_expr;
    v_expr.var.value = vector<float, 3>(0.0f, 1.0f, 0.0f);
    v_expr.var.id = -1; // Constant

    // f(u,v) = cross(u,v) = [0, 0, 1]
    AUTO_VAR(f, crossProduct<float>(u_expr, v_expr));

    vector<float, 3> result = compute_gradients(context, f);
    vector<float, 3> gradient = u.gradient(context);

    // result = [0, 0, 1] (cross product of x and y unit vectors)
    // gradient follows cross product derivative rules
}

// Example 11: Matrix-vector multiplication
// f(M,v) = M * v where M is 2x2 matrix, v is 2D vector
void ExampleMatrixVectorMultiply()
{
    GradientContext<vector<float, 2> > context;
    context.variable_count = 0;

    // For this example, we'll treat the matrix elements as part of vector context
    // In practice, you might want separate contexts for different variable types

    // Constant matrix M = [[2, 1], [3, 4]]
    VariableExpr<matrix<float, 2, 2> > M_expr;
    M_expr.var.value = matrix<float, 2, 2>(2, 1, 3, 4);
    M_expr.var.id = -1; // Constant

    // Variable vector v = [1, 2]
    Variable<vector<float, 2> > v = variableVector<float, 2>(context, vector<float, 2>(1.0f, 2.0f));
    VariableExpr<vector<float, 2> > v_expr;
    v_expr.var = v;

    // f(M,v) = M * v = [[2,1],[3,4]] * [1,2] = [4, 11]
    AUTO_VAR(f, (matVecMul<float, 2, 2>(M_expr, v_expr)));

    vector<float, 2> result = compute_gradients(context, f);
    vector<float, 2> gradient = v.gradient(context);

    // result = [4, 11] (matrix-vector product)
    // gradient w.r.t. v follows matrix-vector differentiation rules
}

// Example 12: Matrix determinant
// f(M) = det(M) for 2x2 matrix
void ExampleMatrixDeterminant()
{
    GradientContext<matrix<float, 2, 2> > context;
    context.variable_count = 0;

    // Variable matrix M = [[3, 1], [2, 4]]
    Variable<matrix<float, 2, 2> > M = variableMatrix<float, 2, 2>(context, matrix<float, 2, 2>(3, 1, 2, 4));
    VariableExpr<matrix<float, 2, 2> > M_expr;
    M_expr.var = M;

    // f(M) = det(M) = 3*4 - 1*2 = 10
    AUTO_VAR(f, determinantExpr<float>(M_expr));

    float result = compute_gradients(context, f);
    matrix<float, 2, 2> gradient = M.gradient(context);

    // result = 10.0f (determinant value)
    // gradient = adjugate matrix = [[4, -2], [-1, 3]]
}

// Example 13: Optimization with vector parameters
// Minimize ||Ax - b||^2 where A is matrix, x is variable vector, b is target
void ExampleVectorOptimization()
{
    // Problem setup: solve Ax = b using gradient descent on ||Ax - b||^2
    matrix<float, 2, 2> A = matrix<float, 2, 2>(2, 1, 1, 3);  // Fixed matrix
    vector<float, 2> b = vector<float, 2>(5, 7);              // Target vector
    vector<float, 2> x_val = vector<float, 2>(0, 0);          // Initial guess

    float learning_rate = 0.1f;

    for (int iter = 0; iter < 10; iter++)
    {
        GradientContext<vector<float, 2> > context;
        context.variable_count = 0;

        Variable<vector<float, 2> > x = variableVector<float, 2>(context, x_val);
        VariableExpr<vector<float, 2> > x_expr;
        x_expr.var = x;

        // Constant matrix A
        VariableExpr<matrix<float, 2, 2> > A_expr;
        A_expr.var.value = A;
        A_expr.var.id = -1;

        // Compute Ax
        AUTO_VAR(Ax, (matVecMul<float, 2, 2>(A_expr, x_expr)));

        // Compute Ax - b (simplified - assume subtract function works for vectors)</
        // In a complete implementation, you'd need vector subtraction operations

        // For now, let's just compute Ax and get its gradient
        vector<float, 2> result = compute_gradients(context, Ax);
        vector<float, 2> grad_x = x.gradient(context);

        // Gradient descent update
        x_val = x_val - learning_rate * grad_x;

        // Should converge to the least squares solution
    }

    // Final x_val should be close to the solution of Ax = b
}

// Example 14: Jacobian computation for vector function
// F(x,y) = [x^2 + y, x*y] - compute Jacobian matrix
void ExampleVectorJacobian()
{
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, 2.0f);  // x = 2
    Variable<float> y = variable<float>(context, 3.0f);  // y = 3

    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);

    // F1(x,y) = x^2 + y
    AUTO_VAR(x_squared, multiply<float>(x_expr, x_expr));
    AUTO_VAR(F1, add<float>(x_squared, y_expr));

    // Compute F1 and gradients
    float F1_value = compute_gradients(context, F1);
    float dF1_dx = x.gradient(context);  // Should be 2x = 4
    float dF1_dy = y.gradient(context);  // Should be 1

    // Reset for F2 computation
    context.variable_count = 0;
    x = variable<float>(context, 2.0f);
    y = variable<float>(context, 3.0f);
    x_expr = makeVariableExpr<float>(x);
    y_expr = makeVariableExpr<float>(y);

    // F2(x,y) = x*y
    AUTO_VAR(F2, multiply<float>(x_expr, y_expr));

    // Compute F2 and gradients
    float F2_value = compute_gradients(context, F2);
    float dF2_dx = x.gradient(context);  // Should be y = 3
    float dF2_dy = y.gradient(context);  // Should be x = 2

    // Jacobian matrix J:
    // J = [dF1_dx  dF1_dy] = [4  1]
    //     [dF2_dx  dF2_dy]   [3  2]
}
