#include "BackwardAD.hlsl"

// ============================================================================
// Backward Automatic Differentiation Examples
// ============================================================================

// Example 1: Simple function f(x) = x^2, f'(x) = 2x
void ExampleSimpleQuadratic()
{
    reset_variables();
    
    Variable x = variable(3.0f);  // x = 3
    VariableExpr x_expr = makeVariableExpr(x);
    
    // f(x) = x^2
    BackMulExpr<VariableExpr, VariableExpr> f = multiply(x_expr, x_expr);
    
    // Compute function value and gradients
    float result = compute_gradients(f);
    
    // result should be 9.0 (3^2)
    // x.gradient() should be 6.0 (2*3)
    float function_value = result;
    float derivative = x.gradient();
}

// Example 2: Multi-variable function f(x,y) = x*y + x^2
void ExampleMultiVariable()
{
    reset_variables();
    
    Variable x = variable(2.0f);  // x = 2
    Variable y = variable(3.0f);  // y = 3
    
    VariableExpr x_expr = makeVariableExpr(x);
    VariableExpr y_expr = makeVariableExpr(y);
    
    // f(x,y) = x*y + x^2
    BackMulExpr<VariableExpr, VariableExpr> xy = multiply(x_expr, y_expr);
    BackMulExpr<VariableExpr, VariableExpr> x_squared = multiply(x_expr, x_expr);
    BackAddExpr<BackMulExpr<VariableExpr, VariableExpr>, BackMulExpr<VariableExpr, VariableExpr> > f = add(xy, x_squared);
    
    // Compute function value and gradients
    float result = compute_gradients(f);
    
    // result should be 10.0 (2*3 + 2^2 = 6 + 4)
    // ∂f/∂x = y + 2x = 3 + 4 = 7
    // ∂f/∂y = x = 2
    float function_value = result;
    float df_dx = x.gradient();
    float df_dy = y.gradient();
}

// Example 3: Complex function with trigonometric operations
// f(x) = sin(x) * cos(x) + exp(x)
void ExampleTrigonometric()
{
    reset_variables();
    
    Variable x = variable(1.0f);  // x = 1
    VariableExpr x_expr = makeVariableExpr(x);
    
    // f(x) = sin(x) * cos(x) + exp(x)
    BackSinExpr<VariableExpr> sin_x = sinExpr(x_expr);
    BackCosExpr<VariableExpr> cos_x = cosExpr(x_expr);
    BackMulExpr<BackSinExpr<VariableExpr>, BackCosExpr<VariableExpr> > sin_cos = multiply(sin_x, cos_x);
    BackExpExpr<VariableExpr> exp_x = expExpr(x_expr);
    BackAddExpr<BackMulExpr<BackSinExpr<VariableExpr>, BackCosExpr<VariableExpr> >, BackExpExpr<VariableExpr> > f = add(sin_cos, exp_x);
    
    // Compute function value and gradients
    float result = compute_gradients(f);
    
    // f'(x) = cos(x)*cos(x) - sin(x)*sin(x) + exp(x) = cos(2x) + exp(x)
    float function_value = result;
    float derivative = x.gradient();
}

// Example 4: Chain rule with nested functions
// f(x) = log(x^2 + 1)
void ExampleChainRule()
{
    reset_variables();
    
    Variable x = variable(2.0f);  // x = 2
    VariableExpr x_expr = makeVariableExpr(x);
    
    // f(x) = log(x^2 + 1)
    BackMulExpr<VariableExpr, VariableExpr> x_squared = multiply(x_expr, x_expr);
    BackAddExpr<BackMulExpr<VariableExpr, VariableExpr>, float> x_sq_plus_1 = add(x_squared, 1.0f);
    BackLogExpr<BackAddExpr<BackMulExpr<VariableExpr, VariableExpr>, float>> f = logExpr(x_sq_plus_1);
    
    // Compute function value and gradients
    float result = compute_gradients(f);
    
    // f'(x) = (2x) / (x^2 + 1) = 4/5 = 0.8 at x=2
    float function_value = result;
    float derivative = x.gradient();
}

// Example 5: Multiple outputs - computing Jacobian
// f1(x,y) = x^2 + y^2 (distance squared)
// f2(x,y) = x*y (product)
void ExampleJacobian()
{
    reset_variables();
    
    Variable x = variable(3.0f);  // x = 3
    Variable y = variable(4.0f);  // y = 4
    
    VariableExpr x_expr = makeVariableExpr(x);
    VariableExpr y_expr = makeVariableExpr(y);
    
    // f1(x,y) = x^2 + y^2
    BackMulExpr<VariableExpr, VariableExpr> x_squared = multiply(x_expr, x_expr);
    BackMulExpr<VariableExpr, VariableExpr> y_squared = multiply(y_expr, y_expr);
    BackAddExpr<BackMulExpr<VariableExpr, VariableExpr>, BackMulExpr<VariableExpr, VariableExpr>> f1 = add(x_squared, y_squared);
    
    // Compute f1 and its gradients
    float f1_value = compute_gradients(f1);
    float df1_dx = x.gradient();
    float df1_dy = y.gradient();
    
    // f2(x,y) = x*y
    BackMulExpr<VariableExpr, VariableExpr> f2 = multiply(x_expr, y_expr);
    
    // Compute f2 and its gradients
    float f2_value = compute_gradients(f2);
    float df2_dx = x.gradient();
    float df2_dy = y.gradient();
    
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
        reset_variables();
        
        Variable x = variable(x_val);
        Variable y = variable(y_val);
        
        VariableExpr x_expr = makeVariableExpr(x);
        VariableExpr y_expr = makeVariableExpr(y);
        
        // f(x,y) = (x-1)^2 + (y-2)^2
        BackSubExpr<VariableExpr, float> x_minus_1 = subtract(x_expr, 1.0f);
        BackSubExpr<VariableExpr, float> y_minus_2 = subtract(y_expr, 2.0f);
        BackMulExpr<BackSubExpr<VariableExpr, float>, BackSubExpr<VariableExpr, float>> term1 = multiply(x_minus_1, x_minus_1);
        BackMulExpr<BackSubExpr<VariableExpr, float>, BackSubExpr<VariableExpr, float>> term2 = multiply(y_minus_2, y_minus_2);
        BackAddExpr<BackMulExpr<BackSubExpr<VariableExpr, float>, BackSubExpr<VariableExpr, float>>, BackMulExpr<BackSubExpr<VariableExpr, float>, BackSubExpr<VariableExpr, float>>> f = add(term1, term2);
        
        // Compute function value and gradients
        float loss = compute_gradients(f);
        float grad_x = x.gradient();
        float grad_y = y.gradient();
        
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
    reset_variables();
    
    // Input
    float x_input = 2.0f;
    float target = 5.0f;
    
    // Parameters (weights and bias)
    Variable W = variable(1.0f);  // Weight
    Variable b = variable(0.0f);  // Bias
    
    VariableExpr W_expr = makeVariableExpr(W);
    VariableExpr b_expr = makeVariableExpr(b);
    
    // Forward pass: y = W*x + b
    BackMulExpr<VariableExpr, float> Wx = multiply(W_expr, x_input);
    BackAddExpr<BackMulExpr<VariableExpr, float>, VariableExpr> y = add(Wx, b_expr);
    
    // Loss: (y - target)^2
    BackSubExpr<BackAddExpr<BackMulExpr<VariableExpr, float>, VariableExpr>, float> error = subtract(y, target);
    BackMulExpr<BackSubExpr<BackAddExpr<BackMulExpr<VariableExpr, float>, VariableExpr>, float>, BackSubExpr<BackAddExpr<BackMulExpr<VariableExpr, float>, VariableExpr>, float>> loss = multiply(error, error);
    
    // Compute loss and gradients
    float loss_value = compute_gradients(loss);
    float dL_dW = W.gradient();  // Gradient w.r.t. weight
    float dL_db = b.gradient();  // Gradient w.r.t. bias
    
    // Use gradients to update parameters with gradient descent
    // W_new = W - learning_rate * dL_dW
    // b_new = b - learning_rate * dL_db
}