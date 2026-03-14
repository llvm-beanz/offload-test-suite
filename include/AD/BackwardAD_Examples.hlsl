#include "BackwardAD.hlsl"

// ============================================================================
// Backward Automatic Differentiation Examples
// ============================================================================

// Example 1: Simple function f(x) = x^2, f'(x) = 2x
void ExampleSimpleQuadratic()
{
    GradientContext<float> context;
    context.variable_count = 0;
    
    Variable<float> x = variable<float>(context, 3.0f);  // x = 3
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    
    // f(x) = x^2
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > f = multiply<float>(x_expr, x_expr);
    
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
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > xy = multiply<float>(x_expr, y_expr);
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > x_squared = multiply<float>(x_expr, x_expr);
    BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> >, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > > f = add<float>(xy, x_squared);
    
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
    BackSinExpr<float, VariableExpr<float> > sin_x = sinExpr<float>(x_expr);
    BackCosExpr<float, VariableExpr<float> > cos_x = cosExpr<float>(x_expr);
    BackMulExpr<float, BackSinExpr<float, VariableExpr<float> >, BackCosExpr<float, VariableExpr<float> > > sin_cos = multiply<float>(sin_x, cos_x);
    BackExpExpr<float, VariableExpr<float> > exp_x = expExpr<float>(x_expr);
    BackAddExpr<float, BackMulExpr<float, BackSinExpr<float, VariableExpr<float> >, BackCosExpr<float, VariableExpr<float> > >, BackExpExpr<float, VariableExpr<float> > > f = add<float>(sin_cos, exp_x);
    
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
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > x_squared = multiply<float>(x_expr, x_expr);
    BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> >, float> x_sq_plus_1 = add<float>(x_squared, 1.0f);
    BackLogExpr<float, BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> >, float> > f = logExpr<float>(x_sq_plus_1);
    
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
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > x_squared = multiply<float>(x_expr, x_expr);
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > y_squared = multiply<float>(y_expr, y_expr);
    BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> >, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > > f1 = add<float>(x_squared, y_squared);
    
    // Compute f1 and its gradients
    float f1_value = compute_gradients<float>(context, f1);
    float df1_dx = x.gradient(context);
    float df1_dy = y.gradient(context);
    
    // f2(x,y) = x*y
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > f2 = multiply<float>(x_expr, y_expr);
    
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
        BackSubExpr<float, VariableExpr<float>, float> x_minus_1 = subtract<float>(x_expr, 1.0f);
        BackSubExpr<float, VariableExpr<float>, float> y_minus_2 = subtract<float>(y_expr, 2.0f);
        BackMulExpr<float, BackSubExpr<float, VariableExpr<float>, float>, BackSubExpr<float, VariableExpr<float>, float> > term1 = multiply<float>(x_minus_1, x_minus_1);
        BackMulExpr<float, BackSubExpr<float, VariableExpr<float>, float>, BackSubExpr<float, VariableExpr<float>, float> > term2 = multiply<float>(y_minus_2, y_minus_2);
        BackAddExpr<float, BackMulExpr<float, BackSubExpr<float, VariableExpr<float>, float>, BackSubExpr<float, VariableExpr<float>, float> >, BackMulExpr<float, BackSubExpr<float, VariableExpr<float>, float>, BackSubExpr<float, VariableExpr<float>, float> > > f = add<float>(term1, term2);
        
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
    BackMulExpr<float, VariableExpr<float>, float> Wx = multiply<float>(W_expr, x_input);
    BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, float>, VariableExpr<float> > y = add<float>(Wx, b_expr);
    
    // Loss: (y - target)^2
    BackSubExpr<float, BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, float>, VariableExpr<float> >, float> error = subtract<float>(y, target);
    BackMulExpr<float, BackSubExpr<float, BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, float>, VariableExpr<float> >, float>, BackSubExpr<float, BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, float>, VariableExpr<float> >, float> > loss = multiply<float>(error, error);
    
    // Compute loss and gradients
    float loss_value = compute_gradients<float>(context, loss);
    float dL_dW = W.gradient(context);  // Gradient w.r.t. weight
    float dL_db = b.gradient(context);  // Gradient w.r.t. bias
    
    // Use gradients to update parameters with gradient descent
    // W_new = W - learning_rate * dL_dW
    // b_new = b - learning_rate * dL_db
}