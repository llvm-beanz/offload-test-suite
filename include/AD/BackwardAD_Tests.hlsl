#include "BackwardAD.hlsl"

// ============================================================================
// Test Suite for Backward Automatic Differentiation
// ============================================================================

struct BackTestResult
{
    bool passed;
    float expected_value;
    float actual_value;
    float expected_gradient;
    float actual_gradient;
    float tolerance;
};

// Helper function to check if two floats are approximately equal
bool BackApproxEqual(float a, float b, float tolerance = 1e-5f)
{
    return abs(a - b) < tolerance;
}

// Test basic quadratic function f(x) = x^2
BackTestResult TestBackwardQuadratic()
{
    BackTestResult result;
    result.tolerance = 1e-5f;
    
    GradientContext<float> context; 
    context.variable_count = 0;
    
    // f(x) = x^2, f'(x) = 2x
    // At x = 3: f(3) = 9, f'(3) = 6
    Variable<float> x = variable<float>(context, 3.0f);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > f = multiply<float>(x_expr, x_expr);
    
    float function_value = compute_gradients<float>(context, f);
    float gradient_value = x.gradient(context);
    
    result.expected_value = 9.0f;
    result.actual_value = function_value;
    result.expected_gradient = 6.0f;
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 9.0f, result.tolerance) &&
                   BackApproxEqual(gradient_value, 6.0f, result.tolerance);
    
    return result;
}

// Test addition: f(x,y) = x + y
BackTestResult TestBackwardAddition()
{
    BackTestResult result;
    result.tolerance = 1e-5f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x,y) = x + y, ∂f/∂x = 1, ∂f/∂y = 1
    Variable<float> x = variable<float>(context, 2.0f);
    Variable<float> y = variable<float>(context, 3.0f);
    
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);
    
    BackAddExpr<float, VariableExpr<float>, VariableExpr<float> > f = add<float>(x_expr, y_expr);
    
    float function_value = compute_gradients<float>(context, f);
    float x_gradient = x.gradient(context);
    float y_gradient = y.gradient(context);
    
    result.expected_value = 5.0f;  // 2 + 3
    result.actual_value = function_value;
    result.expected_gradient = 1.0f;  // ∂f/∂x = 1
    result.actual_gradient = x_gradient;
    
    result.passed = BackApproxEqual(function_value, 5.0f, result.tolerance) &&
                   BackApproxEqual(x_gradient, 1.0f, result.tolerance) &&
                   BackApproxEqual(y_gradient, 1.0f, result.tolerance);
    
    return result;
}

// Test multiplication: f(x,y) = x * y
BackTestResult TestBackwardMultiplication()
{
    BackTestResult result;
    result.tolerance = 1e-5f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x,y) = x * y, ∂f/∂x = y, ∂f/∂y = x
    Variable<float> x = variable<float>(context, 4.0f);
    Variable<float> y = variable<float>(context, 5.0f);
    
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);
    
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > f = multiply<float>(x_expr, y_expr);
    
    float function_value = compute_gradients<float>(context, f);
    float x_gradient = x.gradient(context);
    float y_gradient = y.gradient(context);
    
    result.expected_value = 20.0f;  // 4 * 5
    result.actual_value = function_value;
    result.expected_gradient = 5.0f;  // ∂f/∂x = y = 5
    result.actual_gradient = x_gradient;
    
    result.passed = BackApproxEqual(function_value, 20.0f, result.tolerance) &&
                   BackApproxEqual(x_gradient, 5.0f, result.tolerance) &&
                   BackApproxEqual(y_gradient, 4.0f, result.tolerance);
    
    return result;
}

// Test division: f(x,y) = x / y
BackTestResult TestBackwardDivision()
{
    BackTestResult result;
    result.tolerance = 1e-5f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x,y) = x / y, ∂f/∂x = 1/y, ∂f/∂y = -x/y^2
    Variable<float> x = variable<float>(context, 8.0f);
    Variable<float> y = variable<float>(context, 2.0f);
    
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);
    
    BackDivExpr<float, VariableExpr<float>, VariableExpr<float> > f = divide<float>(x_expr, y_expr);
    
    float function_value = compute_gradients<float>(context, f);
    float x_gradient = x.gradient(context);
    float y_gradient = y.gradient(context);
    
    result.expected_value = 4.0f;    // 8 / 2
    result.actual_value = function_value;
    result.expected_gradient = 0.5f; // ∂f/∂x = 1/y = 1/2
    result.actual_gradient = x_gradient;
    
    // ∂f/∂y = -x/y^2 = -8/4 = -2
    float expected_y_gradient = -2.0f;
    
    result.passed = BackApproxEqual(function_value, 4.0f, result.tolerance) &&
                   BackApproxEqual(x_gradient, 0.5f, result.tolerance) &&
                   BackApproxEqual(y_gradient, expected_y_gradient, result.tolerance);
    
    return result;
}

// Test sine function: f(x) = sin(x)
BackTestResult TestBackwardSine()
{
    BackTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x) = sin(x), f'(x) = cos(x)
    // At x = 0: f(0) = 0, f'(0) = 1
    Variable<float> x = variable<float>(context, 0.0f);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    
    BackSinExpr<float, VariableExpr<float> > f = sinExpr<float>(x_expr);
    
    float function_value = compute_gradients<float>(context, f);
    float gradient_value = x.gradient(context);
    
    result.expected_value = 0.0f;  // sin(0) = 0
    result.actual_value = function_value;
    result.expected_gradient = 1.0f;  // cos(0) = 1
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 0.0f, result.tolerance) &&
                   BackApproxEqual(gradient_value, 1.0f, result.tolerance);
    
    return result;
}

// Test exponential function: f(x) = exp(x)
BackTestResult TestBackwardExponential()
{
    BackTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x) = exp(x), f'(x) = exp(x)
    // At x = 0: f(0) = 1, f'(0) = 1
    Variable<float> x = variable<float>(context, 0.0f);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    
    BackExpExpr<float, VariableExpr<float> > f = expExpr<float>(x_expr);
    
    float function_value = compute_gradients<float>(context, f);
    float gradient_value = x.gradient(context);
    
    result.expected_value = 1.0f;  // exp(0) = 1
    result.actual_value = function_value;
    result.expected_gradient = 1.0f;  // exp(0) = 1
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 1.0f, result.tolerance) &&
                   BackApproxEqual(gradient_value, 1.0f, result.tolerance);
    
    return result;
}

// Test logarithm function: f(x) = log(x)
BackTestResult TestBackwardLogarithm()
{
    BackTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x) = log(x), f'(x) = 1/x
    // At x = 1: f(1) = 0, f'(1) = 1
    Variable<float> x = variable<float>(context, 1.0f);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    
    BackLogExpr<float, VariableExpr<float> > f = logExpr<float>(x_expr);
    
    float function_value = compute_gradients<float>(context, f);
    float gradient_value = x.gradient(context);
    
    result.expected_value = 0.0f;  // log(1) = 0
    result.actual_value = function_value;
    result.expected_gradient = 1.0f;  // 1/1 = 1
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 0.0f, result.tolerance) &&
                   BackApproxEqual(gradient_value, 1.0f, result.tolerance);
    
    return result;
}

// Test chain rule: f(x) = sin(x^2)
BackTestResult TestBackwardChainRule()
{
    BackTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x) = sin(x^2), f'(x) = cos(x^2) * 2x
    // At x = 0: f(0) = sin(0) = 0, f'(0) = cos(0) * 0 = 0
    Variable<float> x = variable<float>(context, 0.0f);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > x_squared = multiply<float>(x_expr, x_expr);
    BackSinExpr<float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > > f = sinExpr<float>(x_squared);
    
    float function_value = compute_gradients<float>(context, f);
    float gradient_value = x.gradient(context);
    
    result.expected_value = 0.0f;  // sin(0) = 0
    result.actual_value = function_value;
    result.expected_gradient = 0.0f;  // cos(0) * 2*0 = 0
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 0.0f, result.tolerance) &&
                   BackApproxEqual(gradient_value, 0.0f, result.tolerance);
    
    return result;
}

// Test product rule: f(x) = x * sin(x)
BackTestResult TestBackwardProductRule()
{
    BackTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x) = x * sin(x), f'(x) = sin(x) + x * cos(x)
    // At x = 0: f(0) = 0, f'(0) = sin(0) + 0*cos(0) = 0
    Variable<float> x = variable<float>(context, 0.0f);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    
    BackSinExpr<float, VariableExpr<float> > sin_x = sinExpr<float>(x_expr);
    BackMulExpr<float, VariableExpr<float>, BackSinExpr<float, VariableExpr<float> > > f = multiply<float>(x_expr, sin_x);
    
    float function_value = compute_gradients<float>(context, f);
    float gradient_value = x.gradient(context);
    
    result.expected_value = 0.0f;  // 0 * sin(0) = 0
    result.actual_value = function_value;
    result.expected_gradient = 0.0f;  // sin(0) + 0*cos(0) = 0
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 0.0f, result.tolerance) &&
                   BackApproxEqual(gradient_value, 0.0f, result.tolerance);
    
    return result;
}

// Test complex expression: f(x,y) = x^2 + y^2 - 2*x*y
BackTestResult TestBackwardComplexExpression()
{
    BackTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<float> context;
    context.variable_count = 0;
    
    // f(x,y) = x^2 + y^2 - 2*x*y
    // ∂f/∂x = 2x - 2y, ∂f/∂y = 2y - 2x
    // At (x,y) = (3,2): f = 9 + 4 - 12 = 1, ∂f/∂x = 6-4=2, ∂f/∂y = 4-6=-2
    Variable<float> x = variable<float>(context, 3.0f);
    Variable<float> y = variable<float>(context, 2.0f);
    
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);
    
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > x_squared = multiply<float>(x_expr, x_expr);
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > y_squared = multiply<float>(y_expr, y_expr);
    BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > xy = multiply<float>(x_expr, y_expr);
    BackMulExpr<float, float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > > two_xy = multiply<float>(2.0f, xy);
    
    BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> >, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > > x2_plus_y2 = add<float>(x_squared, y_squared);
    BackSubExpr<float, BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> >, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > >, BackMulExpr<float, float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > > > f = subtract<float>(x2_plus_y2, two_xy);
    
    float function_value = compute_gradients<float>(context, f);
    float x_gradient = x.gradient(context);
    float y_gradient = y.gradient(context);
    
    result.expected_value = 1.0f;
    result.actual_value = function_value;
    result.expected_gradient = 2.0f;  // ∂f/∂x
    result.actual_gradient = x_gradient;
    
    result.passed = BackApproxEqual(function_value, 1.0f, result.tolerance) &&
                   BackApproxEqual(x_gradient, 2.0f, result.tolerance) &&
                   BackApproxEqual(y_gradient, -2.0f, result.tolerance);
    
    return result;
}

// Run all backward AD tests
void RunAllBackwardTests()
{
    BackTestResult results[9];
    results[0] = TestBackwardQuadratic();
    results[1] = TestBackwardAddition();
    results[2] = TestBackwardMultiplication();
    results[3] = TestBackwardDivision();
    results[4] = TestBackwardSine();
    results[5] = TestBackwardExponential();
    results[6] = TestBackwardLogarithm();
    results[7] = TestBackwardChainRule();
    results[8] = TestBackwardComplexExpression();
    
    // Count passed tests
    int passed_count = 0;
    for (int i = 0; i < 9; i++)
    {
        if (results[i].passed)
            passed_count++;
    }
    
    // In a real application, you'd output these results somehow
    // For now, they're just available for inspection
}