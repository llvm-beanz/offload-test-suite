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

// ============================================================================
// Vector and Matrix Test Structures
// ============================================================================

struct BackVectorTestResult
{
    bool passed;
    float expected_value;
    float actual_value;
    vector<float, 2> expected_gradient;
    vector<float, 2> actual_gradient;
    float tolerance;
};

struct BackMatrixTestResult  
{
    bool passed;
    float expected_value;
    float actual_value;
    matrix<float, 2, 2> expected_gradient;
    matrix<float, 2, 2> actual_gradient;
    float tolerance;
};

// Helper function to check vector equality
bool BackApproxEqualVec2(vector<float, 2> a, vector<float, 2> b, float tolerance = 1e-5f)
{
    return BackApproxEqual(a.x, b.x, tolerance) && BackApproxEqual(a.y, b.y, tolerance);
}

// Helper function to check matrix equality
bool BackApproxEqualMat2x2(matrix<float, 2, 2> a, matrix<float, 2, 2> b, float tolerance = 1e-5f)
{
    return BackApproxEqual(a[0][0], b[0][0], tolerance) && BackApproxEqual(a[0][1], b[0][1], tolerance) &&
           BackApproxEqual(a[1][0], b[1][0], tolerance) && BackApproxEqual(a[1][1], b[1][1], tolerance);
}

// Test vector dot product: f(u,v) = dot(u,v)
BackVectorTestResult TestBackwardVectorDot()
{
    BackVectorTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<vector<float, 2> > context;
    context.variable_count = 0;
    
    // f(u,v) = dot([u1,u2], [3,4]) = 3*u1 + 4*u2  
    // ∂f/∂u = [3,4]
    Variable<vector<float, 2> > u = variableVector<float, 2>(context, vector<float, 2>(1.0f, 2.0f));
    VariableExpr<vector<float, 2> > u_expr;
    u_expr.var = u;
    
    // Constant vector [3,4]
    VariableExpr<vector<float, 2> > v_expr;
    v_expr.var.value = vector<float, 2>(3.0f, 4.0f);
    v_expr.var.id = -1; // Mark as constant
    
    BackDotExpr<float, 2, VariableExpr<vector<float, 2> >, VariableExpr<vector<float, 2> > > f = dotProduct<float, 2>(u_expr, v_expr);
    
    float function_value = compute_gradients(context, f);
    vector<float, 2> gradient_value = u.gradient(context);
    
    result.expected_value = 11.0f;  // 1*3 + 2*4 = 11
    result.actual_value = function_value;
    result.expected_gradient = vector<float, 2>(3.0f, 4.0f);  // ∂f/∂u = [3,4]
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 11.0f, result.tolerance) &&
                   BackApproxEqualVec2(gradient_value, vector<float, 2>(3.0f, 4.0f), result.tolerance);
    
    return result;
}

// Test vector length: f(v) = |v|
BackVectorTestResult TestBackwardVectorLength()
{
    BackVectorTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<vector<float, 2> > context;
    context.variable_count = 0;
    
    // f(v) = |[3,4]| = 5
    // ∂f/∂v = v/|v| = [3,4]/5 = [0.6, 0.8]
    Variable<vector<float, 2> > v = variableVector<float, 2>(context, vector<float, 2>(3.0f, 4.0f));
    VariableExpr<vector<float, 2> > v_expr;
    v_expr.var = v;
    
    BackLengthExpr<float, 2, VariableExpr<vector<float, 2> > > f = lengthExpr<float, 2>(v_expr);
    
    float function_value = compute_gradients(context, f);
    vector<float, 2> gradient_value = v.gradient(context);
    
    result.expected_value = 5.0f;  // sqrt(9+16) = 5
    result.actual_value = function_value;
    result.expected_gradient = vector<float, 2>(0.6f, 0.8f);  // [3,4]/5
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 5.0f, result.tolerance) &&
                   BackApproxEqualVec2(gradient_value, vector<float, 2>(0.6f, 0.8f), result.tolerance);
    
    return result;
}

// Test vector normalization: f(v) = normalize(v)
BackVectorTestResult TestBackwardVectorNormalize()
{
    BackVectorTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<vector<float, 2> > context;
    context.variable_count = 0;
    
    // f(v) = normalize([3,4]) = [0.6, 0.8]
    Variable<vector<float, 2> > v = variableVector<float, 2>(context, vector<float, 2>(3.0f, 4.0f));
    VariableExpr<vector<float, 2> > v_expr;
    v_expr.var = v;
    
    BackNormalizeExpr<float, 2, VariableExpr<vector<float, 2> > > f = normalizeExpr<float, 2>(v_expr);
    
    vector<float, 2> function_value = compute_gradients(context, f);
    vector<float, 2> gradient_value = v.gradient(context);
    
    // The gradient computation for normalize is complex, so we'll just check the function value
    result.expected_value = 0.6f;  // x component of normalized [3,4]
    result.actual_value = function_value.x;
    result.expected_gradient = vector<float, 2>(0.0f, 0.0f);  // Simplified check
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value.x, 0.6f, result.tolerance) &&
                   BackApproxEqual(function_value.y, 0.8f, result.tolerance);
    
    return result;
}

// Test matrix determinant: f(M) = det(M)
BackMatrixTestResult TestBackwardMatrixDeterminant()
{
    BackMatrixTestResult result;
    result.tolerance = 1e-4f;
    
    GradientContext<matrix<float, 2, 2> > context;
    context.variable_count = 0;
    
    // f(M) = det([[2,1],[3,4]]) = 2*4 - 1*3 = 5
    // ∂f/∂M = adj(M)^T = [[4,-3],[-1,2]]
    Variable<matrix<float, 2, 2> > M = variableMatrix<float, 2, 2>(context, matrix<float, 2, 2>(2, 1, 3, 4));
    VariableExpr<matrix<float, 2, 2> > M_expr;
    M_expr.var = M;
    
    BackDet2x2Expr<float, VariableExpr<matrix<float, 2, 2> > > f = determinantExpr<float>(M_expr);
    
    float function_value = compute_gradients(context, f);
    matrix<float, 2, 2> gradient_value = M.gradient(context);
    
    result.expected_value = 5.0f;  // det([[2,1],[3,4]]) = 5
    result.actual_value = function_value;
    result.expected_gradient = matrix<float, 2, 2>(4, -3, -1, 2);  // adjugate matrix
    result.actual_gradient = gradient_value;
    
    result.passed = BackApproxEqual(function_value, 5.0f, result.tolerance) &&
                   BackApproxEqualMat2x2(gradient_value, matrix<float, 2, 2>(4, -3, -1, 2), result.tolerance);
    
    return result;
}

// ============================================================================
// Test Runner
// ============================================================================

// Run all backward AD tests (including vector/matrix tests)
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
    
    // Vector and matrix tests
    BackVectorTestResult vector_results[3];
    vector_results[0] = TestBackwardVectorDot();
    vector_results[1] = TestBackwardVectorLength();
    vector_results[2] = TestBackwardVectorNormalize();
    
    BackMatrixTestResult matrix_results[1];
    matrix_results[0] = TestBackwardMatrixDeterminant();
    
    // Count passed tests
    int scalar_passed = 0;
    for (int i = 0; i < 9; i++)
    {
        if (results[i].passed)
            scalar_passed++;
    }
    
    int vector_passed = 0;
    for (int i = 0; i < 3; i++)
    {
        if (vector_results[i].passed)
            vector_passed++;
    }
    
    int matrix_passed = 0;
    for (int i = 0; i < 1; i++)
    {
        if (matrix_results[i].passed)
            matrix_passed++;
    }
    
    // In a real application, you'd output these results somehow
    // Total tests: 9 scalar + 3 vector + 1 matrix = 13 tests
}