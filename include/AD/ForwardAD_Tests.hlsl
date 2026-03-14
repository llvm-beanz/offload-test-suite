#include "ForwardAD.hlsl"

// Test suite for Forward Automatic Differentiation

struct TestResult
{
    bool passed;
    float expected;
    float actual;
    float tolerance;
};

// Helper function to check if two floats are approximately equal
bool ApproxEqual(float a, float b, float tolerance = 1e-5f)
{
    return abs(a - b) < tolerance;
}

// Test basic arithmetic operations
export TestResult TestBasicArithmetic()
{
    TestResult result;
    result.tolerance = 1e-5f;
    
    // Test: f(x) = 2x + 3, f'(x) = 2
    // At x = 5: f(5) = 13, f'(5) = 2
    Dual<float> x = variable<float>(5.0f);
    Dual<float> f = getValue(add<float>(getValue(multiply<float>(Dual<float>::Create(2.0f, 0.0f), x)), Dual<float>::Create(3.0f, 0.0f)));
    
    result.expected = 2.0f;  // Expected derivative
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 13.0f) && 
                   ApproxEqual(f.derivative, 2.0f, result.tolerance);
    
    return result;
}

// Test quadratic function
export TestResult TestQuadratic()
{
    TestResult result;
    result.tolerance = 1e-5f;
    
    // Test: f(x) = x^2, f'(x) = 2x
    // At x = 3: f(3) = 9, f'(3) = 6
    Dual<float> x = variable<float>(3.0f);
    Dual<float> f = getValue(multiply<float>(x, x));
    
    result.expected = 6.0f;  // Expected derivative
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 9.0f) && 
                   ApproxEqual(f.derivative, 6.0f, result.tolerance);
    
    return result;
}

// Test trigonometric functions
export TestResult TestTrigonometric()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = sin(x), f'(x) = cos(x)
    // At x = 0: f(0) = 0, f'(0) = 1
    Dual<float> x = variable<float>(0.0f);
    Dual<float> f = getValue(sinExpr<float>(x));
    
    result.expected = 1.0f;  // cos(0) = 1
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 0.0f, result.tolerance) && 
                   ApproxEqual(f.derivative, 1.0f, result.tolerance);
    
    return result;
}

// Test exponential function
export TestResult TestExponential()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = exp(x), f'(x) = exp(x)
    // At x = 0: f(0) = 1, f'(0) = 1
    Dual<float> x = variable<float>(0.0f);
    Dual<float> f = getValue(expExpr<float>(x));
    
    result.expected = 1.0f;  // exp(0) = 1
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 1.0f, result.tolerance) && 
                   ApproxEqual(f.derivative, 1.0f, result.tolerance);
    
    return result;
}

// Test logarithm function
export TestResult TestLogarithm()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = log(x), f'(x) = 1/x
    // At x = 1: f(1) = 0, f'(1) = 1
    Dual<float> x = variable<float>(1.0f);
    Dual<float> f = getValue(logExpr<float>(x));
    
    result.expected = 1.0f;  // 1/1 = 1
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 0.0f, result.tolerance) && 
                   ApproxEqual(f.derivative, 1.0f, result.tolerance);
    
    return result;
}

// Test chain rule with composite function
export TestResult TestChainRule()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = sin(2x), f'(x) = 2*cos(2x)
    // At x = 0: f(0) = 0, f'(0) = 2
    Dual<float> x = variable<float>(0.0f);
    Dual<float> f = getValue(sinExpr<float>(getValue(multiply<float>(Dual<float>::Create(2.0f, 0.0f), x))));
    
    result.expected = 2.0f;  // 2*cos(0) = 2*1 = 2
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 0.0f, result.tolerance) && 
                   ApproxEqual(f.derivative, 2.0f, result.tolerance);
    
    return result;
}

// Test product rule
export TestResult TestProductRule()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = x * sin(x), f'(x) = sin(x) + x*cos(x)
    // At x = 0: f(0) = 0, f'(0) = sin(0) + 0*cos(0) = 0
    Dual<float> x = variable<float>(0.0f);
    Dual<float> f = getValue(multiply<float>(x, getValue(sinExpr<float>(x))));
    
    result.expected = 0.0f;  // sin(0) + 0*cos(0) = 0
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 0.0f, result.tolerance) && 
                   ApproxEqual(f.derivative, 0.0f, result.tolerance);
    
    return result;
}

// Test quotient rule
export TestResult TestQuotientRule()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = x / (x + 1), f'(x) = 1 / (x + 1)^2
    // At x = 0: f(0) = 0, f'(0) = 1
    Dual<float> x = variable<float>(0.0f);
    Dual<float> f = getValue(divide<float>(x, getValue(add<float>(x, Dual<float>::Create(1.0f, 0.0f)))));
    
    result.expected = 1.0f;  // 1 / (0 + 1)^2 = 1
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 0.0f, result.tolerance) && 
                   ApproxEqual(f.derivative, 1.0f, result.tolerance);
    
    return result;
}

// Test power rule
export TestResult TestPowerRule()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = x^3, f'(x) = 3*x^2
    // At x = 2: f(2) = 8, f'(2) = 12
    Dual<float> x = variable<float>(2.0f);
    Dual<float> f = getValue(power<float>(x, constant<float>(3.0f)));
    
    result.expected = 12.0f;  // 3 * 2^2 = 12
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 8.0f, result.tolerance) && 
                   ApproxEqual(f.derivative, 12.0f, result.tolerance);
    
    return result;
}

// Test complex expression
export TestResult TestComplexExpression()
{
    TestResult result;
    result.tolerance = 1e-3f;
    
    // Test: f(x) = exp(x) * sin(x) + x^2
    // f'(x) = exp(x)*sin(x) + exp(x)*cos(x) + 2x
    // At x = 0: f(0) = 0, f'(0) = 0 + 1*1 + 0 = 1
    Dual<float> x = variable<float>(0.0f);
    Dual<float> exp_sin = getValue(multiply<float>(getValue(expExpr<float>(x)), getValue(sinExpr<float>(x))));
    Dual<float> x_squared = getValue(multiply<float>(x, x));
    Dual<float> f = getValue(add<float>(exp_sin, x_squared));
    
    result.expected = 1.0f;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 0.0f, result.tolerance) && 
                   ApproxEqual(f.derivative, 1.0f, result.tolerance);
    
    return result;
}

// Forward declarations for test functions
export TestResult TestVectorDotProduct();
export TestResult TestVectorLength();
export TestResult TestVectorNormalize();
export TestResult TestMatrixMultiplication();
export TestResult TestMatrixDeterminant();

// Run all tests
export void RunAllTests()
{
    // Array of test functions and their names
    // Note: In actual HLSL, you'd need to handle this differently
    // as function pointers aren't directly supported
    
    TestResult results[14];  // Increased for new vector/matrix tests
    results[0] = TestBasicArithmetic();
    results[1] = TestQuadratic();
    results[2] = TestTrigonometric();
    results[3] = TestExponential();
    results[4] = TestLogarithm();
    results[5] = TestChainRule();
    results[6] = TestProductRule();
    results[7] = TestQuotientRule();
    results[8] = TestComplexExpression();
    results[9] = TestVectorDotProduct();
    results[10] = TestVectorLength();
    results[11] = TestVectorNormalize();
    results[12] = TestMatrixMultiplication();
    results[13] = TestMatrixDeterminant();
    
    // Count passed tests
    int passed_count = 0;
    for (int i = 0; i < 14; i++)
    {
        if (results[i].passed)
            passed_count++;
    }
    
    // In a real application, you'd output these results somehow
    // For now, they're just available for inspection
}

// ============================================================================
// Vector and Matrix Tests
// ============================================================================

// Test vector dot product differentiation
export TestResult TestVectorDotProduct()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = dot(v, u) where v = [x, 2x] and u = [1, 3]
    // f(x) = x*1 + 2x*3 = x + 6x = 7x
    // f'(x) = 7
    Dual<float> x = variable<float>(2.0f);
    Dual<vector<float, 2> > v = makeVector2<float>(x, getValue(multiply<float>(Dual<float>::Create(2.0f, 0.0f), x)));
    Dual<vector<float, 2> > u = constantVector<float, 2>(vector<float, 2>(1.0f, 3.0f));
    
    Dual<float> f = getValue(dotProduct<float, 2>(v, u));
    
    result.expected = 7.0f;  // 7x at x=2 gives derivative 7
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 14.0f, result.tolerance) &&  // 7*2 = 14
                   ApproxEqual(f.derivative, 7.0f, result.tolerance);
    
    return result;
}

// Test vector length differentiation
export TestResult TestVectorLength()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = |v| where v = [x, 0]
    // f(x) = sqrt(x^2) = |x| = x (for x > 0)
    // f'(x) = 1
    Dual<float> x = variable<float>(3.0f);
    Dual<vector<float, 2> > v = makeVector2<float>(x, constant<float>(0.0f));
    
    Dual<float> f = getValue(lengthExpr<float, 2>(v));
    
    result.expected = 1.0f;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 3.0f, result.tolerance) &&
                   ApproxEqual(f.derivative, 1.0f, result.tolerance);
    
    return result;
}

// Test vector normalization differentiation
export TestResult TestVectorNormalize()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = normalize([x, x])
    // This is more complex - testing that it compiles and produces reasonable results
    Dual<float> x = variable<float>(1.0f);
    Dual<vector<float, 2> > v = makeVector2<float>(x, x);
    
    Dual<vector<float, 2> > f = getValue(normalizeExpr<float, 2>(v));
    
    // normalize([1,1]) = [1/sqrt(2), 1/sqrt(2)]
    float expected_component = 1.0f / sqrt(2.0f);
    
    result.expected = expected_component;
    result.actual = f.value.x;
    result.passed = ApproxEqual(f.value.x, expected_component, result.tolerance) &&
                   ApproxEqual(f.value.y, expected_component, result.tolerance);
    
    return result;
}

// Test matrix multiplication differentiation
export TestResult TestMatrixMultiplication()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = A * v where A = [[x, 0], [0, 1]] and v = [1, 2]
    // Result = [x*1, 1*2] = [x, 2]
    // d/dx of result = [1, 0]
    Dual<float> x = variable<float>(3.0f);
    
    // Create matrix [[x, 0], [0, 1]]
    matrix<float, 2, 2> mat_val = matrix<float, 2, 2>(x.value, 0, 0, 1);
    matrix<float, 2, 2> mat_deriv = matrix<float, 2, 2>(x.derivative, 0, 0, 0);
    Dual<matrix<float, 2, 2> > A = Dual<matrix<float, 2, 2> >::Create(mat_val, mat_deriv);
    
    Dual<vector<float, 2> > v = constantVector<float, 2>(vector<float, 2>(1.0f, 2.0f));
    
    Dual<vector<float, 2> > f = getValue(matMul<float, 2, 2, 2>(A, v));
    
    result.expected = 1.0f;  // derivative of first component should be 1
    result.actual = f.derivative.x;
    result.passed = ApproxEqual(f.value.x, 3.0f, result.tolerance) &&  // x*1 = 3
                   ApproxEqual(f.value.y, 2.0f, result.tolerance) &&  // 1*2 = 2
                   ApproxEqual(f.derivative.x, 1.0f, result.tolerance);  // d/dx[x] = 1
    
    return result;
}

// Test matrix determinant differentiation
export TestResult TestMatrixDeterminant()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = det([[x, 1], [2, 3]]) = x*3 - 1*2 = 3x - 2
    // f'(x) = 3
    Dual<float> x = variable<float>(2.0f);
    
    // Create matrix [[x, 1], [2, 3]]
    matrix<float, 2, 2> mat_val = matrix<float, 2, 2>(x.value, 1, 2, 3);
    matrix<float, 2, 2> mat_deriv = matrix<float, 2, 2>(x.derivative, 0, 0, 0);
    Dual<matrix<float, 2, 2> > A = Dual<matrix<float, 2, 2> >::Create(mat_val, mat_deriv);
    
    Dual<float> f = getValue(determinantExpr<float, 2>(A));
    
    result.expected = 3.0f;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, 4.0f, result.tolerance) &&  // 3*2 - 2 = 4
                   ApproxEqual(f.derivative, 3.0f, result.tolerance);
    
    return result;
}