#include "AD/ForwardAD.hlsl"

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
    Dual x = variable(5.0f);
    Dual f = getValue(add(getValue(multiply(makeDual(2.0f, 0.0f), x)), makeDual(3.0f, 0.0f)));
    
    result.expected = 2.0f;  // Expected derivative
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 13.0f) && 
                   ApproxEqual(derivative(f), 2.0f, result.tolerance);
    
    return result;
}

// Test quadratic function
export TestResult TestQuadratic()
{
    TestResult result;
    result.tolerance = 1e-5f;
    
    // Test: f(x) = x^2, f'(x) = 2x
    // At x = 3: f(3) = 9, f'(3) = 6
    Dual x = variable(3.0f);
    Dual f = getValue(multiply(x, x));
    
    result.expected = 6.0f;  // Expected derivative
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 9.0f) && 
                   ApproxEqual(derivative(f), 6.0f, result.tolerance);
    
    return result;
}

// Test trigonometric functions
export TestResult TestTrigonometric()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = sin(x), f'(x) = cos(x)
    // At x = 0: f(0) = 0, f'(0) = 1
    Dual x = variable(0.0f);
    Dual f = getValue(sinExpr(x));
    
    result.expected = 1.0f;  // cos(0) = 1
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 0.0f, result.tolerance) && 
                   ApproxEqual(derivative(f), 1.0f, result.tolerance);
    
    return result;
}

// Test exponential function
export TestResult TestExponential()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = exp(x), f'(x) = exp(x)
    // At x = 0: f(0) = 1, f'(0) = 1
    Dual x = variable(0.0f);
    Dual f = getValue(expExpr(x));
    
    result.expected = 1.0f;  // exp(0) = 1
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 1.0f, result.tolerance) && 
                   ApproxEqual(derivative(f), 1.0f, result.tolerance);
    
    return result;
}

// Test logarithm function
export TestResult TestLogarithm()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = log(x), f'(x) = 1/x
    // At x = 1: f(1) = 0, f'(1) = 1
    Dual x = variable(1.0f);
    Dual f = getValue(logExpr(x));
    
    result.expected = 1.0f;  // 1/1 = 1
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 0.0f, result.tolerance) && 
                   ApproxEqual(derivative(f), 1.0f, result.tolerance);
    
    return result;
}

// Test chain rule with composite function
export TestResult TestChainRule()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = sin(2x), f'(x) = 2*cos(2x)
    // At x = 0: f(0) = 0, f'(0) = 2
    Dual x = variable(0.0f);
    Dual f = getValue(sinExpr(getValue(multiply(makeDual(2.0f, 0.0f), x))));
    
    result.expected = 2.0f;  // 2*cos(0) = 2*1 = 2
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 0.0f, result.tolerance) && 
                   ApproxEqual(derivative(f), 2.0f, result.tolerance);
    
    return result;
}

// Test product rule
export TestResult TestProductRule()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = x * sin(x), f'(x) = sin(x) + x*cos(x)
    // At x = 0: f(0) = 0, f'(0) = sin(0) + 0*cos(0) = 0
    Dual x = variable(0.0f);
    Dual f = getValue(multiply(x, getValue(sinExpr(x))));
    
    result.expected = 0.0f;  // sin(0) + 0*cos(0) = 0
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 0.0f, result.tolerance) && 
                   ApproxEqual(derivative(f), 0.0f, result.tolerance);
    
    return result;
}

// Test quotient rule
export TestResult TestQuotientRule()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = x / (x + 1), f'(x) = 1 / (x + 1)^2
    // At x = 0: f(0) = 0, f'(0) = 1
    Dual x = variable(0.0f);
    Dual f = getValue(divide(x, getValue(add(x, makeDual(1.0f, 0.0f)))));
    
    result.expected = 1.0f;  // 1 / (0 + 1)^2 = 1
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 0.0f, result.tolerance) && 
                   ApproxEqual(derivative(f), 1.0f, result.tolerance);
    
    return result;
}

// Test power rule
export TestResult TestPowerRule()
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    // Test: f(x) = x^3, f'(x) = 3*x^2
    // At x = 2: f(2) = 8, f'(2) = 12
    Dual x = variable(2.0f);
    Dual f = getValue(power(x, constant(3.0f)));
    
    result.expected = 12.0f;  // 3 * 2^2 = 12
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 8.0f, result.tolerance) && 
                   ApproxEqual(derivative(f), 12.0f, result.tolerance);
    
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
    Dual x = variable(0.0f);
    Dual exp_sin = getValue(multiply(getValue(expExpr(x)), getValue(sinExpr(x))));
    Dual x_squared = getValue(multiply(x, x));
    Dual f = getValue(add(exp_sin, x_squared));
    
    result.expected = 1.0f;
    result.actual = derivative(f);
    result.passed = ApproxEqual(value(f), 0.0f, result.tolerance) && 
                   ApproxEqual(derivative(f), 1.0f, result.tolerance);
    
    return result;
}

// Run all tests
export void RunAllTests()
{
    // Array of test functions and their names
    // Note: In actual HLSL, you'd need to handle this differently
    // as function pointers aren't directly supported
    
    TestResult results[9];
    results[0] = TestBasicArithmetic();
    results[1] = TestQuadratic();
    results[2] = TestTrigonometric();
    results[3] = TestExponential();
    results[4] = TestLogarithm();
    results[5] = TestChainRule();
    results[6] = TestProductRule();
    results[7] = TestQuotientRule();
    results[8] = TestComplexExpression();
    
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