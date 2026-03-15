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

// Test basic arithmetic operations: f(x) = 2x + 3, f'(x) = 2
TestResult TestBasicArithmetic(float input_x)
{
    TestResult result;
    result.tolerance = 1e-5f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(add<float>(getValue(multiply<float>(Dual<float>::Create(2.0f, 0.0f), x)), Dual<float>::Create(3.0f, 0.0f)));
    
    float expected_value = 2.0f * input_x + 3.0f;
    float expected_derivative = 2.0f;
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test quadratic function: f(x) = x^2, f'(x) = 2x
TestResult TestQuadratic(float input_x)
{
    TestResult result;
    result.tolerance = 1e-5f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(multiply<float>(x, x));
    
    float expected_value = input_x * input_x;
    float expected_derivative = 2.0f * input_x;
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test trigonometric functions: f(x) = sin(x), f'(x) = cos(x)
TestResult TestTrigonometric(float input_x)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(sinExpr<float>(x));
    
    float expected_value = sin(input_x);
    float expected_derivative = cos(input_x);
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test exponential function: f(x) = exp(x), f'(x) = exp(x)
TestResult TestExponential(float input_x)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(expExpr<float>(x));
    
    float expected_value = exp(input_x);
    float expected_derivative = exp(input_x);
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test logarithm function: f(x) = log(x), f'(x) = 1/x
TestResult TestLogarithm(float input_x)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(logExpr<float>(x));
    
    float expected_value = log(input_x);
    float expected_derivative = 1.0f / input_x;
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test chain rule with composite function: f(x) = sin(2x), f'(x) = 2*cos(2x)
TestResult TestChainRule(float input_x)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(sinExpr<float>(getValue(multiply<float>(Dual<float>::Create(2.0f, 0.0f), x))));
    
    float expected_value = sin(2.0f * input_x);
    float expected_derivative = 2.0f * cos(2.0f * input_x);
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test product rule: f(x) = x * sin(x), f'(x) = sin(x) + x*cos(x)
TestResult TestProductRule(float input_x)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(multiply<float>(x, getValue(sinExpr<float>(x))));
    
    float expected_value = input_x * sin(input_x);
    float expected_derivative = sin(input_x) + input_x * cos(input_x);
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test quotient rule: f(x) = x / (x + 1), f'(x) = 1 / (x + 1)^2
TestResult TestQuotientRule(float input_x)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(divide<float>(x, getValue(add<float>(x, Dual<float>::Create(1.0f, 0.0f)))));
    
    float denominator = input_x + 1.0f;
    float expected_value = input_x / denominator;
    float expected_derivative = 1.0f / (denominator * denominator);
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test power rule: f(x) = x^exponent, f'(x) = exponent*x^(exponent-1)
TestResult TestPowerRule(float input_x, float exponent)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> f = getValue(power<float>(x, constant<float>(exponent)));
    
    float expected_value = pow(input_x, exponent);
    float expected_derivative = exponent * pow(input_x, exponent - 1.0f);
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test complex expression: f(x) = exp(x) * sin(x) + x^2
// f'(x) = exp(x)*sin(x) + exp(x)*cos(x) + 2x
TestResult TestComplexExpression(float input_x)
{
    TestResult result;
    result.tolerance = 1e-3f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<float> exp_sin = getValue(multiply<float>(getValue(expExpr<float>(x)), getValue(sinExpr<float>(x))));
    Dual<float> x_squared = getValue(multiply<float>(x, x));
    Dual<float> f = getValue(add<float>(exp_sin, x_squared));
    
    float exp_x = exp(input_x);
    float sin_x = sin(input_x);
    float cos_x = cos(input_x);
    float expected_value = exp_x * sin_x + input_x * input_x;
    float expected_derivative = exp_x * sin_x + exp_x * cos_x + 2.0f * input_x;
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) && 
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// ============================================================================
// Vector and Matrix Tests
// ============================================================================

// Test vector dot product differentiation
// f(x) = dot(v, u) where v = [x, scalar*x] and u is constant
TestResult TestVectorDotProduct(float input_x, float scalar, vector<float, 2> u)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<vector<float, 2> > v = makeVector<float>(x, getValue(multiply<float>(Dual<float>::Create(scalar, 0.0f), x)));
    Dual<vector<float, 2> > u_dual = constantVector<float, 2>(u);
    
    Dual<float> f = getValue(dotProduct<float, 2>(v, u_dual));
    
    // f(x) = x*u.x + scalar*x*u.y = x*(u.x + scalar*u.y)
    // f'(x) = u.x + scalar*u.y
    float expected_value = input_x * u.x + scalar * input_x * u.y;
    float expected_derivative = u.x + scalar * u.y;
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) &&
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test vector length differentiation
// f(x) = |v| where v = [x, constant]
TestResult TestVectorLength(float input_x, float constant_component)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<vector<float, 2> > v = makeVector<float>(x, constant<float>(constant_component));
    
    Dual<float> f = getValue(lengthExpr<float, 2>(v));
    
    // f(x) = sqrt(x^2 + constant^2)
    // f'(x) = x / sqrt(x^2 + constant^2)
    float length_val = sqrt(input_x * input_x + constant_component * constant_component);
    float expected_value = length_val;
    float expected_derivative = input_x / length_val;
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) &&
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Test vector normalization differentiation
// f(x) = normalize([x, x]) - testing that it compiles and produces reasonable results
TestResult TestVectorNormalize(float input_x)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    Dual<vector<float, 2> > v = makeVector<float>(x, x);
    
    Dual<vector<float, 2> > f = getValue(normalizeExpr<float, 2>(v));
    
    // normalize([x,x]) = [x/sqrt(2x^2), x/sqrt(2x^2)] = [x/(|x|*sqrt(2)), x/(|x|*sqrt(2))]
    // For x > 0: normalize([x,x]) = [1/sqrt(2), 1/sqrt(2)]
    float expected_component = (input_x >= 0.0f ? 1.0f : -1.0f) / sqrt(2.0f);
    
    result.expected = expected_component;
    result.actual = f.value.x;
    result.passed = ApproxEqual(f.value.x, expected_component, result.tolerance) &&
                   ApproxEqual(f.value.y, expected_component, result.tolerance);
    
    return result;
}

// Test matrix multiplication differentiation
// f(x) = A * v where A = [[x, 0], [0, 1]] and v is constant
TestResult TestMatrixMultiplication(float input_x, vector<float, 2> v_const)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    
    // Create matrix [[x, 0], [0, 1]]
    matrix<float, 2, 2> mat_val = matrix<float, 2, 2>(x.value, 0, 0, 1);
    matrix<float, 2, 2> mat_deriv = matrix<float, 2, 2>(x.derivative, 0, 0, 0);
    Dual<matrix<float, 2, 2> > A = Dual<matrix<float, 2, 2> >::Create(mat_val, mat_deriv);
    
    Dual<vector<float, 2> > v = constantVector<float, 2>(v_const);
    
    Dual<vector<float, 2> > f = getValue(matMul<float, 2, 2, 2>(A, v));
    
    // Result = [x*v.x, 1*v.y] = [x*v.x, v.y]
    // d/dx of result = [v.x, 0]
    result.expected = v_const.x;  // derivative of first component should be v.x
    result.actual = f.derivative.x;
    result.passed = ApproxEqual(f.value.x, input_x * v_const.x, result.tolerance) &&
                   ApproxEqual(f.value.y, v_const.y, result.tolerance) &&
                   ApproxEqual(f.derivative.x, v_const.x, result.tolerance);
    
    return result;
}

// Test matrix determinant differentiation  
// f(x) = det([[x, a], [b, c]]) = x*c - a*b
// f'(x) = c
TestResult TestMatrixDeterminant(float input_x, float a, float b, float c)
{
    TestResult result;
    result.tolerance = 1e-4f;
    
    Dual<float> x = variable<float>(input_x);
    
    // Create matrix [[x, a], [b, c]]
    matrix<float, 2, 2> mat_val = matrix<float, 2, 2>(x.value, a, b, c);
    matrix<float, 2, 2> mat_deriv = matrix<float, 2, 2>(x.derivative, 0, 0, 0);
    Dual<matrix<float, 2, 2> > A = Dual<matrix<float, 2, 2> >::Create(mat_val, mat_deriv);
    
    Dual<float> f = getValue(determinantExpr<float, 2>(A));
    
    float expected_value = input_x * c - a * b;
    float expected_derivative = c;
    
    result.expected = expected_derivative;
    result.actual = f.derivative;
    result.passed = ApproxEqual(f.value, expected_value, result.tolerance) &&
                   ApproxEqual(f.derivative, expected_derivative, result.tolerance);
    
    return result;
}

// Run all tests with the same input values as the original hardcoded tests
void RunAllTests()
{
    // Array of test functions and their names
    // Note: In actual HLSL, you'd need to handle this differently
    // as function pointers aren't directly supported
    
    TestResult results[15];  // Updated for all tests including matrix determinant
    
    // Use the same input values as the original hardcoded tests
    results[0] = TestBasicArithmetic(5.0f);  // Original: x = 5.0f
    results[1] = TestQuadratic(3.0f);        // Original: x = 3.0f
    results[2] = TestTrigonometric(0.0f);    // Original: x = 0.0f
    results[3] = TestExponential(0.0f);      // Original: x = 0.0f
    results[4] = TestLogarithm(1.0f);        // Original: x = 1.0f
    results[5] = TestChainRule(0.0f);        // Original: x = 0.0f
    results[6] = TestProductRule(0.0f);      // Original: x = 0.0f
    results[7] = TestQuotientRule(0.0f);     // Original: x = 0.0f
    results[8] = TestPowerRule(2.0f, 3.0f);  // Original: x = 2.0f, exponent = 3.0f
    results[9] = TestComplexExpression(0.0f); // Original: x = 0.0f
    
    // Vector and matrix tests with original values
    results[10] = TestVectorDotProduct(2.0f, 2.0f, vector<float, 2>(1.0f, 3.0f)); // Original: x=2, v=[x,2x], u=[1,3]
    results[11] = TestVectorLength(3.0f, 0.0f);  // Original: x = 3.0f, v = [x, 0]
    results[12] = TestVectorNormalize(1.0f);     // Original: x = 1.0f
    results[13] = TestMatrixMultiplication(3.0f, vector<float, 2>(1.0f, 2.0f)); // Original: x=3, v=[1,2]
    results[14] = TestMatrixDeterminant(2.0f, 1.0f, 2.0f, 3.0f); // Original: x=2, matrix=[[x,1],[2,3]]
    
    // Count passed tests
    int passed_count = 0;
    for (int i = 0; i < 15; i++)
    {
        if (results[i].passed)
            passed_count++;
    }
    
    // In a real application, you'd output these results somehow
    // For now, they're just available for inspection
}