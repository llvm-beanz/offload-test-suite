#include "ForwardAD.hlsl"

using namespace ad::fwd;

// Test suite for Forward Automatic Differentiation
// Each test function returns its computed result for CPU verification

// Structure to store individual test results
struct ADTestResult
{
    float value;
    float derivative;
};

// Structure to store all test input values (provided by CPU)
struct ADTestInputs
{
    // Basic scalar inputs
    float basic_arithmetic_x;
    float quadratic_x;
    float trigonometric_x;
    float exponential_x;
    float logarithm_x;
    float chain_rule_x;
    float product_rule_x;
    float quotient_rule_x;
    float power_rule_x;
    float power_rule_exponent;
    float complex_expression_x;

    // Vector test inputs
    float vector_dot_x;
    float vector_dot_scalar;
    float2 vector_dot_u;
    float vector_length_x;
    float vector_length_constant;
    float vector_normalize_x;

    // Matrix test inputs
    float matrix_mult_x;
    float2 matrix_mult_v;
    float matrix_det_x;
    float matrix_det_a;
    float matrix_det_b;
    float matrix_det_c;
};

// Input buffer to receive test parameters from CPU
StructuredBuffer<ADTestInputs> InputBuffer : register(t0);

// Structured buffer to store test results for CPU readback
RWStructuredBuffer<ADTestResult> ResultBuffer : register(u0);

// Test basic arithmetic operations: f(x) = 2x + 3, f'(x) = 2
Value<float> TestBasicArithmetic(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = x * 2.0f + 3.0f;
    return f;
}

// Test quadratic function: f(x) = x^2, f'(x) = 2x
Value<float> TestQuadratic(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = x * x;
    return f;
}

// Test trigonometric functions: f(x) = sin(x), f'(x) = cos(x)
Value<float> TestTrigonometric(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = sin(x);
    return f;
}

// Test exponential function: f(x) = exp(x), f'(x) = exp(x)
Value<float> TestExponential(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = exp(x);
    return f;
}

// Test logarithm function: f(x) = log(x), f'(x) = 1/x
Value<float> TestLogarithm(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = log(x);
    return f;
}

// Test chain rule with composite function: f(x) = sin(2x), f'(x) = 2*cos(2x)
Value<float> TestChainRule(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = sin(x * 2.0f);
    return f;
}

// Test product rule: f(x) = x * sin(x), f'(x) = sin(x) + x*cos(x)
Value<float> TestProductRule(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = x * sin(x);
    return f;
}

// Test quotient rule: f(x) = x / (x + 1), f'(x) = 1 / (x + 1)^2
Value<float> TestQuotientRule(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = x / (x + 1.0f);
    return f;
}

// Test power rule: f(x) = x^exponent, f'(x) = exponent*x^(exponent-1)
Value<float> TestPowerRule(float input_x, float exponent)
{
    Value<float> x = variable<float>(input_x);
    Value<float> f = pow(x, constant<float>(exponent));
    return f;
}

// Test complex expression: f(x) = exp(x) * sin(x) + x^2
// f'(x) = exp(x)*sin(x) + exp(x)*cos(x) + 2x
Value<float> TestComplexExpression(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<float> exp_sin = exp(x) * sin(x);
    Value<float> x_squared = x * x;
    Value<float> f = exp_sin + x_squared;
    return f;
}

// ============================================================================
// Vector and Matrix Tests
// ============================================================================

// Test vector dot product differentiation
// f(x) = dot(v, u) where v = [x, scalar*x] and u is constant
Value<float> TestVectorDotProduct(float input_x, float scalar, vector<float, 2> u)
{
    Value<float> x = variable<float>(input_x);
    Value<vector<float, 2> > v = makeVector<float>(x, x * scalar);
    Value<vector<float, 2> > u_val = constantVector<float, 2>(u);
    Value<float> f = dot(v, u_val);
    return f;
}

// Test vector length differentiation
// f(x) = |v| where v = [x, constant]
Value<float> TestVectorLength(float input_x, float constant_component)
{
    Value<float> x = variable<float>(input_x);
    Value<vector<float, 2> > v = makeVector<float>(x, constant<float>(constant_component));
    Value<float> f = length(v);
    return f;
}

// Test vector normalization differentiation
// f(x) = normalize([x, x]) - testing that it compiles and produces reasonable results
Value<vector<float, 2> > TestVectorNormalize(float input_x)
{
    Value<float> x = variable<float>(input_x);
    Value<vector<float, 2> > v = makeVector<float>(x, x);
    Value<vector<float, 2> > f = normalize(v);
    return f;
}

// Test matrix multiplication differentiation
// f(x) = A * v where A = [[x, 0], [0, 1]] and v is constant
Value<vector<float, 2> > TestMatrixMultiplication(float input_x, vector<float, 2> v_const)
{
    Value<float> x = variable<float>(input_x);

    // Create matrix [[x, 0], [0, 1]]
    matrix<float, 2, 2> mat_val = matrix<float, 2, 2>(x.value, 0, 0, 1);
    matrix<float, 2, 2> mat_deriv = matrix<float, 2, 2>(x.derivative, 0, 0, 0);
    Value<matrix<float, 2, 2> > A = Value<matrix<float, 2, 2> >::Create(mat_val, mat_deriv);

    Value<vector<float, 2> > v = constantVector<float, 2>(v_const);
    Value<vector<float, 2> > f = matMul<float, 2, 2, 2>(A, v).eval();
    return f;
}

// Test matrix determinant differentiation
// f(x) = det([[x, a], [b, c]]) = x*c - a*b
// f'(x) = c
Value<float> TestMatrixDeterminant(float input_x, float a, float b, float c)
{
    Value<float> x = variable<float>(input_x);

    // Create matrix [[x, a], [b, c]]
    matrix<float, 2, 2> mat_val = matrix<float, 2, 2>(x.value, a, b, c);
    matrix<float, 2, 2> mat_deriv = matrix<float, 2, 2>(x.derivative, 0, 0, 0);
    Value<matrix<float, 2, 2> > A = Value<matrix<float, 2, 2> >::Create(mat_val, mat_deriv);

    Value<float> f = determinantExpr<float, 2>(A).eval();
    return f;
}

// Compute shader entry point to run all tests and write results to buffer
[numthreads(8, 1, 1)]
void RunAllTests(uint3 DispatchThreadID : SV_DispatchThreadID)
{
    uint threadIndex = DispatchThreadID.x;

    // Read test inputs from CPU-provided buffer for this thread
    ADTestInputs inputs = InputBuffer[threadIndex];

    // Collect all test results in local array
    ADTestResult results[15];

    // Test 0: Basic Arithmetic - f(x) = 2x + 3
    {
        Value<float> result = TestBasicArithmetic(inputs.basic_arithmetic_x);
        results[0].value = result.value;
        results[0].derivative = result.derivative;
    }

    // Test 1: Quadratic - f(x) = x^2
    {
        Value<float> result = TestQuadratic(inputs.quadratic_x);
        results[1].value = result.value;
        results[1].derivative = result.derivative;
    }

    // Test 2: Trigonometric - f(x) = sin(x)
    {
        Value<float> result = TestTrigonometric(inputs.trigonometric_x);
        results[2].value = result.value;
        results[2].derivative = result.derivative;
    }

    // Test 3: Exponential - f(x) = exp(x)
    {
        Value<float> result = TestExponential(inputs.exponential_x);
        results[3].value = result.value;
        results[3].derivative = result.derivative;
    }

    // Test 4: Logarithm - f(x) = log(x)
    {
        Value<float> result = TestLogarithm(inputs.logarithm_x);
        results[4].value = result.value;
        results[4].derivative = result.derivative;
    }

    // Test 5: Chain Rule - f(x) = sin(2x)
    {
        Value<float> result = TestChainRule(inputs.chain_rule_x);
        results[5].value = result.value;
        results[5].derivative = result.derivative;
    }

    // Test 6: Product Rule - f(x) = x * sin(x)
    {
        Value<float> result = TestProductRule(inputs.product_rule_x);
        results[6].value = result.value;
        results[6].derivative = result.derivative;
    }

    // Test 7: Quotient Rule - f(x) = x / (x + 1)
    {
        Value<float> result = TestQuotientRule(inputs.quotient_rule_x);
        results[7].value = result.value;
        results[7].derivative = result.derivative;
    }

    // Test 8: Power Rule - f(x) = x^exponent
    {
        Value<float> result = TestPowerRule(inputs.power_rule_x, inputs.power_rule_exponent);
        results[8].value = result.value;
        results[8].derivative = result.derivative;
    }

    // Test 9: Complex Expression - f(x) = exp(x) * sin(x) + x^2
    {
        Value<float> result = TestComplexExpression(inputs.complex_expression_x);
        results[9].value = result.value;
        results[9].derivative = result.derivative;
    }

    // Test 10: Vector Dot Product
    {
        Value<float> result = TestVectorDotProduct(inputs.vector_dot_x, inputs.vector_dot_scalar, inputs.vector_dot_u);
        results[10].value = result.value;
        results[10].derivative = result.derivative;
    }

    // Test 11: Vector Length
    {
        Value<float> result = TestVectorLength(inputs.vector_length_x, inputs.vector_length_constant);
        results[11].value = result.value;
        results[11].derivative = result.derivative;
    }

    // Test 12: Vector Normalize (store x component only for simplicity)
    {
        Value<vector<float, 2> > result = TestVectorNormalize(inputs.vector_normalize_x);
        results[12].value = result.value.x;
        results[12].derivative = result.derivative.x;
    }

    // Test 13: Matrix Multiplication (store x component only for simplicity)
    {
        Value<vector<float, 2> > result = TestMatrixMultiplication(inputs.matrix_mult_x, inputs.matrix_mult_v);
        results[13].value = result.value.x;
        results[13].derivative = result.derivative.x;
    }

    // Test 14: Matrix Determinant
    {
        Value<float> result = TestMatrixDeterminant(inputs.matrix_det_x, inputs.matrix_det_a, inputs.matrix_det_b, inputs.matrix_det_c);
        results[14].value = result.value;
        results[14].derivative = result.derivative;
    }

    // Write all results to structured buffer at thread-specific indices
    uint baseIndex = threadIndex * 15;
    for (int i = 0; i < 15; i++)
    {
        ResultBuffer[baseIndex + i] = results[i];
    }
}
