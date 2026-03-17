#include "BackwardAD.hlsl"

// ============================================================================
// Test Suite for Backward Automatic Differentiation
// ============================================================================

// Structure to store individual test results
struct BackADTestResult
{
    float value;
    float gradient;
};

// Structure to store all test input values (provided by CPU)
struct BackADTestInputs
{
    // Single-variable tests
    float quadratic_x;
    float sine_x;
    float exponential_x;
    float logarithm_x;
    float chain_rule_x;
    float product_rule_x;

    // Two-variable tests
    float addition_x;
    float addition_y;
    float multiplication_x;
    float multiplication_y;
    float division_x;
    float division_y;
    float complex_x;
    float complex_y;

    // Vector tests
    float vector_dot_ux;
    float vector_dot_uy;
    float vector_dot_vx;
    float vector_dot_vy;
    float vector_length_vx;
    float vector_length_vy;

    // Matrix test
    float matrix_det_a;
    float matrix_det_b;
    float matrix_det_c;
    float matrix_det_d;

    // Additional single-variable tests
    float power_base;
    float power_exponent;
    float negate_x;
    float cosine_x;
    float sqrt_x;

    // Additional vector tests
    float normalize_vx;
    float normalize_vy;
    float cross_ux;
    float cross_uy;
    float cross_uz;
    float cross_vx;
    float cross_vy;
    float cross_vz;
    float matvec_m00;
    float matvec_m01;
    float matvec_m10;
    float matvec_m11;
    float matvec_vx;
    float matvec_vy;
};

// Input buffer to receive test parameters from CPU
StructuredBuffer<BackADTestInputs> InputBuffer : register(t0);

// Structured buffer to store test results for CPU readback
RWStructuredBuffer<BackADTestResult> ResultBuffer : register(u0);

// Test 0: f(x) = x^2, f'(x) = 2x
BackADTestResult TestBackwardQuadratic(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(f, multiply<float>(x_expr, x_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 1: f(x) = sin(x), f'(x) = cos(x)
BackADTestResult TestBackwardSine(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(f, sinExpr<float>(x_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 2: f(x) = exp(x), f'(x) = exp(x)
BackADTestResult TestBackwardExponential(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(f, expExpr<float>(x_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 3: f(x) = log(x), f'(x) = 1/x
BackADTestResult TestBackwardLogarithm(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(f, logExpr<float>(x_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 4: f(x) = sin(x^2), f'(x) = 2x*cos(x^2)
BackADTestResult TestBackwardChainRule(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(x_squared, multiply<float>(x_expr, x_expr));
    AUTO_VAR(f, sinExpr<float>(x_squared));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 5: f(x) = x*sin(x), f'(x) = sin(x) + x*cos(x)
BackADTestResult TestBackwardProductRule(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(sin_x, sinExpr<float>(x_expr));
    AUTO_VAR(f, multiply<float>(x_expr, sin_x));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 6: f(x,y) = x + y, df/dx = 1
BackADTestResult TestBackwardAddition(float input_x, float input_y)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    Variable<float> y = variable<float>(context, input_y);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);
    AUTO_VAR(f, add<float>(x_expr, y_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 7: f(x,y) = x * y, df/dx = y
BackADTestResult TestBackwardMultiplication(float input_x, float input_y)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    Variable<float> y = variable<float>(context, input_y);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);
    AUTO_VAR(f, multiply<float>(x_expr, y_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 8: f(x,y) = x / y, df/dx = 1/y
BackADTestResult TestBackwardDivision(float input_x, float input_y)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    Variable<float> y = variable<float>(context, input_y);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);
    AUTO_VAR(f, divide<float>(x_expr, y_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 9: f(x,y) = x^2 + y^2 - 2xy, df/dx = 2x - 2y
BackADTestResult TestBackwardComplexExpression(float input_x, float input_y)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    Variable<float> y = variable<float>(context, input_y);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    VariableExpr<float> y_expr = makeVariableExpr<float>(y);

    AUTO_VAR(x_squared, multiply<float>(x_expr, x_expr));
    AUTO_VAR(y_squared, multiply<float>(y_expr, y_expr));
    AUTO_VAR(xy, multiply<float>(x_expr, y_expr));
    AUTO_VAR(two_xy, multiply<float>(2.0f, xy));

    AUTO_VAR(x2_plus_y2, add<float>(x_squared, y_squared));
    AUTO_VAR(f, subtract<float>(x2_plus_y2, two_xy));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// ============================================================================
// Vector and Matrix Tests
// ============================================================================

// Test 10: f(u) = dot(u, v_const), df/du = v_const, report df/du[0]
BackADTestResult TestBackwardVectorDot(float ux, float uy, float vx, float vy)
{
    BackADTestResult result;
    GradientContext<float2> context;
    context.variable_count = 0;

    Variable<float2> u = variableVector<float2>(context, float2(ux, uy));
    VariableExpr<float2> u_expr;
    u_expr.var = u;

    VariableExpr<float2> v_expr;
    v_expr.var.value = float2(vx, vy);
    v_expr.var.id = -1;

    AUTO_VAR(f, dotProduct<float2>(u_expr, v_expr));

    result.value = compute_gradients(context, f);
    float2 grad = u.gradient(context);
    result.gradient = grad.x;
    return result;
}

// Test 11: f(v) = |v|, df/dv = v/|v|, report df/dv[0]
BackADTestResult TestBackwardVectorLength(float vx, float vy)
{
    BackADTestResult result;
    GradientContext<float2> context;
    context.variable_count = 0;

    Variable<float2> v = variableVector<float2>(context, float2(vx, vy));
    VariableExpr<float2> v_expr;
    v_expr.var = v;

    AUTO_VAR(f, (lengthExpr<float2>(v_expr)));

    result.value = compute_gradients(context, f);
    float2 grad = v.gradient(context);
    result.gradient = grad.x;
    return result;
}

// Test 12: f(M) = det(M), df/dM = adj(M)^T, report df/dM[0][0]
BackADTestResult TestBackwardMatrixDeterminant(float a, float b, float c, float d)
{
    BackADTestResult result;
    GradientContext<float2x2 > context;
    context.variable_count = 0;

    Variable<float2x2 > M = variableMatrix<float2x2 >(context, float2x2(a, b, c, d));
    VariableExpr<float2x2 > M_expr;
    M_expr.var = M;

    AUTO_VAR(f, determinantExpr<float2x2 >(M_expr));

    result.value = compute_gradients(context, f);
    float2x2 grad = M.gradient(context);
    result.gradient = grad[0][0];
    return result;
}

// Test 13: f(x) = x^n, f'(x) = n*x^(n-1)
BackADTestResult TestBackwardPower(float input_x, float exponent)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);

    // Use a constant variable for the exponent so it doesn't get gradient
    Variable<float> n = variable<float>(context, exponent);
    VariableExpr<float> n_expr = makeVariableExpr<float>(n);

    AUTO_VAR(f, power<float>(x_expr, n_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 14: f(x) = -x, f'(x) = -1
BackADTestResult TestBackwardNegate(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(f, negate<float>(x_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 15: f(x) = cos(x), f'(x) = -sin(x)
BackADTestResult TestBackwardCosine(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(f, cosExpr<float>(x_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 16: f(x) = sqrt(x), f'(x) = 1/(2*sqrt(x))
BackADTestResult TestBackwardSqrt(float input_x)
{
    BackADTestResult result;
    GradientContext<float> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, input_x);
    VariableExpr<float> x_expr = makeVariableExpr<float>(x);
    AUTO_VAR(f, sqrtExpr<float>(x_expr));

    result.value = compute_gradients<float>(context, f);
    result.gradient = x.gradient(context);
    return result;
}

// Test 17: f(v) = normalize(v)[0], report df/dv[0]
BackADTestResult TestBackwardNormalize(float vx, float vy)
{
    BackADTestResult result;
    GradientContext<float2> context;
    context.variable_count = 0;

    Variable<float2> v = variableVector<float2>(context, float2(vx, vy));
    VariableExpr<float2> v_expr;
    v_expr.var = v;

    AUTO_VAR(f, (normalizeExpr<float2>(v_expr)));

    float2 fval = compute_gradients(context, f);
    result.value = fval.x;
    float2 grad = v.gradient(context);
    result.gradient = grad.x;
    return result;
}

// Test 18: f(u,v) = cross(u,v)[0] = u.y*v.z - u.z*v.y, report df/du[0]
BackADTestResult TestBackwardCrossProduct(float ux, float uy, float uz, float vx, float vy, float vz)
{
    BackADTestResult result;
    GradientContext<float3 > context;
    context.variable_count = 0;

    Variable<float3 > u = variableVector<float3 >(context, float3(ux, uy, uz));
    VariableExpr<float3 > u_expr;
    u_expr.var = u;

    VariableExpr<float3 > v_expr;
    v_expr.var.value = float3(vx, vy, vz);
    v_expr.var.id = -1;

    AUTO_VAR(f, crossProduct<float3 >(u_expr, v_expr));

    float3 fval = compute_gradients(context, f);
    result.value = fval.x;
    float3 grad = u.gradient(context);
    result.gradient = grad.x;
    return result;
}

// Test 19: f(v) = (M*v)[0], report df/dv[0]
BackADTestResult TestBackwardMatVecMul(float m00, float m01, float m10, float m11, float vx, float vy)
{
    BackADTestResult result;
    GradientContext<float2> context;
    context.variable_count = 0;

    // Matrix is constant, vector is the variable
    Variable<float2> v = variableVector<float2>(context, float2(vx, vy));
    VariableExpr<float2> v_expr;
    v_expr.var = v;

    // Constant matrix
    VariableExpr<float2x2 > M_expr;
    M_expr.var.value = float2x2(m00, m01, m10, m11);
    M_expr.var.id = -1;

    AUTO_VAR(f, (matVecMul<float2x2, float2>(M_expr, v_expr)));

    float2 fval = compute_gradients(context, f);
    result.value = fval.x;
    float2 grad = v.gradient(context);
    result.gradient = grad.x;
    return result;
}

// ============================================================================
// Compute Shader Entry Point
// ============================================================================

[numthreads(8, 1, 1)]
void RunAllBackwardTests(uint3 DispatchThreadID : SV_DispatchThreadID)
{
    uint threadIndex = DispatchThreadID.x;

    // Read test inputs from CPU-provided buffer for this thread
    BackADTestInputs inputs = InputBuffer[threadIndex];

    // Collect all test results in local array
    BackADTestResult results[20];

    // Test 0: Quadratic - f(x) = x^2
    results[0] = TestBackwardQuadratic(inputs.quadratic_x);

    // Test 1: Sine - f(x) = sin(x)
    results[1] = TestBackwardSine(inputs.sine_x);

    // Test 2: Exponential - f(x) = exp(x)
    results[2] = TestBackwardExponential(inputs.exponential_x);

    // Test 3: Logarithm - f(x) = log(x)
    results[3] = TestBackwardLogarithm(inputs.logarithm_x);

    // Test 4: Chain Rule - f(x) = sin(x^2)
    results[4] = TestBackwardChainRule(inputs.chain_rule_x);

    // Test 5: Product Rule - f(x) = x*sin(x)
    results[5] = TestBackwardProductRule(inputs.product_rule_x);

    // Test 6: Addition - f(x,y) = x + y
    results[6] = TestBackwardAddition(inputs.addition_x, inputs.addition_y);

    // Test 7: Multiplication - f(x,y) = x * y
    results[7] = TestBackwardMultiplication(inputs.multiplication_x, inputs.multiplication_y);

    // Test 8: Division - f(x,y) = x / y
    results[8] = TestBackwardDivision(inputs.division_x, inputs.division_y);

    // Test 9: Complex Expression - f(x,y) = x^2 + y^2 - 2xy
    results[9] = TestBackwardComplexExpression(inputs.complex_x, inputs.complex_y);

    // Test 10: Vector Dot Product
    results[10] = TestBackwardVectorDot(inputs.vector_dot_ux, inputs.vector_dot_uy,
                                        inputs.vector_dot_vx, inputs.vector_dot_vy);

    // Test 11: Vector Length
    results[11] = TestBackwardVectorLength(inputs.vector_length_vx, inputs.vector_length_vy);

    // Test 12: Matrix Determinant
    results[12] = TestBackwardMatrixDeterminant(inputs.matrix_det_a, inputs.matrix_det_b,
                                                inputs.matrix_det_c, inputs.matrix_det_d);

    // Test 13: Power - f(x) = x^n
    results[13] = TestBackwardPower(inputs.power_base, inputs.power_exponent);

    // Test 14: Negate - f(x) = -x
    results[14] = TestBackwardNegate(inputs.negate_x);

    // Test 15: Cosine - f(x) = cos(x)
    results[15] = TestBackwardCosine(inputs.cosine_x);

    // Test 16: Sqrt - f(x) = sqrt(x)
    results[16] = TestBackwardSqrt(inputs.sqrt_x);

    // Test 17: Vector Normalize
    results[17] = TestBackwardNormalize(inputs.normalize_vx, inputs.normalize_vy);

    // Test 18: Cross Product
    results[18] = TestBackwardCrossProduct(inputs.cross_ux, inputs.cross_uy, inputs.cross_uz,
                                           inputs.cross_vx, inputs.cross_vy, inputs.cross_vz);

    // Test 19: Matrix-Vector Multiply
    results[19] = TestBackwardMatVecMul(inputs.matvec_m00, inputs.matvec_m01,
                                        inputs.matvec_m10, inputs.matvec_m11,
                                        inputs.matvec_vx, inputs.matvec_vy);

    // Write all results to structured buffer at thread-specific indices
    uint baseIndex = threadIndex * 20;
    for (int i = 0; i < 20; i++)
    {
        ResultBuffer[baseIndex + i] = results[i];
    }
}
