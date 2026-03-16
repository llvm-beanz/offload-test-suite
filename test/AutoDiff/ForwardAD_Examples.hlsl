#include "ForwardAD.hlsl"

using namespace ad::fwd;
using namespace ad::fwd::__detail;

// Example demonstrating forward automatic differentiation usage with templated types
// (HLSL doesn't support global operator overloading)

// Example 1: Simple polynomial differentiation
// f(x) = 3x^2 + 2x + 1
// f'(x) = 6x + 2
void ExamplePolynomial()
{
    // Define variable x with value 2.0, derivative 1.0 (differentiating w.r.t. x)
    Value<float> x = variable<float>(2.0f);

    // Define the function: f(x) = 3x^2 + 2x + 1
    // f = 3 * x^2 + 2 * x + 1
    Value<float> x_squared = pow(x, constant<float>(2.0f));
    Value<float> three_x_squared = x_squared * 3.0f;
    Value<float> two_x = x * 2.0f;
    Value<float> partial = three_x_squared + two_x;
    Value<float> f = partial + 1.0f;

    // At x = 2.0:
    // f(2.0) = 3(4) + 2(2) + 1 = 17
    // f'(2.0) = 6(2) + 2 = 14
    float function_value = f.value;      // Should be 17.0
    float derivative_value = f.derivative; // Should be 14.0
}

// Example 2: Trigonometric function differentiation
// f(x) = sin(x) * cos(x) + exp(x)
// f'(x) = cos²(x) - sin²(x) + exp(x) = cos(2x) + exp(x)
void ExampleTrigonometric()
{
    Value<float> x = variable<float>(1.0f); // x = 1.0

    // Define the function: f(x) = sin(x) * cos(x) + exp(x)
    Value<float> sin_x = sinExpr<float>(x).eval();
    Value<float> cos_x = cosExpr<float>(x).eval();
    Value<float> sin_cos = multiply<float>(sin_x, cos_x).eval();
    Value<float> exp_x = expExpr<float>(x).eval();
    Value<float> f = sin_cos + exp_x;

    // At x = 1.0:
    // f(1.0) = sin(1) * cos(1) + exp(1) ≈ 0.454 + 2.718 = 3.172
    // f'(1.0) = cos(2) + exp(1) ≈ -0.416 + 2.718 = 2.302
    float function_value = f.value;
    float derivative_value = f.derivative;
}

// Example 3: Direct Value arithmetic (when available)
void ExampleDirectValue()
{
    Value<float> x = variable<float>(3.0f);
    Value<float> y = variable<float>(2.0f);

    // Use named functions for Value operations
    Value<float> sum = x + y;
    Value<float> product = x * y;
    Value<float> quotient = x / y;
    Value<float> difference = x - y;
#if __hlsl_dx_compiler
    // Unary operators are broken in DXC.
    // https://github.com/microsoft/DirectXShaderCompiler/issues/7944
    Value<float> negated = negate<float>(x).eval();
#else
    Value<float> negated = -x;
#endif

    // Operations with floats
    Value<float> x_plus_5 = x + 5.0f;
    Value<float> x_times_3 = x * 3.0f;
}

// Example 4: Complex composite function
// f(x) = sqrt(x^2 + 1) * log(x + 2)
void ExampleComposite()
{
    Value<float> x = variable<float>(3.0f); // x = 3.0

    // Define the function: f(x) = sqrt(x^2 + 1) * log(x + 2)
    Value<float> x_squared = x * x;
    Value<float> x_sq_plus_1 = x_squared + 1.0f;
    Value<float> sqrt_part = sqrtExpr<float>(x_sq_plus_1).eval();

    Value<float> x_plus_2 = x + 2.0f;
    Value<float> log_part = logExpr<float>(x_plus_2).eval();

    Value<float> f = sqrt_part * log_part;

    // The derivative is computed automatically using the product and chain rules
    float function_value = f.value;
    float derivative_value = f.derivative;
}

// Example 5: Multi-variable partial derivatives
// For f(x,y) = x*y + sin(x), compute ∂f/∂x at (2, 3)
void ExamplePartialX()
{
    // To compute ∂f/∂x, set dx = 1, dy = 0
    Value<float> x = Value<float>::Create(2.0f, 1.0f); // x = 2, dx = 1
    Value<float> y = Value<float>::Create(3.0f, 0.0f); // y = 3, dy = 0

    Value<float> xy = x * y;
    Value<float> sin_x = sinExpr<float>(x).eval();
    Value<float> f = xy + sin_x;;

    // ∂f/∂x = y + cos(x) = 3 + cos(2) ≈ 3 - 0.416 = 2.584
    float partial_derivative = f.derivative;
}

// Example 6: Shader-style computation with forward AD
// Compute both lighting and its gradient for optimization
struct LightingResult
{
    Value<float> intensity;        // Light intensity and its derivative
    float3 gradient_dir;   // Direction of steepest intensity increase
};

LightingResult ComputeLightingWithGradient(float3 position, float3 light_pos)
{
    LightingResult result;

    // Compute lighting intensity and gradient w.r.t. x-coordinate
    Value<float> x = variable<float>(position.x);
    Value<float> y = constant<float>(position.y);
    Value<float> z = constant<float>(position.z);

    // Distance to light
    Value<float> dx = x - light_pos.x;
    Value<float> dy = y - light_pos.y;
    Value<float> dz = z - light_pos.z;

    Value<float> dx_sq = dx * dx;
    Value<float> dy_sq = dy * dy;
    Value<float> dz_sq = dz * dz;
    Value<float> dist_sq = dx_sq + dy_sq + dz_sq;
    Value<float> distance = sqrtExpr<float>(dist_sq).eval();

    // Inverse square law with some artistic falloff
    Value<float> dist_sq_plus_1 = distance * distance + 1.0f;

    #if __hlsl_dx_compiler
    result.intensity = Value<float>::Create(1.0f, 0.0f) / dist_sq_plus_1;
    #else
    // This case works with Clang but not DXC because DXC does not support
    // resolving global operator overloads.
    result.intensity = 1.0f / dist_sq_plus_1;
    #endif

    // The derivative gives us the gradient component in x-direction
    result.gradient_dir.x = result.intensity.derivative;

    // Would need separate evaluations for y and z gradients
    // (or use a vector-valued Value number system)
    return result;
}

// Example 7: Using expression templates for deferred evaluation
void ExampleDeferredEvaluation()
{
    Value<float> x = variable<float>(1.0f);

    // Build complex expression step by step
    Value<float> x_squared = x * x;
    Value<float> sin_x_sq = sinExpr<float>(x_squared).eval();

    Value<float> exp_x = expExpr<float>(x).eval();
    Value<float> cos_exp_x = cosExpr<float>(exp_x).eval();

    Value<float> x_plus_1 = x + Value<float>::Create(1.0f, 0.0f);
    Value<float> log_x_plus_1 = logExpr<float>(x_plus_1).eval();

    Value<float> cos_exp_log = cos_exp_x * log_x_plus_1;
    Value<float> complex_expr = sin_x_sq + cos_exp_log;

    // Extract components
    float val = complex_expr.value;
    float deriv = complex_expr.derivative;
}

// Example 8: Newton's method using automatic differentiation
float NewtonMethod(float initial_guess)
{
    float x_current = initial_guess;

    for (int i = 0; i < 10; i++) // 10 iterations
    {
        // Define function f(x) = x^3 - 2x - 5 (finding roots)
        Value<float> x = variable<float>(x_current);

        // f(x) = x^3 - 2x - 5
        Value<float> x_cubed = power<float>(x, constant<float>(3.0f)).eval();
        Value<float> two_x = x * 2.0f;
        Value<float> x_cubed_minus_2x = x_cubed - two_x;
        Value<float> f = x_cubed_minus_2x - 5.0f;

        float f_value = f.value;
        float f_derivative = f.derivative;

        // Newton's update: x_new = x - f(x)/f'(x)
        x_current = x_current - f_value / f_derivative;

        // Check for convergence
        if (abs(f_value) < 1e-6f)
            break;
    }

    return x_current;
}

// ============================================================================
// Vector and Matrix Examples
// ============================================================================

// Example 9: Vector operations with automatic differentiation
void ExampleVectorOperations()
{
    // Define variables for vector components
    Value<float> x = variable<float>(1.0f);
    Value<float> y = variable<float>(2.0f);

    // Create a vector [x, y]
    Value<vector<float, 2> > v = makeVector<float>(x, y);

    // Create a constant vector [3, 4]
    Value<vector<float, 2> > u = constantVector<float, 2>(vector<float, 2>(3.0f, 4.0f));

    // Compute dot product: f(x,y) = dot([x,y], [3,4]) = 3x + 4y
    // ∂f/∂x = 3, ∂f/∂y = 4 (but we're computing w.r.t. x with dx=1, dy=0)
    Value<float> dot_result = dotProduct<float, 2>(v, u).eval();

    // Compute vector length: |[x,y]| = sqrt(x^2 + y^2)
    // d/dx[sqrt(x^2 + y^2)] = x/sqrt(x^2 + y^2)
    Value<float> length_result = lengthExpr<float, 2>(v).eval();

    // Normalize vector
    Value<vector<float, 2> > normalized = normalizeExpr<float, 2>(v).eval();

    float dot_val = dot_result.value;         // 3*1 + 4*2 = 11
    float dot_deriv = dot_result.derivative;  // 3 (derivative w.r.t. x)
    float length_val = length_result.value;   // sqrt(1 + 4) = sqrt(5) ≈ 2.236
    float length_deriv = length_result.derivative; // 1/sqrt(5) ≈ 0.447
}

// Example 10: Cross product for 3D vectors
void ExampleCrossProduct()
{
    // Define variables
    Value<float> x = variable<float>(2.0f);

    // Create vectors: v = [x, 1, 0] and u = [0, x, 1]
    Value<vector<float, 3> > v = makeVector<float>(x, constant<float>(1.0f), constant<float>(0.0f));
    Value<vector<float, 3> > u = makeVector<float>(constant<float>(0.0f), x, constant<float>(1.0f));

    // Compute cross product: v × u
    // [x,1,0] × [0,x,1] = [1*1-0*x, 0*0-x*1, x*x-1*0] = [1, -x, x^2]
    Value<vector<float, 3> > cross_result = crossProduct<float>(v, u).eval();

    // Extract components and their derivatives
    Value<float> result_x = getX(cross_result);  // value = 1, derivative = 0
    Value<float> result_y = getY(cross_result);  // value = -x = -2, derivative = -1
    Value<float> result_z = getZ(cross_result);  // value = x^2 = 4, derivative = 2x = 4
}

// Example 11: Matrix-vector multiplication
void ExampleMatrixVector()
{
    // Define parameter
    Value<float> t = variable<float>(1.5f);

    // Create parameterized matrix [[t, 0], [0, 2t]]
    matrix<float, 2, 2> mat_val = matrix<float, 2, 2>(t.value, 0, 0, 2*t.value);
    matrix<float, 2, 2> mat_deriv = matrix<float, 2, 2>(t.derivative, 0, 0, 2*t.derivative);
    Value<matrix<float, 2, 2> > A = Value<matrix<float, 2, 2> >::Create(mat_val, mat_deriv);

    // Create vector [1, 3]
    Value<vector<float, 2> > v = constantVector<float, 2>(vector<float, 2>(1.0f, 3.0f));

    // Compute A*v = [[t,0],[0,2t]] * [1,3] = [t, 6t]
    Value<vector<float, 2> > result = matMul<float, 2, 2, 2>(A, v).eval();

    // Extract components
    Value<float> result_x = getX(result);  // value = t = 1.5, derivative = 1
    Value<float> result_y = getY(result);  // value = 6t = 9, derivative = 6
}

// Example 12: Matrix determinant differentiation
void ExampleMatrixDeterminant()
{
    // Define variables
    Value<float> a = variable<float>(2.0f);
    Value<float> b = constant<float>(3.0f);

    // Create matrix [[a, 1], [b, 4]] = [[a, 1], [3, 4]]
    matrix<float, 2, 2> mat_val = matrix<float, 2, 2>(a.value, 1, b.value, 4);
    matrix<float, 2, 2> mat_deriv = matrix<float, 2, 2>(a.derivative, 0, b.derivative, 0);
    Value<matrix<float, 2, 2> > M = Value<matrix<float, 2, 2> >::Create(mat_val, mat_deriv);

    // Compute determinant: det([[a,1],[3,4]]) = a*4 - 1*3 = 4a - 3
    // d/da[4a - 3] = 4
    Value<float> det_result = determinantExpr<float, 2>(M).eval();

    float det_val = det_result.value;       // 4*2 - 3 = 5
    float det_deriv = det_result.derivative; // 4
}

// Example 13: Transformation pipeline with gradients
void ExampleTransformationPipeline()
{
    // Define rotation angle as parameter
    Value<float> theta = variable<float>(0.785f); // π/4 radians

    // Create 2D rotation matrix
    Value<float> cos_theta = cosExpr<float>(theta).eval();
    Value<float> sin_theta = sinExpr<float>(theta).eval();
    Value<float> neg_sin = negate<float>(sin_theta).eval();

    // Rotation matrix [[cos θ, -sin θ], [sin θ, cos θ]]
    matrix<float, 2, 2> rot_val = matrix<float, 2, 2>(cos_theta.value, neg_sin.value,
                                                   sin_theta.value, cos_theta.value);
    matrix<float, 2, 2> rot_deriv = matrix<float, 2, 2>(cos_theta.derivative, neg_sin.derivative,
                                                     sin_theta.derivative, cos_theta.derivative);
    Value<matrix<float, 2, 2> > R = Value<matrix<float, 2, 2> >::Create(rot_val, rot_deriv);

    // Input vector to transform
    Value<vector<float, 2> > input = constantVector<float, 2>(vector<float, 2>(1.0f, 0.0f));

    // Apply rotation
    Value<vector<float, 2> > rotated = matMul<float, 2, 2, 2>(R, input).eval();
    // Compute length (should remain 1 for rotation)
    Value<float> length_after = lengthExpr<float, 2>(rotated).eval();

    // The gradient tells us how the transformed point moves with rotation angle
    Value<float> x_component = getX(rotated);
    Value<float> y_component = getY(rotated);

    // At θ = π/4: rotated point = [cos(π/4), sin(π/4)] = [√2/2, √2/2]
    // Derivatives give velocity of rotation: [-sin(π/4), cos(π/4)] = [-√2/2, √2/2]
}

// Example 14: Computing Jacobian matrix elements
void ExampleJacobianComputation()
{
    // For function f: R² → R² defined as f([x,y]) = [x²+y, xy]
    // Compute partial derivatives to form Jacobian matrix

    float x_val = 2.0f, y_val = 3.0f;

    // Compute ∂f₁/∂x and ∂f₂/∂x (partial derivatives w.r.t. x)
    {
        Value<float> x = variable<float>(x_val);  // dx = 1
        Value<float> y = constant<float>(y_val);   // dy = 0

        // f₁(x,y) = x² + y
        Value<float> x_sq = x * x;
        Value<float> f1 = x_sq + y;

        // f₂(x,y) = xy
        Value<float> f2 = x * y;

        float df1_dx = f1.derivative;  // ∂(x²+y)/∂x = 2x = 4
        float df2_dx = f2.derivative;  // ∂(xy)/∂x = y = 3
    }

    // Compute ∂f₁/∂y and ∂f₂/∂y (partial derivatives w.r.t. y)
    {
        Value<float> x = constant<float>(x_val);   // dx = 0
        Value<float> y = variable<float>(y_val);   // dy = 1

        // f₁(x,y) = x² + y
        Value<float> x_sq = x * x;
        Value<float> f1 = x_sq + y;

        // f₂(x,y) = xy
        Value<float> f2 = x * y;

        float df1_dy = f1.derivative;  // ∂(x²+y)/∂y = 1
        float df2_dy = f2.derivative;  // ∂(xy)/∂y = x = 2

        // Jacobian matrix at (2,3) is [[4,1],[3,2]]
    }
}
