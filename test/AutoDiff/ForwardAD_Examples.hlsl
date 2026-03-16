#include "ForwardAD.hlsl"

using namespace ad::fwd;

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
    Value<float> x_squared = getValue(power<float>(x, constant<float>(2.0f)));
    Value<float> three_x_squared = getValue(multiply<float>(constant<float>(3.0f), x_squared));
    Value<float> two_x = getValue(multiply<float>(constant<float>(2.0f), x));
    Value<float> partial = getValue(add<float>(three_x_squared, two_x));
    Value<float> f = getValue(add<float>(partial, constant<float>(1.0f)));
    
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
    Value<float> sin_x = getValue(sinExpr<float>(x));
    Value<float> cos_x = getValue(cosExpr<float>(x));
    Value<float> sin_cos = getValue(multiply<float>(sin_x, cos_x));
    Value<float> exp_x = getValue(expExpr<float>(x));
    Value<float> f = getValue(add<float>(sin_cos, exp_x));
    
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
    Value<float> sum = getValue(add<float>(x, y));                    // x + y
    Value<float> product = getValue(multiply<float>(x, y));           // x * y  
    Value<float> quotient = getValue(divide<float>(x, y));            // x / y
    Value<float> difference = getValue(subtract<float>(x, y));        // x - y
    Value<float> negated = getValue(negate<float>(x));                // -x
    
    // Operations with floats
    Value<float> x_plus_5 = getValue(add<float>(x, Value<float>::Create(5.0f, 0.0f)));
    Value<float> x_times_3 = getValue(multiply<float>(x, Value<float>::Create(3.0f, 0.0f)));
}

// Example 4: Complex composite function
// f(x) = sqrt(x^2 + 1) * log(x + 2)
void ExampleComposite()
{
    Value<float> x = variable<float>(3.0f); // x = 3.0
    
    // Define the function: f(x) = sqrt(x^2 + 1) * log(x + 2)
    Value<float> x_squared = getValue(multiply<float>(x, x));
    Value<float> x_sq_plus_1 = getValue(add<float>(x_squared, Value<float>::Create(1.0f, 0.0f)));
    Value<float> sqrt_part = getValue(sqrtExpr<float>(x_sq_plus_1));
    
    Value<float> x_plus_2 = getValue(add<float>(x, Value<float>::Create(2.0f, 0.0f)));
    Value<float> log_part = getValue(logExpr<float>(x_plus_2));
    
    Value<float> f = getValue(multiply<float>(sqrt_part, log_part));
    
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
    
    Value<float> xy = getValue(multiply<float>(x, y));
    Value<float> sin_x = getValue(sinExpr<float>(x));
    Value<float> f = getValue(add<float>(xy, sin_x));
    
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
    Value<float> dx = getValue(subtract<float>(x, constant<float>(light_pos.x)));
    Value<float> dy = getValue(subtract<float>(y, constant<float>(light_pos.y)));
    Value<float> dz = getValue(subtract<float>(z, constant<float>(light_pos.z)));
    
    Value<float> dx_sq = getValue(multiply<float>(dx, dx));
    Value<float> dy_sq = getValue(multiply<float>(dy, dy));
    Value<float> dz_sq = getValue(multiply<float>(dz, dz));
    Value<float> dist_sq = getValue(add<float>(getValue(add<float>(dx_sq, dy_sq)), dz_sq));
    Value<float> distance = getValue(sqrtExpr<float>(dist_sq));
    
    // Inverse square law with some artistic falloff
    Value<float> dist_sq_plus_1 = getValue(add<float>(getValue(multiply<float>(distance, distance)), Value<float>::Create(1.0f, 0.0f)));
    result.intensity = getValue(divide<float>(Value<float>::Create(1.0f, 0.0f), dist_sq_plus_1));
    
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
    Value<float> x_squared = getValue(multiply<float>(x, x));
    Value<float> sin_x_sq = getValue(sinExpr<float>(x_squared));
    
    Value<float> exp_x = getValue(expExpr<float>(x));
    Value<float> cos_exp_x = getValue(cosExpr<float>(exp_x));
    
    Value<float> x_plus_1 = getValue(add<float>(x, Value<float>::Create(1.0f, 0.0f)));
    Value<float> log_x_plus_1 = getValue(logExpr<float>(x_plus_1));
    
    Value<float> cos_exp_log = getValue(multiply<float>(cos_exp_x, log_x_plus_1));
    Value<float> complex_expr = getValue(add<float>(sin_x_sq, cos_exp_log));
    
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
        Value<float> x_cubed = getValue(power<float>(x, constant<float>(3.0f)));
        Value<float> two_x = getValue(multiply<float>(Value<float>::Create(2.0f, 0.0f), x));
        Value<float> x_cubed_minus_2x = getValue(subtract<float>(x_cubed, two_x));
        Value<float> f = getValue(subtract<float>(x_cubed_minus_2x, Value<float>::Create(5.0f, 0.0f)));
        
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
    Value<float> dot_result = getValue(dotProduct<float, 2>(v, u));
    
    // Compute vector length: |[x,y]| = sqrt(x^2 + y^2)
    // d/dx[sqrt(x^2 + y^2)] = x/sqrt(x^2 + y^2)
    Value<float> length_result = getValue(lengthExpr<float, 2>(v));
    
    // Normalize vector
    Value<vector<float, 2> > normalized = getValue(normalizeExpr<float, 2>(v));
    
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
    Value<vector<float, 3> > cross_result = getValue(crossProduct<float>(v, u));
    
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
    Value<vector<float, 2> > result = getValue(matMul<float, 2, 2, 2>(A, v));
    
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
    Value<float> det_result = getValue(determinantExpr<float, 2>(M));
    
    float det_val = det_result.value;       // 4*2 - 3 = 5
    float det_deriv = det_result.derivative; // 4
}

// Example 13: Transformation pipeline with gradients
void ExampleTransformationPipeline()
{
    // Define rotation angle as parameter
    Value<float> theta = variable<float>(0.785f); // π/4 radians
    
    // Create 2D rotation matrix
    Value<float> cos_theta = getValue(cosExpr<float>(theta));
    Value<float> sin_theta = getValue(sinExpr<float>(theta));
    Value<float> neg_sin = getValue(negate<float>(sin_theta));
    
    // Rotation matrix [[cos θ, -sin θ], [sin θ, cos θ]]
    matrix<float, 2, 2> rot_val = matrix<float, 2, 2>(cos_theta.value, neg_sin.value, 
                                                   sin_theta.value, cos_theta.value);
    matrix<float, 2, 2> rot_deriv = matrix<float, 2, 2>(cos_theta.derivative, neg_sin.derivative,
                                                     sin_theta.derivative, cos_theta.derivative);
    Value<matrix<float, 2, 2> > R = Value<matrix<float, 2, 2> >::Create(rot_val, rot_deriv);
    
    // Input vector to transform
    Value<vector<float, 2> > input = constantVector<float, 2>(vector<float, 2>(1.0f, 0.0f));
    
    // Apply rotation
    Value<vector<float, 2> > rotated = getValue(matMul<float, 2, 2, 2>(R, input));    
    // Compute length (should remain 1 for rotation)
    Value<float> length_after = getValue(lengthExpr<float, 2>(rotated));
    
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
        Value<float> x_sq = getValue(multiply<float>(x, x));
        Value<float> f1 = getValue(add<float>(x_sq, y));
        
        // f₂(x,y) = xy
        Value<float> f2 = getValue(multiply<float>(x, y));
        
        float df1_dx = f1.derivative;  // ∂(x²+y)/∂x = 2x = 4
        float df2_dx = f2.derivative;  // ∂(xy)/∂x = y = 3
    }
    
    // Compute ∂f₁/∂y and ∂f₂/∂y (partial derivatives w.r.t. y)
    {
        Value<float> x = constant<float>(x_val);   // dx = 0  
        Value<float> y = variable<float>(y_val);   // dy = 1
        
        // f₁(x,y) = x² + y
        Value<float> x_sq = getValue(multiply<float>(x, x));
        Value<float> f1 = getValue(add<float>(x_sq, y));
        
        // f₂(x,y) = xy
        Value<float> f2 = getValue(multiply<float>(x, y));
        
        float df1_dy = f1.derivative;  // ∂(x²+y)/∂y = 1
        float df2_dy = f2.derivative;  // ∂(xy)/∂y = x = 2
        
        // Jacobian matrix at (2,3) is [[4,1],[3,2]]
    }
}
