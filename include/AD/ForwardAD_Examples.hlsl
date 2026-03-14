#include "ForwardAD.hlsl"

// Example demonstrating forward automatic differentiation usage with templated types
// (HLSL doesn't support global operator overloading)

// Example 1: Simple polynomial differentiation  
// f(x) = 3x^2 + 2x + 1
// f'(x) = 6x + 2
void ExamplePolynomial()
{
    // Define variable x with value 2.0, derivative 1.0 (differentiating w.r.t. x)
    Dual<float> x = variable<float>(2.0f);
    
    // Define the function: f(x) = 3x^2 + 2x + 1
    // f = 3 * x^2 + 2 * x + 1
    Dual<float> x_squared = getValue(power<float>(x, constant<float>(2.0f)));
    Dual<float> three_x_squared = getValue(multiply<float>(constant<float>(3.0f), x_squared));
    Dual<float> two_x = getValue(multiply<float>(constant<float>(2.0f), x));
    Dual<float> partial = getValue(add<float>(three_x_squared, two_x));
    Dual<float> f = getValue(add<float>(partial, constant<float>(1.0f)));
    
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
    Dual<float> x = variable<float>(1.0f); // x = 1.0
    
    // Define the function: f(x) = sin(x) * cos(x) + exp(x)
    Dual<float> sin_x = getValue(sinExpr<float>(x));
    Dual<float> cos_x = getValue(cosExpr<float>(x));
    Dual<float> sin_cos = getValue(multiply<float>(sin_x, cos_x));
    Dual<float> exp_x = getValue(expExpr<float>(x));
    Dual<float> f = getValue(add<float>(sin_cos, exp_x));
    
    // At x = 1.0:
    // f(1.0) = sin(1) * cos(1) + exp(1) ≈ 0.454 + 2.718 = 3.172
    // f'(1.0) = cos(2) + exp(1) ≈ -0.416 + 2.718 = 2.302
    float function_value = f.value;
    float derivative_value = f.derivative;
}

// Example 3: Direct Dual arithmetic (when available)
void ExampleDirectDual()
{
    Dual<float> x = variable<float>(3.0f);
    Dual<float> y = variable<float>(2.0f);
    
    // Use named functions for Dual operations
    Dual<float> sum = getValue(add<float>(x, y));                    // x + y
    Dual<float> product = getValue(multiply<float>(x, y));           // x * y  
    Dual<float> quotient = getValue(divide<float>(x, y));            // x / y
    Dual<float> difference = getValue(subtract<float>(x, y));        // x - y
    Dual<float> negated = getValue(negate<float>(x));                // -x
    
    // Operations with floats
    Dual<float> x_plus_5 = getValue(add<float>(x, makeDual<float>(5.0f, 0.0f)));
    Dual<float> x_times_3 = getValue(multiply<float>(x, makeDual<float>(3.0f, 0.0f)));
}

// Example 4: Complex composite function
// f(x) = sqrt(x^2 + 1) * log(x + 2)
void ExampleComposite()
{
    Dual<float> x = variable<float>(3.0f); // x = 3.0
    
    // Define the function: f(x) = sqrt(x^2 + 1) * log(x + 2)
    Dual<float> x_squared = getValue(multiply<float>(x, x));
    Dual<float> x_sq_plus_1 = getValue(add<float>(x_squared, makeDual<float>(1.0f, 0.0f)));
    Dual<float> sqrt_part = getValue(sqrtExpr<float>(x_sq_plus_1));
    
    Dual<float> x_plus_2 = getValue(add<float>(x, makeDual<float>(2.0f, 0.0f)));
    Dual<float> log_part = getValue(logExpr<float>(x_plus_2));
    
    Dual<float> f = getValue(multiply<float>(sqrt_part, log_part));
    
    // The derivative is computed automatically using the product and chain rules
    float function_value = f.value;
    float derivative_value = f.derivative;
}

// Example 5: Multi-variable partial derivatives
// For f(x,y) = x*y + sin(x), compute ∂f/∂x at (2, 3)
void ExamplePartialX()
{
    // To compute ∂f/∂x, set dx = 1, dy = 0
    Dual<float> x = makeDual<float>(2.0f, 1.0f); // x = 2, dx = 1
    Dual<float> y = makeDual<float>(3.0f, 0.0f); // y = 3, dy = 0
    
    Dual<float> xy = getValue(multiply<float>(x, y));
    Dual<float> sin_x = getValue(sinExpr<float>(x));
    Dual<float> f = getValue(add<float>(xy, sin_x));
    
    // ∂f/∂x = y + cos(x) = 3 + cos(2) ≈ 3 - 0.416 = 2.584
    float partial_derivative = f.derivative;
}

// Example 6: Shader-style computation with forward AD
// Compute both lighting and its gradient for optimization
struct LightingResult
{
    Dual<float> intensity;        // Light intensity and its derivative
    float3 gradient_dir;   // Direction of steepest intensity increase
};

LightingResult ComputeLightingWithGradient(float3 position, float3 light_pos)
{
    LightingResult result;
    
    // Compute lighting intensity and gradient w.r.t. x-coordinate
    Dual<float> x = variable<float>(position.x);
    Dual<float> y = constant<float>(position.y);
    Dual<float> z = constant<float>(position.z);
    
    // Distance to light
    Dual<float> dx = getValue(subtract<float>(x, constant<float>(light_pos.x)));
    Dual<float> dy = getValue(subtract<float>(y, constant<float>(light_pos.y)));
    Dual<float> dz = getValue(subtract<float>(z, constant<float>(light_pos.z)));
    
    Dual<float> dx_sq = getValue(multiply<float>(dx, dx));
    Dual<float> dy_sq = getValue(multiply<float>(dy, dy));
    Dual<float> dz_sq = getValue(multiply<float>(dz, dz));
    Dual<float> dist_sq = getValue(add<float>(getValue(add<float>(dx_sq, dy_sq)), dz_sq));
    Dual<float> distance = getValue(sqrtExpr<float>(dist_sq));
    
    // Inverse square law with some artistic falloff
    Dual<float> dist_sq_plus_1 = getValue(add<float>(getValue(multiply<float>(distance, distance)), makeDual<float>(1.0f, 0.0f)));
    result.intensity = getValue(divide<float>(makeDual<float>(1.0f, 0.0f), dist_sq_plus_1));
    
    // The derivative gives us the gradient component in x-direction
    result.gradient_dir.x = result.intensity.derivative;
    
    // Would need separate evaluations for y and z gradients
    // (or use a vector-valued dual number system)
    
    return result;
}

// Example 7: Using expression templates for deferred evaluation
void ExampleDeferredEvaluation()
{
    Dual<float> x = variable<float>(1.0f);
    
    // Build complex expression step by step
    Dual<float> x_squared = getValue(multiply<float>(x, x));
    Dual<float> sin_x_sq = getValue(sinExpr<float>(x_squared));
    
    Dual<float> exp_x = getValue(expExpr<float>(x));
    Dual<float> cos_exp_x = getValue(cosExpr<float>(exp_x));
    
    Dual<float> x_plus_1 = getValue(add<float>(x, makeDual<float>(1.0f, 0.0f)));
    Dual<float> log_x_plus_1 = getValue(logExpr<float>(x_plus_1));
    
    Dual<float> cos_exp_log = getValue(multiply<float>(cos_exp_x, log_x_plus_1));
    Dual<float> complex_expr = getValue(add<float>(sin_x_sq, cos_exp_log));
    
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
        Dual<float> x = variable<float>(x_current);
        
        // f(x) = x^3 - 2x - 5
        Dual<float> x_cubed = getValue(power<float>(x, constant<float>(3.0f)));
        Dual<float> two_x = getValue(multiply<float>(makeDual<float>(2.0f, 0.0f), x));
        Dual<float> x_cubed_minus_2x = getValue(subtract<float>(x_cubed, two_x));
        Dual<float> f = getValue(subtract<float>(x_cubed_minus_2x, makeDual<float>(5.0f, 0.0f)));
        
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