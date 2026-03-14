#include "ForwardAD.hlsl"

// Example demonstrating forward automatic differentiation usage with named functions
// (HLSL doesn't support global operator overloading)

// Example 1: Simple polynomial differentiation  
// f(x) = 3x^2 + 2x + 1
// f'(x) = 6x + 2
void ExamplePolynomial()
{
    // Define variable x with value 2.0, derivative 1.0 (differentiating w.r.t. x)
    Dual x = variable(2.0f);
    
    // Define the function: f(x) = 3x^2 + 2x + 1
    // f = 3 * x^2 + 2 * x + 1
    Dual x_squared = getValue(power(x, constant(2.0f)));
    Dual three_x_squared = getValue(multiply(constant(3.0f), x_squared));
    Dual two_x = getValue(multiply(constant(2.0f), x));
    Dual partial = getValue(add(three_x_squared, two_x));
    Dual f = getValue(add(partial, constant(1.0f)));
    
    // At x = 2.0:
    // f(2.0) = 3(4) + 2(2) + 1 = 17
    // f'(2.0) = 6(2) + 2 = 14
    float function_value = value(f);      // Should be 17.0
    float derivative_value = derivative(f); // Should be 14.0
}

// Example 2: Trigonometric function differentiation
// f(x) = sin(x) * cos(x) + exp(x)
// f'(x) = cos²(x) - sin²(x) + exp(x) = cos(2x) + exp(x)
void ExampleTrigonometric()
{
    Dual x = variable(1.0f); // x = 1.0
    
    // Define the function: f(x) = sin(x) * cos(x) + exp(x)
    Dual sin_x = getValue(sinExpr(x));
    Dual cos_x = getValue(cosExpr(x));
    Dual sin_cos = getValue(multiply(sin_x, cos_x));
    Dual exp_x = getValue(expExpr(x));
    Dual f = getValue(add(sin_cos, exp_x));
    
    // At x = 1.0:
    // f(1.0) = sin(1) * cos(1) + exp(1) ≈ 0.454 + 2.718 = 3.172
    // f'(1.0) = cos(2) + exp(1) ≈ -0.416 + 2.718 = 2.302
    float function_value = value(f);
    float derivative_value = derivative(f);
}

// Example 3: Direct Dual arithmetic (when available)
void ExampleDirectDual()
{
    Dual x = variable(3.0f);
    Dual y = variable(2.0f);
    
    // Use named functions for Dual operations
    Dual sum = getValue(add(x, y));                    // x + y
    Dual product = getValue(multiply(x, y));           // x * y  
    Dual quotient = getValue(divide(x, y));            // x / y
    Dual difference = getValue(subtract(x, y));        // x - y
    Dual negated = getValue(negate(x));                // -x
    
    // Operations with floats
    Dual x_plus_5 = getValue(add(x, makeDual(5.0f, 0.0f)));
    Dual x_times_3 = getValue(multiply(x, makeDual(3.0f, 0.0f)));
}

// Example 4: Complex composite function
// f(x) = sqrt(x^2 + 1) * log(x + 2)
void ExampleComposite()
{
    Dual x = variable(3.0f); // x = 3.0
    
    // Define the function: f(x) = sqrt(x^2 + 1) * log(x + 2)
    Dual x_squared = getValue(multiply(x, x));
    Dual x_sq_plus_1 = getValue(add(x_squared, makeDual(1.0f, 0.0f)));
    Dual sqrt_part = getValue(sqrtExpr(x_sq_plus_1));
    
    Dual x_plus_2 = getValue(add(x, makeDual(2.0f, 0.0f)));
    Dual log_part = getValue(logExpr(x_plus_2));
    
    Dual f = getValue(multiply(sqrt_part, log_part));
    
    // The derivative is computed automatically using the product and chain rules
    float function_value = value(f);
    float derivative_value = derivative(f);
}

// Example 5: Multi-variable partial derivatives
// For f(x,y) = x*y + sin(x), compute ∂f/∂x at (2, 3)
void ExamplePartialX()
{
    // To compute ∂f/∂x, set dx = 1, dy = 0
    Dual x = makeDual(2.0f, 1.0f); // x = 2, dx = 1
    Dual y = makeDual(3.0f, 0.0f); // y = 3, dy = 0
    
    Dual xy = getValue(multiply(x, y));
    Dual sin_x = getValue(sinExpr(x));
    Dual f = getValue(add(xy, sin_x));
    
    // ∂f/∂x = y + cos(x) = 3 + cos(2) ≈ 3 - 0.416 = 2.584
    float partial_derivative = derivative(f);
}

// Example 6: Shader-style computation with forward AD
// Compute both lighting and its gradient for optimization
struct LightingResult
{
    Dual intensity;        // Light intensity and its derivative
    float3 gradient_dir;   // Direction of steepest intensity increase
};

LightingResult ComputeLightingWithGradient(float3 position, float3 light_pos)
{
    LightingResult result;
    
    // Compute lighting intensity and gradient w.r.t. x-coordinate
    Dual x = variable(position.x);
    Dual y = constant(position.y);
    Dual z = constant(position.z);
    
    // Distance to light
    Dual dx = getValue(subtract(x, constant(light_pos.x)));
    Dual dy = getValue(subtract(y, constant(light_pos.y)));
    Dual dz = getValue(subtract(z, constant(light_pos.z)));
    
    Dual dx_sq = getValue(multiply(dx, dx));
    Dual dy_sq = getValue(multiply(dy, dy));
    Dual dz_sq = getValue(multiply(dz, dz));
    Dual dist_sq = getValue(add(getValue(add(dx_sq, dy_sq)), dz_sq));
    Dual distance = getValue(sqrtExpr(dist_sq));
    
    // Inverse square law with some artistic falloff
    Dual dist_sq_plus_1 = getValue(add(getValue(multiply(distance, distance)), makeDual(1.0f, 0.0f)));
    result.intensity = getValue(divide(makeDual(1.0f, 0.0f), dist_sq_plus_1));
    
    // The derivative gives us the gradient component in x-direction
    result.gradient_dir.x = derivative(result.intensity);
    
    // Would need separate evaluations for y and z gradients
    // (or use a vector-valued dual number system)
    
    return result;
}

// Example 7: Using expression templates for deferred evaluation
void ExampleDeferredEvaluation()
{
    Dual x = variable(1.0f);
    
    // Build complex expression step by step
    Dual x_squared = getValue(multiply(x, x));
    Dual sin_x_sq = getValue(sinExpr(x_squared));
    
    Dual exp_x = getValue(expExpr(x));
    Dual cos_exp_x = getValue(cosExpr(exp_x));
    
    Dual x_plus_1 = getValue(add(x, makeDual(1.0f, 0.0f)));
    Dual log_x_plus_1 = getValue(logExpr(x_plus_1));
    
    Dual cos_exp_log = getValue(multiply(cos_exp_x, log_x_plus_1));
    Dual complex_expr = getValue(add(sin_x_sq, cos_exp_log));
    
    // Extract components
    float val = value(complex_expr);
    float deriv = derivative(complex_expr);
}

// Example 8: Newton's method using automatic differentiation
float NewtonMethod(float initial_guess)
{
    float x_current = initial_guess;
    
    for (int i = 0; i < 10; i++) // 10 iterations
    {
        // Define function f(x) = x^3 - 2x - 5 (finding roots)
        Dual x = variable(x_current);
        
        // f(x) = x^3 - 2x - 5
        Dual x_cubed = getValue(power(x, constant(3.0f)));
        Dual two_x = getValue(multiply(makeDual(2.0f, 0.0f), x));
        Dual x_cubed_minus_2x = getValue(subtract(x_cubed, two_x));
        Dual f = getValue(subtract(x_cubed_minus_2x, makeDual(5.0f, 0.0f)));
        
        float f_value = value(f);
        float f_derivative = derivative(f);
        
        // Newton's update: x_new = x - f(x)/f'(x)
        x_current = x_current - f_value / f_derivative;
        
        // Check for convergence
        if (abs(f_value) < 1e-6f)
            break;
    }
    
    return x_current;
}