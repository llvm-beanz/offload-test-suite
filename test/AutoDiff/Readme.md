# HLSL Automatic Differentiation Library

This folder contins a prototype library implementation of an expression template
system for automatic differentiation. This library can be used to generate
functions and their first derivatives.

The library includes two different approaches for defining and running
derivative functions. The first (and simplest) is forward differentiation. The
forward differentiation evaluates each operation of a function and that
operation's first derivative at the same time. It only requires additional
storage for the immediate derivative value. This solution is ideal if you have
few inputs and can remain efficient even with many output parameters.

The second approach is backward differentiation. The backward differentiation
implementation builds up an expression tree and can evaluate a gradient with
respect to a wide number of inputs. This approach is more powerful for machine
learning applications, however it has additional overhead because it requires
storage for all intermediate variables.

## Contents

### Utilities
* [Type Traits](type_traits.h) - Traits for arithmetic, vector and matrix types.
* [hlsl::enable_if](enable_if.h) - SFINAE primitive for `enable_if`.

### Forward Differentiation

* [ForwardAD Header](ForwardAD.hlsl) - Header implementation for forward
  differntiation.
* [ForwardAD Examples](ForwardAD_Examples.hlsl) - Examples using the forward
  differentiation APIs.
* [ForwardAD Tests](ForwardAD_Tests.hlsl) - Incomplete tests for the forward
  differntiation APIs.
* [ForwardAD Test Pipeline](Forward.yaml) - Offload-test-suite YAML file to
  execute tests with test data.

### Backward Differentiation

* [BackwardAD Header](BackwardAD.hlsl) - Header implementation for backward
  differntiation.
* [BackwardAD Examples](BackwardAD_Examples.hlsl) - Examples using the backward
  differentiation APIs.
* [BackwardAD Tests](BackwardAD_Tests.hlsl) - Incomplete tests for the backward
  differntiation APIs.
* [BackwardAD Test Pipeline](Backward.yaml) - Offload-test-suite YAML file to
  execute tests with test data.

## Limitations

### General Limitations

The current implementations of both approaches only support a limited surface
area of math expressions. These are fairly easy to extend (even outside the
library code itself), so this isn't a critical limitation.

### Limitations in Backward Approach

#### Complex Type Names

Because HLSL does not support pointers or dynamic polymorphism building an
expression tree becomes quite complicated. The approach used in this
implementation effectively ends up with the tree being a _really_ complicated
template name. For example using fully qualified names the function `x*y + x^2`
becomes something like:

```c++
BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > xy = multiply<float>(x_expr, y_expr);
BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > x_squared = multiply<float>(x_expr, x_expr);
BackAddExpr<float, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> >, BackMulExpr<float, VariableExpr<float>, VariableExpr<float> > > f = add<float>(xy, x_squared);
```

This is incredibly unwieldy to work with. If HLSL had the C++ `auto` keyword
this would be better, but in its absence we can do something awful and use a
macro:

```c++
#define AUTO_VAR(var,...) __decltype(__VA_ARGS__) var = __VA_ARGS__
```

This allows rewriting the code more simply as:

```c++
AUTO_VAR(xy, multiply<float>(x_expr, y_expr));
AUTO_VAR(x_squared, multiply<float>(x_expr, x_expr));
AUTO_VAR(f, add<float>(xy, x_squared));
```

#### Manually Allocated Storage

Other languages like Slang support compiler-aided Automatic Differentiation,
which allows the compiler to handle things like memory allocation for the user.
This makes reverse differentiation _much_ easier to use. In this library we
require that the user allocate storage in a `GradientContext<T,N=64>`. By
default the `GradientContext` template allocates 64 slots for variables. At the
moment this has a **significant limitation** because it only allows handling a
single type of gradient at a time.

```c++
// Example 1: Simple function f(x) = x^2, f'(x) = 2x
void ExampleSimpleQuadratic()
{
    GradientContext<float/*, 64*/> context;
    context.variable_count = 0;

    Variable<float> x = variable<float>(context, 3.0f);  // x = 3
    AUTO_VAR(x_expr, makeVariableExpr<float>(x));

    // f(x) = x^2
    AUTO_VAR(f, multiply<float>(x_expr, x_expr));

    // Compute function value and gradients
    float result = compute_gradients<float>(context, f);

    // result should be 9.0 (3^2)
    // x.gradient() should be 6.0 (2*3)
    float function_value = result;
    float derivative = x.gradient(context);
}
```