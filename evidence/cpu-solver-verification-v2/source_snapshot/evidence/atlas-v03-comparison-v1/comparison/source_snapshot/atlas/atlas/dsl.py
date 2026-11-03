"""Restricted tensor expressions: one AST for checking, interpretation and C++.

No eval, exec, attribute access, indexing, strings or arbitrary function calls.
Dimensions use (mass, length, time, temperature) SI exponents, including fractional powers.
"""
from __future__ import annotations
import ast
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from .scalar_math import log1p as checked_log1p
from .symmetric_spectral import SPECTRAL_FUNCTIONS, spectral

ZERO = (Fraction(0),) * 4
PRESSURE = (Fraction(1), Fraction(-1), Fraction(-2), Fraction(0))
TIME = (Fraction(0), Fraction(0), Fraction(1), Fraction(0))
DIMS = {"1": ZERO, "Pa": PRESSURE, "s": TIME,
        "K": (Fraction(0), Fraction(0), Fraction(0), Fraction(1))}


def dimensions(unit):
    if isinstance(unit, str) and unit in DIMS:
        return DIMS[unit]
    if isinstance(unit, dict) and set(unit) == {'si'}:
        values = unit['si']
        if isinstance(values, list) and len(values) == 4:
            def valid_exponent(v):
                return (type(v) in (int,float) and math.isfinite(v)) or (
                    isinstance(v,str) and re.fullmatch(r'-?[0-9]{1,3}/[1-9][0-9]?',v) is not None)
            if all(valid_exponent(v) for v in values):
                result = tuple(Fraction(str(v)) for v in values)
                if all(abs(v) <= 16 and v.denominator <= 16 for v in result):
                    return result
    raise ValidationError('unit must be 1, Pa, s, K or four bounded rational SI exponents')


@dataclass(frozen=True)
class Type:
    shape: str = "scalar"
    dim: tuple = ZERO
    value_only: bool = False

    def __post_init__(self):
        # Existing callers using the v0.1 mass/length/time tuple remain valid.
        if len(self.dim) == 3:
            object.__setattr__(self, 'dim', tuple(self.dim) + (Fraction(0),))


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class Sym:
    v: tuple

    def __add__(self, other):
        return Sym(tuple(a + b for a, b in zip(self.v, other.v)))

    def __sub__(self, other):
        return self + (-other)

    def __neg__(self):
        return Sym(tuple(-a for a in self.v))

    def __mul__(self, value):
        return Sym(tuple(a * value for a in self.v))

    __rmul__ = __mul__

    def __truediv__(self, value):
        return self * (1 / value)


I = Sym((1., 1., 1., 0., 0., 0.))


def trace(a):
    """Nearest-even exact sum of binary64 diagonals; validate full tensor."""
    if not isinstance(a, Sym) or len(a.v) != 6:
        raise ValidationError("trace requires a six-component symmetric tensor")
    values = []
    for value in a.v:
        if type(value) not in (int, float):
            raise ValidationError("trace requires finite real components")
        try:
            value = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValidationError("trace requires finite real components") from exc
        if not math.isfinite(value):
            raise ValidationError("trace requires finite real components")
        values.append(value)
    try:
        result = float(sum((Fraction.from_float(v) for v in values[:3]), Fraction()))
    except OverflowError as exc:
        raise ValidationError("trace final sum overflows binary64") from exc
    return 0. if result == 0. else result


def dev(a):
    result = a - I * (trace(a) / 3)
    if not all(math.isfinite(v) for v in result.v):
        raise ValidationError("deviator result is nonfinite")
    return result

def dot(a, b):
    return sum(x * y for x, y in zip(a.v, b.v))


def cube_trace(a):
    x, y, z, xy, yz, xz = a.v
    xy, yz, xz = (v / math.sqrt(2) for v in (xy, yz, xz))
    return x**3 + y**3 + z**3 + 3*((x+y)*xy**2+(y+z)*yz**2+(x+z)*xz**2) + 6*xy*yz*xz


def tensor_square(a):
    """Mandel representation of the Cartesian matrix product A @ A."""
    x,y,z,u,v,w=a.v
    r=math.sqrt(2)
    return Sym((x*x+(u*u+w*w)/2, y*y+(u*u+v*v)/2,
                z*z+(v*v+w*w)/2, (x+y)*u+v*w/r,
                (y+z)*v+u*w/r, (x+z)*w+u*v/r))


def determinant(a):
    """Exact-input Mandel determinant, independently rounded to binary64."""
    from .numerics.determinant import determinant as exact_determinant
    if not isinstance(a, Sym):
        raise ValidationError('determinant requires a six-component symmetric tensor')
    try:
        return exact_determinant(a.v)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def normalized_determinant(a):
    """Exact-input homogeneous determinant ratio; dimensionless result."""
    from .numerics.normalized_determinant import normalized_determinant as exact_ratio
    if not isinstance(a, Sym):
        raise ValidationError('normalized_determinant requires a six-component symmetric tensor')
    try:
        return exact_ratio(a.v)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def tensor_expm(a):
    from .symmetric_expm import symmetric_expm
    if not isinstance(a,Sym):raise ValidationError('expm requires dimensionless symmetric tensor')
    try:return Sym(tuple(symmetric_expm(a.v)))
    except (ValueError,ArithmeticError) as exc:raise ValidationError(str(exc)) from exc


FUNCTIONS = {"sym": lambda *values: Sym(tuple(values)), "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
             "log1p": checked_log1p, "sin": math.sin, "cos": math.cos, "abs": abs,
             "min": min, "max": max, "trace": trace, "dev": dev,
             "dot": dot, "norm": lambda a: math.sqrt(dot(a, a)),
             "cube_trace": cube_trace, "tensor_square": tensor_square, "expm": tensor_expm,
             "determinant": determinant, "normalized_determinant": normalized_determinant}


def _spectral_function(operation, shape):
    def apply(a):
        if not isinstance(a, Sym):
            raise ValidationError('spectral operations require a symmetric tensor')
        try:
            result = spectral(a.v, operation)
        except (ValueError, ArithmeticError) as exc:
            raise ValidationError(str(exc)) from exc
        return Sym(result) if shape == 'sym' else result
    return apply


FUNCTIONS.update({name: _spectral_function(operation, shape)
                  for name, (operation, shape, _) in SPECTRAL_FUNCTIONS.items()})


def _same_physical_type(a, b):
    return a.shape == b.shape and a.dim == b.dim


def has_value_only_dependency(node, aliases=()):
    """Include predicate dependencies for reset events, whose policy is stricter."""
    return any((isinstance(n, ast.Name) and n.id in aliases) or
               (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and
                n.func.id in SPECTRAL_FUNCTIONS and SPECTRAL_FUNCTIONS[n.func.id][2])
               for n in ast.walk(node))


def parse(expression):
    if not isinstance(expression, str) or len(expression) > 10000:
        raise ValidationError("expression must be a string of at most 10000 characters")
    try:
        tree = ast.parse(expression, mode="eval").body
    except (SyntaxError, RecursionError) as exc:
        raise ValidationError(f"invalid expression: {expression}") from exc
    if sum(1 for _ in ast.walk(tree)) > 1000:
        raise ValidationError("expression is too complex")
    return tree


def constant_fraction(node):
    """Exact bounded-expression literals for dimensional powers, e.g. 1/3."""
    if isinstance(node,ast.Constant) and type(node.value) in (int,float) and math.isfinite(node.value):
        return Fraction(str(node.value))
    if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
        value=constant_fraction(node.operand)
        return -value if isinstance(node.op,ast.USub) else value
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div):
        return constant_fraction(node.left)/constant_fraction(node.right)
    raise ValueError('nonconstant dimensional exponent')


def infer(node, symbols):
    if isinstance(node, ast.Constant):
        if type(node.value) not in (float, int) or not math.isfinite(node.value):
            raise ValidationError("only finite numeric literals are allowed")
        return Type()
    if isinstance(node, ast.Name):
        if node.id not in symbols:
            raise ValidationError(f"undefined symbol: {node.id}")
        return symbols[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a = infer(node.operand, symbols)
        if a.shape not in ('scalar', 'sym'):
            raise ValidationError('numeric unary operators require numeric operands')
        return a
    if isinstance(node, ast.BinOp):
        a, b = infer(node.left, symbols), infer(node.right, symbols)
        if a.shape not in ('scalar', 'sym') or b.shape not in ('scalar', 'sym'):
            raise ValidationError('arithmetic requires numeric operands')
        value_only = a.value_only or b.value_only
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if not _same_physical_type(a, b):
                raise ValidationError(f"addition/subtraction requires equal types: {a}, {b}")
            return Type(a.shape, a.dim, value_only)
        if isinstance(node.op, (ast.Mult, ast.Div)):
            if (a.shape == b.shape == "sym") or (isinstance(node.op, ast.Div) and b.shape != "scalar"):
                raise ValidationError("tensor multiplication is explicit: use dot; divisor must be scalar")
            sign = -1 if isinstance(node.op, ast.Div) else 1
            return Type("sym" if "sym" in (a.shape, b.shape) else "scalar",
                        tuple(x + sign*y for x, y in zip(a.dim, b.dim)), value_only)
        if isinstance(node.op, ast.Pow):
            if a.shape != "scalar" or not _same_physical_type(b, Type()):
                raise ValidationError("power requires scalar base and dimensionless exponent")
            if a.dim == ZERO:
                return Type(value_only=value_only)
            try:
                exponent = constant_fraction(node.right)
            except (ValueError, TypeError, ZeroDivisionError):
                raise ValidationError("dimensioned power requires a constant exponent")
            return Type(dim=tuple(x*exponent for x in a.dim), value_only=value_only)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Gt, ast.GtE, ast.Lt, ast.LtE)):
        a, b = infer(node.left, symbols), infer(node.comparators[0], symbols)
        if not _same_physical_type(a, b) or a.shape != "scalar":
            raise ValidationError("comparison requires equally dimensioned scalars")
        # A comparison consumes value-only numeric dependencies as primal control.
        return Type("bool")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        args = [infer(a, symbols) for a in node.args]
        if name == "select" and len(args) == 3:
            if args[0] != Type("bool") or not _same_physical_type(args[1], args[2]):
                raise ValidationError("select requires condition and equally typed branches")
            return Type(args[1].shape, args[1].dim, args[1].value_only or args[2].value_only)
        value_only = any(a.value_only for a in args)
        if name == "sym":
            if len(args) != 6 or any(a.shape != "scalar" or not _same_physical_type(a, args[0]) for a in args):
                raise ValidationError("sym requires six equally dimensioned scalars in Mandel order")
            return Type("sym", args[0].dim, value_only)
        arity = 2 if name in ("dot", "min", "max") else 1
        if name not in FUNCTIONS or len(args) != arity:
            raise ValidationError(f"unsupported function or arity: {name}")
        a = args[0]
        if name in SPECTRAL_FUNCTIONS:
            _, shape, primitive_value_only = SPECTRAL_FUNCTIONS[name]
            if a.shape != 'sym':
                raise ValidationError(f'{name} requires a symmetric tensor')
            return Type(shape, ZERO if shape == 'sym' else a.dim,
                        value_only or primitive_value_only)
        if name == "expm":
            if not _same_physical_type(a, Type("sym")):
                raise ValidationError("expm requires dimensionless symmetric tensor")
            return a
        if name in ("trace", "dev", "norm", "cube_trace", "dot", "tensor_square", "determinant", "normalized_determinant"):
            if a.shape != "sym" or (name == "dot" and args[1].shape != "sym"):
                raise ValidationError(f"{name} requires symmetric tensors")
            if name == "normalized_determinant":
                return Type(value_only=value_only)
            if name == "dot":
                return Type(dim=tuple(x+y for x,y in zip(a.dim,args[1].dim)), value_only=value_only)
            return Type("sym" if name in ("dev", "tensor_square") else "scalar",
                        tuple(x*(3 if name in ("cube_trace", "determinant") else 2 if name == "tensor_square" else 1) for x in a.dim), value_only)
        if a.shape != "scalar":
            raise ValidationError(f"{name} requires scalar")
        if name in ("min", "max") and not _same_physical_type(a, args[1]):
            raise ValidationError(f"{name} arguments must have matching dimensions")
        if name in ("exp", "log", "log1p", "sin", "cos") and a.dim != ZERO:
            raise ValidationError(f"{name} requires dimensionless argument")
        return Type(dim=tuple(x/2 for x in a.dim) if name == "sqrt" else a.dim, value_only=value_only)
    raise ValidationError(f"unsupported syntax: {ast.dump(node)}")


def _finite_expression_value(value):
    """Validate executed AST results; primitive-internal arithmetic is separate."""
    if isinstance(value, bool):
        return value
    components = value.v if isinstance(value, Sym) else (value,)
    if isinstance(value, Sym) and len(components) != 6:
        raise ValidationError("expression tensor must have six components")
    for component in components:
        try:
            finite = type(component) in (int, float) and math.isfinite(component)
        except OverflowError:
            finite = False
        if not finite:
            raise ValidationError("nonfinite or invalid executed expression value")
    return value


def evaluate(node, symbols):
    return _finite_expression_value(_evaluate_node(node, symbols))


def _evaluate_node(node, symbols):
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        return symbols[node.id]
    if isinstance(node, ast.UnaryOp):
        value = evaluate(node.operand, symbols)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        a, b = evaluate(node.left, symbols), evaluate(node.right, symbols)
        if isinstance(node.op, ast.Add): return a+b
        if isinstance(node.op, ast.Sub): return a-b
        if isinstance(node.op, ast.Mult): return a*b
        if isinstance(node.op, ast.Div): return a/b
        if isinstance(node.op, ast.Pow): return math.pow(a,b)
    if isinstance(node, ast.Compare):
        a,b = evaluate(node.left, symbols),evaluate(node.comparators[0], symbols)
        op = node.ops[0]
        if isinstance(op, ast.Gt): return a>b
        if isinstance(op, ast.GtE): return a>=b
        if isinstance(op, ast.Lt): return a<b
        return a<=b
    if isinstance(node, ast.Call):
        name = node.func.id
        if name == "select":
            return evaluate(node.args[1] if evaluate(node.args[0], symbols) else node.args[2], symbols)
        return FUNCTIONS[name](*(evaluate(a, symbols) for a in node.args))
    raise ValidationError("invalid AST")


def cpp(node):
    # Function-body flag writes are sequenced; the false flag is sticky.
    return f"atlas::expression_or_mark({_cpp_node(node)}, atlas_eval_ok)"


def _cpp_node(node):
    if isinstance(node, ast.Constant): return f"T({float(node.value)!r})"
    if isinstance(node, ast.Name): return node.id
    if isinstance(node, ast.UnaryOp): return f"({'-' if isinstance(node.op,ast.USub) else '+'}{cpp(node.operand)})"
    if isinstance(node, ast.BinOp):
        a,b=cpp(node.left),cpp(node.right)
        if isinstance(node.op,ast.Pow): return f"apow({a}, {b})"
        op={ast.Add:'+',ast.Sub:'-',ast.Mult:'*',ast.Div:'/'}[type(node.op)]
        return f"({a} {op} {b})"
    if isinstance(node,ast.Compare):
        op={ast.Gt:'>',ast.GtE:'>=',ast.Lt:'<',ast.LtE:'<='}[type(node.ops[0])]
        return f"({cpp(node.left)} {op} {cpp(node.comparators[0])})"
    if isinstance(node,ast.Call):
        name=node.func.id
        if name in ('trace', 'dev', 'determinant', 'normalized_determinant'): return f"atlas::{name}_or_mark({cpp(node.args[0])}, atlas_eval_ok)"
        if name in SPECTRAL_FUNCTIONS:
            operation, shape, _ = SPECTRAL_FUNCTIONS[name]
            helper = 'projector' if shape == 'sym' else 'scalar'
            return f"atlas::spectral_{helper}_or_mark({cpp(node.args[0])}, atlas::spectral_candidate::Operation::{operation}, atlas_eval_ok)"
        if name=='log1p': return f"atlas::scalar_log1p_or_mark({cpp(node.args[0])}, atlas_eval_ok)"
        if name=='expm': return f"atlas::tensor_expm_or_mark({cpp(node.args[0])}, atlas_eval_ok)"
        if name=='select': return f"({cpp(node.args[0])} ? {cpp(node.args[1])} : {cpp(node.args[2])})"
        name={'tensor_square':'atlas::tensor_square','sym':'make_sym','min':'amin','max':'amax','sqrt':'asqrt','exp':'aexp','log':'alog','sin':'asin','cos':'acos','abs':'aabs'}.get(name,name)
        return f"{name}({', '.join(cpp(a) for a in node.args)})"
    raise ValidationError("invalid AST")
