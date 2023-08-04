use halo2_base::QuantumCell;
use halo2_base::{
    gates::{GateInstructions, GateChip, RangeChip, RangeInstructions},
    utils::{ScalarField},
    AssignedValue, Context,
};
use itertools::Itertools;

mod utils;
mod fixed_point;
/*
def black_scholes(call_put_flag, S, X, T, r, q, sigma):
    d1 = (ln(S / X) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * (T ** 0.5))
    d2 = d1 - sigma * (T ** 0.5)

    if call_put_flag == 'c':
        option_price = S * exp(-q * T) * cdf(d1) - X * exp(-r * T) * cdf(d2)
    elif call_put_flag == 'p':
        option_price = X * exp(-r * T) * cdf(-d2) - S * exp(-q * T) * cdf(-d1)
    else:
        raise ValueError("Call/Put flag must be 'c' or 'p'")

    return option_price

def pdf(x):
    return (1.0 / (2.0 * pi) ** 0.5) * exp(-0.5 * x * x)

def cdf(x, terms=100):
    if x < 0:
        return 1.0 - cdf(-x)
    h = x / terms
    s = pdf(0) + pdf(x)
    for i in range(1, terms, 2):
        s += 4 * pdf(i * h)
    for i in range(2, terms-1, 2):
        s += 2 * pdf(i * h)
    return (h / 3) * s

def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

def exp(x, terms=100):
    return sum((x**n) / factorial(n) for n in range(terms))

def ln(x, terms=100):
    return sum(((-1)**n) * ((x - 1)**(n + 1)) / (n + 1) for n in range(terms))

*/

// const UNIT: i128 = 10_i128.pow(27);
// const SQRT_TWOPI: i128 = 2506628274631000543434113024;

// const MIN_CDF_INPUT: i128 = -5 * UNIT;
// const MAX_CDF_INPUT: i128 = 5 * UNIT;

// const MIN_T_ANNUALISED: i128 = 31709791983764586496;
// const MIN_VOLATILITY: i128 = UNIT / 10000;
// const DIV_BOUND: i128 = 2_i128.pow(128) / 2;
