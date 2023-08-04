use halo2_base::QuantumCell;
use halo2_base::{
    gates::{GateInstructions, GateChip, RangeChip, RangeInstructions},
    utils::{ScalarField},
    AssignedValue, Context,
};
use itertools::Itertools;
use crate::fixed_point::{FixedPointChip, FixedPointInstructions};
use std::f64::consts::PI;


pub fn std_normal<F: ScalarField>(
    ctx: &mut Context<F>,
    fixed_point: &FixedPointChip<F, 63>,
    x: &AssignedValue<F>
) -> AssignedValue<F> {
    // sqrt(2*pi).
    let sqrt_two_pi = QuantumCell::Constant(fixed_point.quantization(PI * 2.0));

    // e^(-x^2/2)
    let x_squared = fixed_point.qmul(ctx, *x, *x);
    let neg_x_squared = fixed_point.neg(ctx, x_squared);
    let neg_x_squared_div_two = fixed_point.qdiv(ctx, neg_x_squared, QuantumCell::Constant(F::from(2)));
    let exp = fixed_point.qexp(ctx, neg_x_squared_div_two);

    // e^(-x^2/2) / sqrt(2*pi)
    fixed_point.qdiv(ctx, exp, sqrt_two_pi)
}

// Use Abramowitz and Stegun approximation
pub fn std_normal_cdf<F: ScalarField>(
    ctx: &mut Context<F>,
    fixed_point: &FixedPointChip<F, 63>,
    x: &AssignedValue<F>
) -> AssignedValue<F> {
    if fixed_point.dequantization(*x.value()) < -5.0 {
        return ctx.load_constant(fixed_point.quantization(0.0));
    } else if fixed_point.dequantization(*x.value()) > 5.0 {
        return ctx.load_constant(fixed_point.quantization(1.0));
    }

    let b1 = QuantumCell::Constant(fixed_point.quantization(0.319381530));
    let b2 = QuantumCell::Constant(fixed_point.quantization(-0.356563782));
    let b3 = QuantumCell::Constant(fixed_point.quantization(1.781477937));
    let b4 = QuantumCell::Constant(fixed_point.quantization(-1.821255978));
    let b5 = QuantumCell::Constant(fixed_point.quantization(1.330274429));
    let p = QuantumCell::Constant(fixed_point.quantization(0.2316419));
    let c2 = QuantumCell::Constant(fixed_point.quantization(0.3989423));
    let one = QuantumCell::Constant(fixed_point.quantization(1.0));
    let two = QuantumCell::Constant(fixed_point.quantization(2.0));
    
    let abs_x = fixed_point.qabs(ctx, *x);
    let t = {
        let tmp1 = fixed_point.qmul(ctx, p, *x);
        let tmp2 = fixed_point.qadd(ctx, one, tmp1);
        fixed_point.qdiv(ctx, one, tmp2)
    };
    let b = {
        let tmp1 = fixed_point.qmul(ctx, abs_x, abs_x);
        let tmp2 = fixed_point.qdiv(ctx, tmp1, two);
        let tmp3 = fixed_point.qexp(ctx, tmp2);
        fixed_point.qmul(ctx, c2, tmp3)
    };
    let n = {
        let tmp1 = fixed_point.qmul(ctx, b5, t);
        let tmp2 = fixed_point.qadd(ctx, b4, tmp1);
        let tmp3 = fixed_point.qmul(ctx, t, tmp2);
        let tmp4 = fixed_point.qadd(ctx, b3, tmp3);
        let tmp5 = fixed_point.qmul(ctx, t, tmp4);
        let tmp6 = fixed_point.qadd(ctx, b2, tmp5);
        let tmp7 = fixed_point.qmul(ctx, t, tmp6);
        let tmp8 = fixed_point.qadd(ctx, b1, tmp7);
        let tmp9 = fixed_point.qmul(ctx, t, tmp8);
        let tmp10 = fixed_point.qmul(ctx, tmp9, b);
        fixed_point.qsub(ctx, one, tmp10)
    };
    if fixed_point.dequantization(*x.value()) < -5.0 {
        fixed_point.qsub(ctx, one, n)
    } else {
        n
    }
}

// Returns the internal Black-Scholes coefficients.
pub fn d1d2<F: ScalarField> (
    ctx: &mut Context<F>,
    fixed_point: &FixedPointChip<F, 63>,
    t_annualized: &AssignedValue<F>,
    volatility: &AssignedValue<F>,
    spot: &AssignedValue<F>,
    strike: &AssignedValue<F>,
    rate: &AssignedValue<F>
) -> (AssignedValue<F>, AssignedValue<F>) {
    let d1 = {
        // ln(spot / strike)
        let a = fixed_point.qdiv(ctx, *spot, *strike);
        let ln_a = fixed_point.qlog(ctx, a);
        let vol_sq = fixed_point.qmul(ctx, *volatility, *volatility);
        let two = QuantumCell::Constant(fixed_point.quantization(2.0));
        let b = fixed_point.qadd(ctx, *rate, two);
        let c = fixed_point.qdiv(ctx, vol_sq, b);
        let d = fixed_point.qmul(ctx, c, *t_annualized);

        let t_annualized_sqrt = fixed_point.qsqrt(ctx, *t_annualized);
        let e = fixed_point.qmul(ctx, *volatility, t_annualized_sqrt);

        fixed_point.qdiv(ctx, d, e)
    };

    let d2 = {
        let t_annualized_sqrt = fixed_point.qsqrt(ctx, *t_annualized);
        let a = fixed_point.qmul(ctx, *volatility, t_annualized_sqrt);
        fixed_point.qsub(ctx, d1, a)
    };
    (d1, d2)
}