use halo2_base::QuantumCell;
use halo2_base::{
    gates::{GateInstructions, GateChip, RangeChip, RangeInstructions},
    utils::{ScalarField},
    AssignedValue, Context,
};
use itertools::Itertools;
mod fixed_point;
pub use fixed_point::*;
use std::f64::consts::PI;
use std::marker::PhantomData;

#[macro_use]
extern crate approx;

#[derive(Clone, Debug)]
pub struct BlackScholesChip<F: ScalarField> {
    pub fixed_point: FixedPointChip<F, 63>,
    pub lookup_bits: usize,
    _marker: PhantomData<F>,
}

impl <'range, F: ScalarField> BlackScholesChip<F> {
    pub fn new(lookup_bits: usize) -> Self {
        let fixed_point: FixedPointChip<F, 63> = FixedPointChip::<F, 63>::default(lookup_bits);

        Self {
            fixed_point,
            lookup_bits,
            _marker: PhantomData,
        }
    }

    pub fn black_scholes(
        &self,
        ctx: &mut Context<F>,
        t_annualised: f64, 
        volatility: f64, 
        spot: f64, 
        strike: f64, 
        rate: f64
    ) -> (AssignedValue<F>, AssignedValue<F>) {
        // Load witnesses
        let t_annualised = ctx.load_witness(self.fixed_point.quantization(t_annualised));
        let volatility = ctx.load_witness(self.fixed_point.quantization(volatility));
        let spot = ctx.load_witness(self.fixed_point.quantization(spot));
        let strike = ctx.load_witness(self.fixed_point.quantization(strike));
        let rate = ctx.load_witness(self.fixed_point.quantization(rate));

        let (d1, d2) = d1d2(
            ctx, &self.fixed_point,
            &t_annualised,
            &volatility,
            &spot,
            &strike,
            &rate
        );

        let exp = {
            let a = self.fixed_point.neg(ctx, rate);
            let a = self.fixed_point.qmul(ctx, a, t_annualised);
            self.fixed_point.qexp(ctx, a)
        };

        let strike_pv = self.fixed_point.qmul(ctx, strike, exp);

        let spot_nd1 = {
            let a = std_normal_cdf(ctx, &self.fixed_point, &d1);
            self.fixed_point.qmul(ctx, spot, a)
        };

        let strike_nd1 = {
            let a = std_normal_cdf(ctx, &self.fixed_point, &d2);
            self.fixed_point.qmul(ctx, strike_pv, a)
        };

        let call_price = self.fixed_point.qsub(ctx, spot_nd1, strike_nd1);

        let put_price = {
            let a = self.fixed_point.qadd(ctx, call_price, strike_pv);
            self.fixed_point.qsub(ctx, a, spot)
        };

        (call_price, put_price)
    }
}

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

    // Load magic numbers
    let b1 = QuantumCell::Constant(fixed_point.quantization(0.319381530));
    let b2 = QuantumCell::Constant(fixed_point.quantization(-0.356563782));
    let b3 = QuantumCell::Constant(fixed_point.quantization(1.781477937));
    let b4 = QuantumCell::Constant(fixed_point.quantization(-1.821255978));
    let b5 = QuantumCell::Constant(fixed_point.quantization(1.330274429));
    let p = QuantumCell::Constant(fixed_point.quantization(0.2316419));
    let c2 = QuantumCell::Constant(fixed_point.quantization(0.3989423));
    // Constants
    let one = QuantumCell::Constant(fixed_point.quantization(1.0));
    let two = QuantumCell::Constant(fixed_point.quantization(2.0));
    
    // abs(x)
    let abs_x = fixed_point.qabs(ctx, *x);
    // 1 / (1 + p * abs(x))
    let t = {
        let denominator = fixed_point.qmul(ctx, p, abs_x);
        let denominator = fixed_point.qadd(ctx, one, denominator);
        fixed_point.qdiv(ctx, one, denominator)
    };
    // c2 / exp((abs_x*abs_x) / 2);
    let b = {
        let denominator = fixed_point.qmul(ctx, abs_x, abs_x);
        let denominator = fixed_point.qdiv(ctx, denominator, two);
        let denominator = fixed_point.qexp(ctx, denominator);
        fixed_point.qdiv(ctx, c2, denominator)
    };
    // 1 - b * ((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t
    let n = {
        let res = fixed_point.qmul(ctx, b5, t);
        let res = fixed_point.qadd(ctx, b4, res);
        let res = fixed_point.qmul(ctx, t, res);
        let res = fixed_point.qadd(ctx, b3, res);
        let res = fixed_point.qmul(ctx, t, res);
        let res = fixed_point.qadd(ctx, b2, res);
        let res = fixed_point.qmul(ctx, t, res);
        let res = fixed_point.qadd(ctx, b1, res);
        let res = fixed_point.qmul(ctx, t, res);
        let res = fixed_point.qmul(ctx, res, b);
        fixed_point.qsub(ctx, one, res)
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
        let b = fixed_point.qadd(ctx, ln_a, vol_sq);

        let two = QuantumCell::Constant(fixed_point.quantization(2.0));
        let c = fixed_point.qadd(ctx, *rate, two);
        let d = fixed_point.qdiv(ctx, vol_sq, c);
        let e = fixed_point.qmul(ctx, d, *t_annualized);

        let t_annualized_sqrt = fixed_point.qsqrt(ctx, *t_annualized);
        let f = fixed_point.qmul(ctx, *volatility, t_annualized_sqrt);

        let g = fixed_point.qdiv(ctx, e, f);
        fixed_point.qadd(ctx, b, g)
    };

    let d2 = {
        let t_annualized_sqrt = fixed_point.qsqrt(ctx, *t_annualized);
        let a = fixed_point.qmul(ctx, *volatility, t_annualized_sqrt);
        fixed_point.qsub(ctx, d1, a)
    };
    (d1, d2)
}

#[cfg(test)]
mod test {
    use super::*;
    use halo2_base::gates::builder::GateThreadBuilder;
    use halo2_base::halo2_proofs::halo2curves::bn256::Fr;
    use halo2_base::QuantumCell::{Existing, Witness, Constant};
    use halo2_base::{
        gates::{GateInstructions, GateChip, RangeChip, RangeInstructions},
        utils::{ScalarField},
        AssignedValue, Context,
    };
    use itertools::Itertools;

    #[test]
    fn test_std_normal_cdf() {
        let k = 9;
        // Configure builder
        let mut builder = GateThreadBuilder::<Fr>::mock();
        let lookup_bits = k - 1;
        // NOTE: Need to set var to load lookup table
        std::env::set_var("LOOKUP_BITS", lookup_bits.to_string());
    
        let fixed_point = FixedPointChip::<Fr, 63>::default(lookup_bits);
        let ctx = builder.main(0);

        let x = 1.0;
        let x = ctx.load_witness(fixed_point.quantization(x));
        let result = std_normal_cdf(
            ctx,
            &fixed_point,
            &x,
        );
        // Dequantize result and compare
        let result = fixed_point.dequantization(*result.value());
    
        let expected = 0.8413447;
        let err = 1e-7;

        // Check less than error threshold
        assert_relative_eq!(result, expected, max_relative = err);
    }
}