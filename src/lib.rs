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
use black_scholes;

#[macro_use]
extern crate approx;

#[derive(Clone, Debug)]
pub struct BlackScholesChip<F: ScalarField> {
    pub fixed_point: FixedPointChip<F, 63>,
    pub lookup_bits: usize,
    _marker: PhantomData<F>,
}

// TODO: Implement greeks
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
        t_annualized: f64, 
        volatility: f64, 
        spot: f64, 
        strike: f64, 
        rate: f64
    ) -> (AssignedValue<F>, AssignedValue<F>) {
        // Load witnesses
        let t_annualized = ctx.load_witness(self.fixed_point.quantization(t_annualized));
        let volatility = ctx.load_witness(self.fixed_point.quantization(volatility));
        let spot = ctx.load_witness(self.fixed_point.quantization(spot));
        let strike = ctx.load_witness(self.fixed_point.quantization(strike));
        let rate = ctx.load_witness(self.fixed_point.quantization(rate));

        let (d1, d2) = d1d2(
            ctx, &self.fixed_point,
            &t_annualized,
            &volatility,
            &spot,
            &strike,
            &rate
        );

        let exp = {
            let a = self.fixed_point.qmul(ctx, rate, t_annualized);
            let a = self.fixed_point.neg(ctx, a);
            self.fixed_point.qexp(ctx, a)
        };

        let strike_pv = self.fixed_point.qmul(ctx, strike, exp);

        let spot_nd1 = {
            let a = std_normal_cdf(ctx, &self.fixed_point, &d1);
            self.fixed_point.qmul(ctx, spot, a)
        };

        let strike_nd2 = {
            let a = std_normal_cdf(ctx, &self.fixed_point, &d2);
            self.fixed_point.qmul(ctx, strike_pv, a)
        };

        let call_price = self.fixed_point.qsub(ctx, spot_nd1, strike_nd2);

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
    let two = ctx.load_constant(fixed_point.quantization(2.0));

    // sqrt(2*pi).
    let sqrt_two_pi = ctx.load_constant(fixed_point.quantization(PI * 2.0));

    // e^(-x^2/2)
    let x_squared = fixed_point.qmul(ctx, *x, *x);
    let neg_x_squared = fixed_point.neg(ctx, x_squared);
    let neg_x_squared_div_two = fixed_point.qdiv(ctx, neg_x_squared, two);
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
    // Load magic numbers
    let b1 = ctx.load_constant(fixed_point.quantization(0.319381530));
    let b2 = ctx.load_constant(fixed_point.quantization(-0.356563782));
    let b3 = ctx.load_constant(fixed_point.quantization(1.781477937));
    let b4 = ctx.load_constant(fixed_point.quantization(-1.821255978));
    let b5 = ctx.load_constant(fixed_point.quantization(1.330274429));
    let p = ctx.load_constant(fixed_point.quantization(0.2316419));
    let c2 = ctx.load_constant(fixed_point.quantization(0.3989423));

    let one = ctx.load_constant(fixed_point.quantization(1.0));
    let two = ctx.load_constant(fixed_point.quantization(2.0));
    
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

    let neg_five = ctx.load_constant(fixed_point.quantization(-5.0));
    let zero = ctx.load_constant(fixed_point.quantization(0.0));
    let five = ctx.load_constant(fixed_point.quantization(5.0));
    let num_bits = 126; // PRECISION_BITS * 2

    // Check if negative
    let lt_zero = fixed_point.range_gate().is_less_than(ctx, *x, zero, num_bits);
    let result = {
        let a = fixed_point.qsub(ctx, one, n);
        fixed_point.range_gate().gate().select(ctx, a, n, lt_zero)
    };

    // Check less than -5.0
    let lt_neg_five = fixed_point.range_gate().is_less_than(ctx, *x, neg_five, num_bits);
    let result = fixed_point.range_gate().gate().select(ctx, QuantumCell::Constant(fixed_point.quantization(0.0)), result, lt_neg_five);

    // Check greater than 5.0
    let gt_five = fixed_point.range_gate().is_less_than(ctx, five, *x, num_bits);
    fixed_point.range_gate().gate().select(ctx, QuantumCell::Constant(fixed_point.quantization(1.0)), result, gt_five)
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
    let two = ctx.load_constant(fixed_point.quantization(2.0));
    let t_annualized_sqrt = fixed_point.qsqrt(ctx, *t_annualized);
    // volatility * sqrt(t_annualized)
    let denominator = fixed_point.qmul(ctx, *volatility, t_annualized_sqrt);

    // ( ln(spot / strike) + ( ( (volatility * volatility) / 2 + rate) * t_annualized) ) / (volatility * sqrt(t_annualized) )
    let d1 = {
        // ln(spot / strike)
        let a = fixed_point.qdiv(ctx, *spot, *strike);
        let ln_a = fixed_point.qlog(ctx, a);
        // (volatility * volatility) / 2 + rate) * t_annualized
        let numerator = fixed_point.qmul(ctx, *volatility, *volatility);
        let numerator = fixed_point.qdiv(ctx, numerator, two);
        let numerator = fixed_point.qadd(ctx, numerator, *rate);
        let numerator = fixed_point.qmul(ctx, numerator, *t_annualized);
        let numerator = fixed_point.qadd(ctx, numerator, ln_a);
        fixed_point.qdiv(ctx, numerator, denominator)
    };
    
    let d2 = fixed_point.qsub(ctx, d1, denominator);

    (d1, d2)
}

#[cfg(test)]
mod test {
    use super::*;
    use halo2_base::gates::builder::{GateThreadBuilder, RangeWithInstanceCircuitBuilder};
    use halo2_base::halo2_proofs::{halo2curves::bn256::Fr, dev::MockProver};

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

        // Test 1.0
        let x = 1.0;
        let x = ctx.load_witness(fixed_point.quantization(x));
        let result = std_normal_cdf(
            ctx,
            &fixed_point,
            &x,
        );
        let result = fixed_point.dequantization(*result.value());
        let expected = 0.8413447;
        // Abramowitz and Stegun approximation is accurate to about 7 decimals
        let err = 1e-7;
        assert_relative_eq!(result, expected, max_relative = err);

        // Test -1.0
        let x = -1.0;
        let x = ctx.load_witness(fixed_point.quantization(x));
        let result = std_normal_cdf(
            ctx,
            &fixed_point,
            &x,
        );
        let result = fixed_point.dequantization(*result.value());
        // Same even if negative
        let expected = 0.15865526;
        let err = 1e-7;
        assert_relative_eq!(result, expected, max_relative = err);

        // Test 6.0
        let x = 6.0;
        let x = ctx.load_witness(fixed_point.quantization(x));
        let result = std_normal_cdf(
            ctx,
            &fixed_point,
            &x,
        );
        let result = fixed_point.dequantization(*result.value());
        let expected = 1.0;
        let err = 1e-7;
        assert_relative_eq!(result, expected, max_relative = err);

        // Test -6.0
        let x = -6.0;
        let x = ctx.load_witness(fixed_point.quantization(x));
        let result = std_normal_cdf(
            ctx,
            &fixed_point,
            &x,
        );
        let result = fixed_point.dequantization(*result.value());
        let expected = 0.0;
        let err = 1e-7;
        assert_relative_eq!(result, expected, max_relative = err);
    }

    #[test]
    fn test_d1d2() {
        let k = 9;
        // Configure builder
        let mut builder = GateThreadBuilder::<Fr>::mock();
        let lookup_bits = k - 1;
        // NOTE: Need to set var to load lookup table
        std::env::set_var("LOOKUP_BITS", lookup_bits.to_string());
    
        let fixed_point = FixedPointChip::<Fr, 63>::default(lookup_bits);
        let ctx = builder.main(0);

        // Test values
        let t_annualized = 1.0;
        let volatility = 0.2;
        let spot = 50.0;
        let strike = 100.0;
        let rate = 0.05;

        // Calculate expected d1 d2
        fn calculate_d1_d2(spot: f64, strike: f64, t_annualized: f64, rate: f64, volatility: f64) -> (f64, f64) {
            let d1 = ((spot / strike).ln() + (rate + 0.5 * volatility.powi(2)) * t_annualized) / (volatility * t_annualized.sqrt());
            let d2 = d1 - volatility * t_annualized.sqrt();
            (d1, d2)
        }
        let (expected_d1, expected_d2) = calculate_d1_d2(spot, strike, t_annualized, rate, volatility);

        // Get actual values
        let t_annualized = ctx.load_witness(fixed_point.quantization(t_annualized));
        let volatility = ctx.load_witness(fixed_point.quantization(volatility));
        let spot = ctx.load_witness(fixed_point.quantization(spot));
        let strike = ctx.load_witness(fixed_point.quantization(strike));
        let rate = ctx.load_witness(fixed_point.quantization(rate));

        let result = d1d2(
            ctx,
            &fixed_point,
            &t_annualized,
            &volatility,
            &spot,
            &strike,
            &rate
        );
        let d1 = fixed_point.dequantization(*result.0.value());
        let d2 = fixed_point.dequantization(*result.1.value());
        let err = 1e-7;

        assert_relative_eq!(d1, expected_d1, max_relative = err);
        assert_relative_eq!(d2, expected_d2, max_relative = err);
    }

    #[test]
    fn test_black_scholes() {
        // Uncomment to enable RUST_LOG
        // env_logger::init();

        let k = 16;
        // Configure builder
        let mut builder = GateThreadBuilder::<Fr>::mock();
        let lookup_bits = 8;
        // NOTE: Need to set var to load lookup table
        std::env::set_var("LOOKUP_BITS", lookup_bits.to_string());
        
        // Circuit inputs
        let t_annualized = 1.0;
        let volatility = 0.2;
        let spot = 50.0;
        let strike = 100.0;
        let rate = 0.05;

        // Configure black scholes chip
        let chip = BlackScholesChip::<Fr>::new(lookup_bits);
        let (call, put) = chip.black_scholes(
            builder.main(0),
            t_annualized,
            volatility,
            spot,
            strike,
            rate
        );

        // Assign public instances to circuit
        let mut instances = vec![];
        instances.push(call);
        instances.push(put);

        // Minimum rows is the number of rows used for blinding factors
        // This depends on the circuit itself, but we can guess the number and change it if something breaks (default 9 usually works)
        builder.config(k, Some(9));
        // Create mock circuit
        let circuit = RangeWithInstanceCircuitBuilder::mock(builder, instances);

        let mut test_public_inputs = vec![];
        let expected_call = black_scholes::call(spot, strike, rate, volatility, t_annualized);
        let expected_put = black_scholes::put(spot, strike, rate, volatility, t_annualized);
        
        // TODO: improve the error tolerance in circuit
        // One way is to calculate put first and use put-call parity to calculate call price in certain scenarios
        let err = 1e-3;
        assert_relative_eq!(chip.fixed_point.dequantization(*call.value()), expected_call, max_relative = err);
        assert_relative_eq!(chip.fixed_point.dequantization(*put.value()), expected_put, max_relative = err);
        
        test_public_inputs.push(*call.value());
        test_public_inputs.push(*put.value());
        
        // let result_call = chip.fixed_point.dequantization(*call.value());
        // let result_put = chip.fixed_point.dequantization(*put.value());
        // println!("Call: {:?}, Put: {:?}", result_call, result_put);
        // println!("Expected Call: {}, Expected Put: {}", expected_call, expected_put);

        // Run mock prover to ensure output is correct
        MockProver::run(k as u32, &circuit, vec![test_public_inputs]).unwrap().assert_satisfied();
    }

    #[cfg(feature = "dev-graph")]
    #[test]
    fn plot_black_scholes() {
        use plotters::prelude::*;

        // Uncomment to enable RUST_LOG
        env_logger::init();

        let k = 16;
        // Configure builder
        let mut builder = GateThreadBuilder::<Fr>::mock();
        let lookup_bits = 12;
        // NOTE: Need to set var to load lookup table
        std::env::set_var("LOOKUP_BITS", lookup_bits.to_string());
        
        // Circuit inputs
        let t_annualized = 1.0;
        let volatility = 0.2;
        let spot = 50.0;
        let strike = 100.0;
        let rate = 0.05;

        // Configure black scholes chip
        let chip = BlackScholesChip::<Fr>::new(lookup_bits);
        let (call, put) = chip.black_scholes(
            builder.main(0),
            t_annualized,
            volatility,
            spot,
            strike,
            rate
        );

        // Assign public instances to circuit
        let mut instances = vec![];
        instances.push(call);
        instances.push(put);

        // Minimum rows is the number of rows used for blinding factors
        // This depends on the circuit itself, but we can guess the number and change it if something breaks (default 9 usually works)
        builder.config(k, Some(9));
        // Create mock circuit
        let circuit = RangeWithInstanceCircuitBuilder::mock(builder, instances);

        // Plot layout
        let root = BitMapBackend::new("bs_options_price.png", (1024, 1024)).into_drawing_area();
        root.fill(&WHITE).unwrap();
        let root = root.titled("BS Option Price Circuit Layout", ("sans-serif", 60)).unwrap();
        
        halo2_base::halo2_proofs::dev::CircuitLayout::default()
            // The first argument is the size parameter for the circuit.
            .render((k) as u32, &circuit, &root)
            .unwrap();
    }
}