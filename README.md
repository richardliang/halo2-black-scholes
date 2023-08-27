# halo2-black-scholes

This is a halo2 library that implements a Black Scholes chip using [halo2-lib](https://github.com/axiom-crypto/halo2-lib/) and [fixed point chip](https://github.com/DCMMC/ZKFixedPointChip)

**WARNING: Not audited, do not use in production**

## Features
**BlackScholesChip**

Proves the following
* option_prices
* delta
* gamma
* vega
* theta
* rho

Helpers
* Normal distribution
* Cumulative normal distribution

Todos
* implied_volatility
* calculate_all

## Usage 
Install rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Run tests
```
cargo test -- --nocapture
```


## Benchmarks
TODO

## References
* [Cairo Black Scholes](https://github.com/0xSachaEth/black-scholes)
* [Rust Black Scholes](https://github.com/danielhstahl/black_scholes_rust)
