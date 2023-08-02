use halo2_base::QuantumCell;
use halo2_base::{
    gates::{GateInstructions, RangeInstructions, RangeChip},
    utils::{ScalarField},
    AssignedValue, Context,
};
use itertools::Itertools;
use std::{marker::PhantomData};
