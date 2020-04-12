use crate::fa::{DifferentiableStateActionFunction, DifferentiableStateFunction};

pub use lfa::*;

import_all!(gradient);

pub(crate) fn dot_features(f1: &Features, f2: &Features) -> f64 {
    match (f1, f2) {
        (Features::Sparse(_, a1), Features::Sparse(_, a2)) => a1.iter().fold(0.0, |acc, (i, a)| {
            acc + a2.get(i).copied().map_or(0.0, |a2| a2 * a)
        }),
        (Features::Sparse(_, a1), Features::Dense(a2)) => a1.iter().fold(0.0, |acc, (&i, a)| {
            acc + a2.get(i).copied().map_or(0.0, |a2| a2 * a)
        }),
        (Features::Dense(a1), Features::Sparse(_, a2)) => a2.iter().fold(0.0, |acc, (&i, a)| {
            acc + a1.get(i).copied().map_or(0.0, |a1| a1 * a)
        }),
        (Features::Dense(a1), Features::Dense(a2)) => a1.dot(a2),
    }
}

pub trait LinearStateFunction<X: ?Sized>:
    DifferentiableStateFunction<X, Gradient = LFAGradient> + Parameterised
{
    fn n_features(&self) -> usize;

    fn features(&self, state: &X) -> Features;

    fn evaluate_features(&self, features: &Features) -> f64;

    fn update_features(&mut self, features: &Features, error: f64);
}

pub trait LinearStateActionFunction<X: ?Sized, U: ?Sized>:
    DifferentiableStateActionFunction<X, U, Gradient = LFAGradient> + Parameterised
{
    fn n_features(&self) -> usize;

    fn features(&self, state: &X, action: &U) -> Features;

    fn evaluate_features(&self, features: &Features, action: &U) -> f64;

    fn update_features(&mut self, features: &Features, action: &U, error: f64);
}

import_all!(vanilla);
import_all!(compatible);
