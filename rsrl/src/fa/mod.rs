//! Function approximation and value function representation module.
use std::ops::{
    Mul,
    Sub,
};

#[cfg(test)]
pub(crate) mod mocking;

pub mod linear;
pub mod tabular;

pub mod transforms;
import_all!(transformed);

import_all!(shared);

pub use self::linear::{Parameterised, Weights, WeightsView, WeightsViewMut};

/// An interface for state value functions.
pub trait StateFunction<X: ?Sized> {
    type Output;

    fn evaluate(&self, state: &X) -> Self::Output;

    fn update(&mut self, state: &X, error: Self::Output);
}

pub trait DifferentiableStateFunction<X: ?Sized>: StateFunction<X> + Parameterised {
    type Gradient: crate::linalg::MatrixLike;

    fn grad(&self, state: &X) -> Self::Gradient;

    fn update_grad<G: crate::linalg::MatrixLike>(&mut self, grad: &G) {
        grad.addto(&mut self.weights_view_mut());
    }

    fn update_grad_scaled<G: crate::linalg::MatrixLike>(&mut self, grad: &G, factor: f64) {
        grad.scaled_addto(factor, &mut self.weights_view_mut());
    }
}

/// An interface for state-action value functions.
pub trait StateActionFunction<X: ?Sized, U: ?Sized> {
    type Output : Mul<Output=Self::Output>+Sub<Output=Self::Output>;

    fn evaluate(&self, state: &X, action: &U) -> Self::Output;

    /// Update the function by giving a precomputed error.
    ///
    /// This is the raw error and not scale by learning rate.
    ///
    /// This is the default mode. Implementors can override `update` to circumvent the call to
    /// this method.
    fn update_by_error(&mut self, state: &X, action: &U, error: Self::Output);

    /// Update the function by giving the observed value and estimated value of a state-action pair
    ///
    /// This default implementation computes an estimated value of the state-action pair by calling
    /// `evaluate`. An error is then calculated and passed to `update_by_error`.
    ///
    /// Those wishing to do their own error calculation and their own value estimation---for example,
    /// as part of the forward pass of a neural network---should implement this method directly,
    /// in which case `update_by_error` will never be called
    fn update(&mut self, state: &X, action: &U, value: Self::Output, estimate: Self::Output, learning_rate: Self::Output) {
        let error = learning_rate * (value - estimate);
        self.update_by_error(state, action, error);
    }
}

pub trait DifferentiableStateActionFunction<X: ?Sized, U: ?Sized>:
    StateActionFunction<X, U> + Parameterised
{
    type Gradient: crate::linalg::MatrixLike;

    fn grad(&self, state: &X, action: &U) -> Self::Gradient;

    fn update_grad<G: crate::linalg::MatrixLike>(&mut self, grad: &G) {
        grad.addto(&mut self.weights_view_mut());
    }

    fn update_grad_scaled<G: crate::linalg::MatrixLike>(&mut self, grad: &G, factor: f64) {
        grad.scaled_addto(factor, &mut self.weights_view_mut());
    }
}

pub trait EnumerableStateActionFunction<X: ?Sized>:
    StateActionFunction<X, usize, Output = f64>
{
    fn n_actions(&self) -> usize;

    fn evaluate_all(&self, state: &X) -> Vec<f64>;

    fn update_all_by_errors(&mut self, state: &X, errors: Vec<f64>);

    fn update_all(&mut self, state: &X, values: Vec<f64>, estimates: Vec<f64>, learning_rate: f64) {
        let mut errors: Vec<f64> = Vec::with_capacity(values.len());
        for (i, value) in values.iter().enumerate() {
            errors.push(learning_rate * (*value - estimates[i]));
        }
        self.update_all_by_errors(state, errors);
    }

    fn find_min(&self, state: &X) -> (usize, f64) {
        let mut iter = self.evaluate_all(state).into_iter().enumerate();
        let first = iter.next().unwrap();

        iter.fold(first, |acc, (i, x)| if acc.1 < x { acc } else { (i, x) })
    }

    fn find_max(&self, state: &X) -> (usize, f64) {
        let mut iter = self.evaluate_all(state).into_iter().enumerate();
        let first = iter.next().unwrap();

        iter.fold(first, |acc, (i, x)| if acc.1 > x { acc } else { (i, x) })
    }
}
