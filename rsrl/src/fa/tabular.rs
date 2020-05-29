use crate::fa::{EnumerableStateActionFunction, StateActionFunction};
use std::ops::IndexMut;

#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
pub struct Tabular(Vec<Vec<f64>>);

impl Tabular {
    pub fn new(weights: Vec<Vec<f64>>) -> Self { Tabular(weights) }

    pub fn zeros(dim: [usize; 2]) -> Self { Tabular(vec![vec![0.0; dim[0]]; dim[1]]) }
}

// Q(s, a):
impl StateActionFunction<usize, usize> for Tabular {
    type Output = f64;

    fn evaluate(&self, state: &usize, action: &usize) -> f64 { self.0[*action][*state] }

    fn update_with_error(&mut self, state: &usize, action: &usize, _value: Self::Output, _estimate: Self::Output,
        error: Self::Output, _raw_error: Self::Output, _learning_rate: Self::Output) {

            *self.0.index_mut(*action).index_mut(*state) += error;
    }
}

impl EnumerableStateActionFunction<usize> for Tabular {
    fn n_actions(&self) -> usize { self.0.len() }

    fn evaluate_all(&self, state: &usize) -> Vec<f64> { self.0.iter().map(|c| c[*state]).collect() }

    fn update_all_with_errors(&mut self, state: &usize, _values: Vec<Self::Output>, _estimates: Vec<Self::Output>,
        errors: Vec<Self::Output>, _raw_errors: Vec<Self::Output>, _learning_rate: Self::Output) {

        for (c, e) in self.0.iter_mut().zip(errors.into_iter()) {
            *c.index_mut(*state) += e;
        }
    }
}
