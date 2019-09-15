use crate::{
    core::*,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction,
        linear::{Features, LinearStateFunction},
    },
    geometry::{Space, Matrix, MatrixView, MatrixViewMut},
    utils::{argmaxima, pinv},
};
use ndarray::{Axis, Array2, ArrayView2, ArrayViewMut2};
use ndarray_linalg::solve::Solve;

#[derive(Parameterised)]
pub struct LSTD<F> {
    #[weights] pub fa_theta: F,

    pub gamma: Parameter,

    a: Matrix<f64>,
    b: Vector<f64>,
}

impl<F: Parameterised> LSTD<F> {
    pub fn new<T: Into<Parameter>>(fa_theta: F, gamma: T) -> Self {
        let dim = fa_theta.weights_dim();

        LSTD {
            fa_theta,

            gamma: gamma.into(),

            a: Matrix::eye(dim[0]) * 1e-6,
            b: Vector::zeros(dim[0]),
        }
    }
}

impl<F: Parameterised> LSTD<F> {
    pub fn solve(&mut self) {
        let mut w = self.fa_theta.weights_view_mut();

        if let Ok(theta) = self.a.solve(&self.b) {
            // First try the clean approach:
            w.assign(&theta);
        } else if let Ok(ainv) = pinv(&self.a) {
            // Otherwise solve via SVD:
            w.assign(&ainv.dot(&self.b));
        }
    }
}

impl<F> Algorithm for LSTD<F> {
    fn handle_terminal(&mut self) {
        self.gamma = self.gamma.step();
    }
}

impl<S, A, F> BatchLearner<S, A> for LSTD<F>
where
    F: LinearStateFunction<S, Output = f64>,
{
    fn handle_batch(&mut self, ts: &[Transition<S, A>]) {
        ts.into_iter().for_each(|ref t| {
            let (s, ns) = t.states();

            let phi_s = self.fa_theta.features(s).expanded();

            self.b.scaled_add(t.reward, &phi_s);

            if t.terminated() {
                let phi_s = phi_s.insert_axis(Axis(1));

                self.a += &phi_s.view().dot(&phi_s.t());
            } else {
                let phi_ns = self.fa_theta.features(ns).expanded();
                let pd = (self.gamma.value() * phi_ns - &phi_s).insert_axis(Axis(0));

                self.a -= &phi_s.insert_axis(Axis(1)).dot(&pd);
            }
        });

        self.solve();
    }
}

impl<S, F> ValuePredictor<S> for LSTD<F>
where
    F: StateFunction<S, Output = f64>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
