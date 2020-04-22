use crate::{
    BatchLearner,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction,
        linear::LinearStateFunction,
    },
    prediction::ValuePredictor,
    utils::pinv,
};
use ndarray::{Array1, Array2, Axis};
use ndarray_linalg::Solve;

#[derive(Parameterised)]
pub struct LSTD<S, F> {
    #[weights] pub fa_theta: F,

    pub gamma: f64,

    a: Array2<f64>,
    b: Array1<f64>,

    prior_state: S,
}

impl<S, F: Parameterised> LSTD<S, F> {
    pub fn new(fa_theta: F, gamma: f64, initial_state: S) -> Self {
        let dim = fa_theta.weights_dim();

        LSTD {
            fa_theta,

            gamma,

            a: Array2::eye(dim[0]) * 1e-6,
            b: Array1::zeros(dim[0]),

            prior_state: initial_state,
        }
    }
}

impl<S, F: Parameterised> LSTD<S, F> {
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

impl<S, A, F> BatchLearner<S, A> for LSTD<S, F>
where
    F: LinearStateFunction<S, Output = f64>,
{
    fn handle_batch(&mut self, ts: &[Transition<S, A>]) {
        ts.into_iter().for_each(|ref t| {
            // let (s, ns) = t.states();
            let ns = t.to.state();

            let phi_s = self.fa_theta.features(&self.prior_state).expanded();

            self.b.scaled_add(t.reward, &phi_s);

            if t.terminated() {
                let phi_s = phi_s.insert_axis(Axis(1));

                self.a += &phi_s.view().dot(&phi_s.t());
            } else {
                let phi_ns = self.fa_theta.features(ns).expanded();
                let pd = (self.gamma * phi_ns - &phi_s).insert_axis(Axis(0));

                self.a -= &phi_s.insert_axis(Axis(1)).dot(&pd);
            }

            self.prior_state = t.to.owned_state();
        });

        self.solve();
    }
}

impl<S, F> ValuePredictor<S> for LSTD<S, F>
where
    F: StateFunction<S, Output = f64>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
