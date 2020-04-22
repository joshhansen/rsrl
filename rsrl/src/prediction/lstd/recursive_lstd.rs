use crate::{
    OnlineLearner,
    domains::Transition,
    fa::{
        StateFunction,
        Weights, WeightsView, WeightsViewMut, Parameterised,
        linear::LinearStateFunction,
    },
    prediction::ValuePredictor,
};
use ndarray::{Array2, Axis};

#[derive(Parameterised)]
pub struct RecursiveLSTD<F, S> {
    #[weights] pub fa_theta: F,

    pub gamma: f64,

    c_mat: Array2<f64>,

    prior_state: S,
}

impl<F: Parameterised, S> RecursiveLSTD<F, S> {
    pub fn new(fa_theta: F, gamma: f64, initial_state: S) -> Self {
        let n_features = fa_theta.weights_dim()[0];

        RecursiveLSTD {
            fa_theta,

            gamma,

            c_mat: Array2::eye(n_features) * 1e-5,

            prior_state: initial_state,
        }
    }
}

impl<S, A, F> OnlineLearner<S, A> for RecursiveLSTD<F, S>
where
    F: LinearStateFunction<S, Output = f64>
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        // let (s, ns) = t.states();

        let ns = t.to.state();

        let phi_s = self.fa_theta.features(&self.prior_state);
        let theta_s = self.fa_theta.evaluate_features(&phi_s);

        if t.terminated() {
            // (D x 1)
            let phi_s = phi_s.expanded().insert_axis(Axis(1));

            // (D x 1) <- (D x D) . (D x 1)
            let v = self.c_mat.dot(&phi_s);

            // (1 x D) <- (D x 1)T
            let g = v.t();

            // (1 x 1) <- (1 x D) . (D x 1)
            let a = 1.0 + unsafe { g.dot(&phi_s).uget((0, 0)) };
            let residual = t.reward - theta_s;

            // (D x D) <- (D x 1) . (1 x D)
            let vg = v.dot(&g);

            self.c_mat.scaled_add(-1.0 / a, &vg);
            self.fa_theta.weights_view_mut().scaled_add(residual / a, &v);
        } else {
            let phi_ns = self.fa_theta.features(ns);
            let theta_ns = self.fa_theta.evaluate_features(&phi_ns);

            // (D x 1)
            let phi_s = phi_s.expanded().insert_axis(Axis(1));
            let phi_ns = phi_ns.expanded().insert_axis(Axis(1));

            let pd = (-self.gamma * phi_ns) + &phi_s;

            // (1 x D) <- ((D x D) . (D x 1))T
            //  Note: self.permuted_axes([1, 0]) is equivalent to taking the transpose.
            let g = self.c_mat.dot(&pd).permuted_axes([1, 0]);

            // (1 x 1) <- (1 x D) . (D x 1)
            let a = 1.0 + unsafe { g.dot(&phi_s).uget((0, 0)) };

            // (D x 1) <- (D x D) . (D x 1)
            let v = self.c_mat.dot(&phi_s);
            let residual = t.reward + self.gamma * theta_ns - theta_s;

            // (D x D) <- (D x 1) . (1 x D)
            let vg = v.dot(&g);

            self.c_mat.scaled_add(-1.0 / a, &vg);
            self.fa_theta.weights_view_mut().scaled_add(residual / a, &v);
        }

        self.prior_state = t.to.owned_state();
    }

    fn handle_terminal(&mut self) {
        self.c_mat.fill(0.0);
    }
}

impl<S, F: StateFunction<S>> ValuePredictor<S> for RecursiveLSTD<F, S>
where
    F: StateFunction<S, Output = f64>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
