use crate::{
    OnlineLearner,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
    },
    linalg::MatrixLike,
    prediction::ValuePredictor,
};

#[derive(Parameterised)]
pub struct TDC<F> {
    #[weights] pub fa_theta: F,
    pub fa_w: F,

    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl<F: Parameterised> TDC<F> {
    pub fn new(
        fa_theta: F,
        fa_w: F,
        alpha: f64,
        beta: f64,
        gamma: f64,
    ) -> Self {
        if fa_theta.weights_dim() != fa_w.weights_dim() {
            panic!("fa_theta and fa_w must be equivalent function approximators.")
        }

        TDC {
            fa_theta,
            fa_w,

            alpha,
            beta,
            gamma,
        }
    }
}

impl<S, A, F> OnlineLearner<S, A> for TDC<F>
where
    F: DifferentiableStateFunction<S, Output = f64>,
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        let (s, ns) = t.states();

        let w_s = self.fa_w.evaluate(s);
        let theta_s = self.fa_theta.evaluate(s);

        let td_error = if t.terminated() {
            t.reward - theta_s
        } else {
            t.reward + self.gamma * self.fa_theta.evaluate(ns) - theta_s
        };

        self.fa_w.update(s, self.beta * (td_error - w_s));

        let grad = self.fa_theta
            .grad(s).combine(&self.fa_theta.grad(ns), |x, y| td_error * x - w_s * y);

        self.fa_theta.update_grad_scaled(&grad, self.alpha);
    }
}

impl<S, F> ValuePredictor<S> for TDC<F>
where
    F: StateFunction<S, Output = f64>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
