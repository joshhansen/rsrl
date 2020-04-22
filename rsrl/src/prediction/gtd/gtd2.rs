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
pub struct GTD2<S, F> {
    #[weights] pub fa_theta: F,
    pub fa_w: F,

    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,

    prior_state: S,
}

impl<S, F: Parameterised> GTD2<S, F> {
    pub fn new(
        fa_theta: F,
        fa_w: F,
        alpha: f64,
        beta: f64,
        gamma: f64,
        initial_state: S,
    ) -> Self {
        if fa_theta.weights_dim() != fa_w.weights_dim() {
            panic!("fa_theta and fa_w must be equivalent function approximators.")
        }

        GTD2 {
            fa_theta,
            fa_w,

            alpha,
            beta,
            gamma,

            prior_state: initial_state,
        }
    }
}

impl<S, A, F> OnlineLearner<S, A> for GTD2<S, F>
where
    F: DifferentiableStateFunction<S, Output = f64>
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        // let (s, ns) = t.states();
        let ns = t.to.state();

        let w_s = self.fa_w.evaluate(&self.prior_state);
        let theta_s = self.fa_theta.evaluate(&self.prior_state);
        let theta_ns = self.fa_theta.evaluate(ns);

        let td_error = if t.terminated() {
            t.reward - theta_s
        } else {
            t.reward + self.gamma * theta_ns - theta_s
        };

        let grad = self.fa_theta.grad(&self.prior_state);

        self.fa_w.update_grad_scaled(&grad, self.beta * (td_error - w_s));

        let grad = grad.combine(&self.fa_theta.grad(ns), |x, y| x - self.gamma * y);

        self.fa_theta.update_grad_scaled(&grad, self.alpha * w_s);

        self.prior_state = t.to.owned_state();
    }
}

impl<S, F> ValuePredictor<S> for GTD2<S, F>
where
    F: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
