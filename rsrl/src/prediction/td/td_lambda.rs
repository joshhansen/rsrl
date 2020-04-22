use crate::{
    OnlineLearner,
    domains::Transition,
    fa::{
        Weights, WeightsView, WeightsViewMut, Parameterised,
        StateFunction, DifferentiableStateFunction,
    },
    prediction::ValuePredictor,
    traces::Trace,
};

#[derive(Parameterised)]
pub struct TDLambda<F, S, T> {
    #[weights] pub fa_theta: F,

    pub alpha: f64,
    pub gamma: f64,
    pub lambda: f64,

    trace: T,

    prior_state: S,
}

impl<F, S, T> TDLambda<F, S, T> {
    pub fn new(
        fa_theta: F,
        trace: T,
        alpha: f64,
        gamma: f64,
        lambda: f64,
        initial_state: S,
    ) -> Self {
        TDLambda {
            fa_theta,

            alpha,
            gamma,
            lambda,

            trace,

            prior_state: initial_state,
        }
    }
}

impl<S, A, F, T> OnlineLearner<S, A> for TDLambda<F, S, T>
where
    F: DifferentiableStateFunction<S, Output = f64>,
    T: Trace<F::Gradient>,
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        // let s = t.from.state();
        let v = self.fa_theta.evaluate(&self.prior_state);

        self.trace.scaled_update(self.lambda * self.gamma, &self.fa_theta.grad(&self.prior_state));

        if t.terminated() {
            self.fa_theta.update_grad_scaled(self.trace.deref(), t.reward - v);
            self.trace.reset();
        } else {
            let td_error = t.reward + self.gamma * self.fa_theta.evaluate(t.to.state()) - v;

            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * td_error);
        };

        self.prior_state = *t.to.state().clone();
    }

    fn handle_terminal(&mut self) {
        self.trace.reset();
    }
}

impl<S, F, T> ValuePredictor<S> for TDLambda<F, S, T>
where
    F: StateFunction<S, Output = f64>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s)
    }
}
