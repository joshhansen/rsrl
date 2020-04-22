use crate::{
    OnlineLearner,
    control::Controller,
    domains::Transition,
    fa::{
        Parameterised, Weights, WeightsView, WeightsViewMut,
        StateActionFunction,
        EnumerableStateActionFunction,
        DifferentiableStateActionFunction,
    },
    policies::{Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
    traces::Trace,
};
use rand::{thread_rng, Rng};

/// On-policy variant of Watkins' Q-learning with eligibility traces (aka
/// "modified Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - Singh, S. P., Sutton, R. S. (1996). Reinforcement learning with replacing
/// eligibility traces. Machine Learning 22:123–158.
#[derive(Parameterised, Serialize, Deserialize)]
pub struct SARSALambda<S, F, P, T> {
    #[weights] pub fa_theta: F,
    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
    pub lambda: f64,

    trace: T,

    prior_state: S,
}

impl<S, F, P, T> SARSALambda<S, F, P, T> {
    pub fn new(
        fa_theta: F,
        policy: P,
        trace: T,
        alpha: f64,
        gamma: f64,
        lambda: f64,
        initial_state: S,
    ) -> Self {
        SARSALambda {
            fa_theta,
            policy,

            alpha,
            gamma,
            lambda,

            trace,

            prior_state: initial_state,
        }
    }
}

impl<S, Q, P, T> OnlineLearner<S, P::Action> for SARSALambda<S, Q, P, T>
where
    Q: DifferentiableStateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
    T: Trace<Q::Gradient>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        // let s = t.from.state();
        let qsa = self.fa_theta.evaluate(&self.prior_state, &t.action);

        // Update trace with latest feature vector:
        self.trace.scale(self.lambda * self.gamma);
        self.trace.update(&self.fa_theta.grad(&self.prior_state, &t.action));

        // Update weight vectors:
        if t.terminated() {
            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * (t.reward - qsa));
            self.trace.reset();
        } else {
            let ns = t.to.state();
            let na = self.policy.sample(&mut thread_rng(), ns);
            let nqsna = self.fa_theta.evaluate(ns, &na);
            let residual = t.reward + self.gamma * nqsna - qsa;

            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * residual);
        };

        self.prior_state = t.to.owned_state();
    }
}

impl<S, F, P: Policy<S>, T> Controller<S, P::Action> for SARSALambda<S, F, P, T> {
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, F, P, T> ValuePredictor<S> for SARSALambda<S, F, P, T>
where
    F: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.fa_theta.evaluate(s, &self.sample_behaviour(&mut thread_rng(), s))
    }
}

impl<S, F, P, T> ActionValuePredictor<S, P::Action> for SARSALambda<S, F, P, T>
where
    F: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn predict_q(&self, s: &S, a: &P::Action) -> f64 {
        self.fa_theta.evaluate(s, a)
    }
}
