use crate::{
    OnlineLearner, Shared, make_shared,
    control::Controller,
    domains::Transition,
    fa::{
        Parameterised, Weights, WeightsView, WeightsViewMut,
        StateActionFunction, EnumerableStateActionFunction,
        DifferentiableStateActionFunction,
    },
    policies::{Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
    traces::Trace,
};
use rand::Rng;

/// Watkins' Q-learning with eligibility traces.
///
/// # References
/// - Watkins, C. J. C. H. (1989). Learning from Delayed Rewards. Ph.D. thesis,
/// Cambridge University.
/// - Watkins, C. J. C. H., Dayan, P. (1992). Q-learning. Machine Learning,
/// 8:279–292.
#[derive(Parameterised, Serialize, Deserialize)]
pub struct QLambda<S, F, P, T> {
    #[weights] pub fa_theta: F,

    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
    pub lambda: f64,

    trace: T,

    prior_state: S,
}

impl<S, F, P, T> QLambda<S, Shared<F>, P, T> {
    pub fn new(
        fa_theta: F,
        policy: P,
        trace: T,
        alpha: f64,
        gamma: f64,
        lambda: f64,
        initial_state: S,
    ) -> Self {
        let fa_theta = make_shared(fa_theta);

        QLambda {
            fa_theta: fa_theta.clone(),

            policy,

            alpha,
            gamma,
            lambda,

            trace,

            prior_state: initial_state,
        }
    }
}

impl<S, F, P, T> OnlineLearner<S, P::Action> for QLambda<S, F, P, T>
where
    F: EnumerableStateActionFunction<S> + DifferentiableStateActionFunction<S, usize>,
    P: EnumerablePolicy<S>,
    T: Trace<F::Gradient>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        // let s = t.from.state();
        let qsa = self.fa_theta.evaluate(&self.prior_state, &t.action);

        // Update trace:
        self.trace.scale(if t.action == self.fa_theta.find_max(&self.prior_state).0 {
            self.lambda * self.gamma
        } else {
            0.0
        });
        self.trace.update(&self.fa_theta.grad(&self.prior_state, &t.action));

        // Update weight vectors:
        if t.terminated() {
            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * (t.reward - qsa));
            self.trace.reset();
        } else {
            let ns = t.to.state();
            let (_, nqs_max) = self.fa_theta.find_max(ns);
            let residual = t.reward + self.gamma * nqs_max - qsa;

            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * residual);
        }

        self.prior_state = t.to.owned_state();
    }
}

impl<S, F, P, T> Controller<S, P::Action> for QLambda<S, F, P, T>
where
    F: EnumerableStateActionFunction<S, Output = f64>,
    P: EnumerablePolicy<S>,
{
    fn sample_target(&self, _: &mut impl Rng, s: &S) -> P::Action {
        self.fa_theta.find_max(s).0
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, F, P, T> ValuePredictor<S> for QLambda<S, F, P, T>
where
    F: EnumerableStateActionFunction<S, Output = f64>,
    P: Policy<S>,
{
    fn predict_v(&self, s: &S) -> f64 { self.fa_theta.find_max(s).1 }
}

impl<S, F, P, T> ActionValuePredictor<S, P::Action> for QLambda<S, F, P, T>
where
    F: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn predict_q(&self, s: &S, a: &P::Action) -> f64 {
        self.fa_theta.evaluate(s, a)
    }
}
