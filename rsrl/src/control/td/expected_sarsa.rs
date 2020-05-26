use crate::{
    OnlineLearner,
    control::Controller,
    domains::Transition,
    fa::{
        Parameterised, Weights, WeightsView, WeightsViewMut,
        StateActionFunction, EnumerableStateActionFunction,
    },
    policies::{Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::Rng;

/// Action probability-weighted variant of SARSA (aka "summation Q-learning").
///
/// # References
/// - Rummery, G. A. (1995). Problem Solving with Reinforcement Learning. Ph.D
/// thesis, Cambridge University.
/// - van Seijen, H., van Hasselt, H., Whiteson, S., Wiering, M. (2009). A
/// theoretical and empirical analysis of Expected Sarsa. In Proceedings of the
/// IEEE Symposium on Adaptive Dynamic Programming and Reinforcement Learning,
/// pp. 177–184.
#[derive(Parameterised)]
pub struct ExpectedSARSA<Q, P> {
    #[weights] pub q_func: Q,
    pub policy: P,

    pub alpha: f64,
    pub gamma: f64,
}

impl<Q, P> ExpectedSARSA<Q, P> {
    pub fn new(q_func: Q, policy: P, alpha: f64, gamma: f64) -> Self {
        ExpectedSARSA {
            q_func,
            policy,

            alpha,
            gamma,
        }
    }
}

impl<S, Q, P> OnlineLearner<S, P::Action> for ExpectedSARSA<Q, P>
where
    Q: EnumerableStateActionFunction<S>,
    P: EnumerablePolicy<S>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let s = t.from.state();
        let qsa = self.predict_q(s, &t.action);

        let actual = if t.terminated() {
            t.reward
        } else {
            let ns = t.to.state();
            let exp_nv = self.predict_v(ns);

            t.reward + self.gamma * exp_nv
        };

        self.q_func.update(s, &t.action, actual, qsa, self.alpha);
    }
}

impl<S, Q, P: Policy<S>> Controller<S, P::Action> for ExpectedSARSA<Q, P> {
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, Q, P> ValuePredictor<S> for ExpectedSARSA<Q, P>
where
    Q: EnumerableStateActionFunction<S>,
    P: EnumerablePolicy<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.q_func.evaluate_all(s).into_iter()
            .zip(self.policy.probabilities(s).into_iter())
            .fold(0.0, |acc, (q, p)| acc + q * p)
    }
}

impl<S, Q, P> ActionValuePredictor<S, P::Action> for ExpectedSARSA<Q, P>
where
    Q: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn predict_q(&self, s: &S, a: &P::Action) -> f64 {
        self.q_func.evaluate(s, a)
    }
}
