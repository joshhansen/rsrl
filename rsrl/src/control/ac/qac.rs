use crate::{
    OnlineLearner,
    control::Controller,
    domains::Transition,
    policies::{Policy, DifferentiablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::Rng;

/// Action-value actor-critic.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QAC<S, C, P> {
    pub critic: C,
    pub policy: P,

    pub alpha: f64,

    prior_state: S,
}

impl<S, C, P> QAC<S, C, P> {
    pub fn new(critic: C, policy: P, alpha: f64, initial_state: S) -> Self {
        QAC {
            critic,
            policy,

            alpha,

            prior_state: initial_state,
        }
    }
}

impl<S, C, P> QAC<S, C, P> {
    pub fn update_policy(&mut self, t: &Transition<S, P::Action>)
    where
        C: OnlineLearner<S, P::Action> + ActionValuePredictor<S, P::Action>,
        P: DifferentiablePolicy<S>,
        P::Action: Clone,
    {
        // let s = t.from.state();
        let qsa = self.critic.predict_q(&self.prior_state, &t.action);

        self.policy.update(&self.prior_state, &t.action, self.alpha * qsa);
    }
}

impl<S, C, P> OnlineLearner<S, P::Action> for QAC<S, C, P>
where
    C: OnlineLearner<S, P::Action> + ActionValuePredictor<S, P::Action>,
    P: DifferentiablePolicy<S>,
    P::Action: Clone,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        self.critic.handle_transition(t);

        self.update_policy(t);

        self.prior_state = t.to.owned_state();
    }

    fn handle_terminal(&mut self) {
        self.critic.handle_terminal();
    }
}

impl<S, C, P> ValuePredictor<S> for QAC<S, C, P>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.critic.predict_v(s)
    }
}

impl<S, C, P> ActionValuePredictor<S, P::Action> for QAC<S, C, P>
where
    C: ActionValuePredictor<S, P::Action>,
    P: Policy<S>,
{
    fn predict_q(&self, s: &S, a: &P::Action) -> f64 {
        self.critic.predict_q(s, a)
    }
}

impl<S, C, P> Controller<S, P::Action> for QAC<S, C, P>
where
    P: Policy<S>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}
