use crate::{
    OnlineLearner,
    control::Controller,
    domains::Transition,
    policies::{Policy, DifferentiablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
};
use rand::Rng;

/// Continuous Actor-Critic Learning Automaton
pub struct CACLA<S, C, PT, PB> {
    pub critic: C,

    pub target_policy: PT,
    pub behaviour_policy: PB,

    pub alpha: f64,
    pub gamma: f64,

    prior_state: S,
}

impl<S, C, PT, PB> CACLA<S, C, PT, PB> {
    pub fn new(
        critic: C,
        target_policy: PT,
        behaviour_policy: PB,
        alpha: f64,
        gamma: f64,
        initial_state: S,
    ) -> Self {
        CACLA {
            critic,

            target_policy,
            behaviour_policy,

            alpha,
            gamma,
            
            prior_state: initial_state,
        }
    }
}

impl<S, C, PT, PB> OnlineLearner<S, PT::Action> for CACLA<S, C, PT, PB>
where
    C: OnlineLearner<S, PT::Action> + ValuePredictor<S>,
    PT: DifferentiablePolicy<S, Action = f64>,
{
    fn handle_transition(&mut self, t: &Transition<S, PT::Action>) {
        // let s = t.from.state();
        // let s = &self.prior_state;
        let v = self.critic.predict_v(&self.prior_state);
        let target = if t.terminated() {
            t.reward
        } else {
            t.reward + self.gamma * self.critic.predict_v(t.to.state())
        };

        self.critic.handle_transition(t);

        if target > v {
            let mpa = self.target_policy.mpa(&self.prior_state);

            self.target_policy.update(&self.prior_state, &t.action, self.alpha * (t.action - mpa));
        }

        self.prior_state = t.to.owned_state();
    }

    fn handle_terminal(&mut self) {
        self.critic.handle_terminal();
    }
}

impl<S, C, PT, PB> ValuePredictor<S> for CACLA<S, C, PT, PB>
where
    C: ValuePredictor<S>,
{
    fn predict_v(&self, s: &S) -> f64 {
        self.critic.predict_v(s)
    }
}

impl<S, C, PT, PB> ActionValuePredictor<S, PT::Action> for CACLA<S, C, PT, PB>
where
    C: ActionValuePredictor<S, PT::Action>,
    PT: Policy<S>,
{
    fn predict_q(&self, s: &S, a: &PT::Action) -> f64 {
        self.critic.predict_q(s, a)
    }
}

impl<S, C, PT, PB> Controller<S, PT::Action> for CACLA<S, C, PT, PB>
where
    PT: DifferentiablePolicy<S>,
    PB: Policy<S, Action = PT::Action>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> PT::Action {
        self.target_policy.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> PB::Action {
        self.behaviour_policy.sample(rng, s)
    }
}
