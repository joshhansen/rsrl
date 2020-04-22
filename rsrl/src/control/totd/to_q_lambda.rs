use crate::{
    OnlineLearner, Shared, make_shared,
    control::Controller,
    domains::Transition,
    fa::{
        Parameterised, StateActionFunction, EnumerableStateActionFunction,
        linear::{
            LinearStateActionFunction,
            Weights, WeightsView, WeightsViewMut,
            dot_features,
        },
    },
    linalg::MatrixLike,
    policies::{Greedy, Policy, EnumerablePolicy},
    prediction::{ValuePredictor, ActionValuePredictor},
    traces::Trace,
};
use rand::{thread_rng, Rng};

/// True online variant of the Q(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
#[derive(Parameterised)]
pub struct TOQLambda<S, F, P, T> {
    #[weights] pub fa_theta: F,

    pub policy: P,
    pub target: Greedy<F>,

    pub alpha: f64,
    pub gamma: f64,
    pub lambda: f64,

    trace: T,
    q_old: f64,

    prior_state: S,
}

impl<S, F, P, T> TOQLambda<S, Shared<F>, P, T> {
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

        TOQLambda {
            fa_theta: fa_theta.clone(),

            policy,
            target: Greedy::new(fa_theta),

            alpha: alpha.into(),
            gamma: gamma.into(),
            lambda: lambda.into(),

            trace,
            q_old: 0.0,

            prior_state: initial_state,
        }
    }
}

impl<S, F, P, T> OnlineLearner<S, P::Action> for TOQLambda<S, F, P, T>
where
    F: EnumerableStateActionFunction<S> + LinearStateActionFunction<S, usize>,
    P: EnumerablePolicy<S>,
    T: Trace<F::Gradient>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        // let s = t.from.state();
        let qsa = self.fa_theta.evaluate(&self.prior_state, &t.action);

        // Update trace:
        let grad_sa = self.fa_theta.grad(&self.prior_state, &t.action);
        let phi_sa = grad_sa.features(&t.action).unwrap();

        if t.action == self.fa_theta.find_max(&self.prior_state).0 {
            let a = self.alpha;
            let c = self.lambda * self.gamma;
            let dotted = if let Some(trace_f) = self.trace.deref().features(&t.action) {
                dot_features(phi_sa, trace_f)
            } else { 0.0 };

            self.trace.combine_inplace(&grad_sa, move |x, y| {
                c * x + (1.0 - a * c * dotted) * y
            });
        } else {
            self.trace.combine_inplace(&grad_sa, |_, y| y);
        }

        // Update weight vectors:
        if t.terminated() {
            self.fa_theta.update_grad_scaled(
                self.trace.deref(),
                self.alpha * (t.reward - self.q_old)
            );
            self.fa_theta.update_grad_scaled(&grad_sa, self.alpha * (self.q_old - qsa));

            self.q_old = 0.0;
            self.trace.reset();
        } else {
            let mut rng = thread_rng();

            let ns = t.to.state();
            let na = self.sample_target(&mut rng, ns);
            let nqsna = self.fa_theta.evaluate(ns, &na);
            let residual = t.reward + self.gamma * nqsna - self.q_old;

            self.fa_theta.update_grad_scaled(self.trace.deref(), self.alpha * residual);
            self.fa_theta.update_grad_scaled(&grad_sa, self.alpha * (self.q_old - qsa));

            self.q_old = nqsna;
            if t.action != self.sample_target(&mut rng, &self.prior_state) {
                self.trace.reset();
            }
        }

        self.prior_state = t.to.owned_state();
    }

    fn handle_terminal(&mut self) {
        self.q_old = 0.0;
    }
}

impl<S, F, P, T> Controller<S, P::Action> for TOQLambda<S, F, P, T>
where
    F: EnumerableStateActionFunction<S>,
    P: EnumerablePolicy<S>,
{
    fn sample_target(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.target.sample(rng, s)
    }

    fn sample_behaviour(&self, rng: &mut impl Rng, s: &S) -> P::Action {
        self.policy.sample(rng, s)
    }
}

impl<S, F, P, T> ValuePredictor<S> for TOQLambda<S, F, P, T>
where
    F: EnumerableStateActionFunction<S, Output = f64>,
    P: EnumerablePolicy<S>,
{
    fn predict_v(&self, s: &S) -> f64 { self.predict_q(s, &self.target.mpa(s)) }
}

impl<S, F, P, T> ActionValuePredictor<S, P::Action> for TOQLambda<S, F, P, T>
where
    F: StateActionFunction<S, P::Action, Output = f64>,
    P: Policy<S>,
{
    fn predict_q(&self, s: &S, a: &P::Action) -> f64 {
        self.fa_theta.evaluate(s, a)
    }
}
