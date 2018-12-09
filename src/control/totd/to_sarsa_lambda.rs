use core::*;
use domains::Transition;
use fa::{Approximator, Parameterised, MultiLFA, Projection, Projector, QFunction};
use policies::{Policy, FinitePolicy};

/// True online variant of the SARSA(lambda) algorithm.
///
/// # References
/// - [Van Seijen, H., Mahmood, A. R., Pilarski, P. M., Machado, M. C., &
/// Sutton, R. S. (2016). True online temporal-difference learning. Journal of
/// Machine Learning Research, 17(145), 1-40.](https://arxiv.org/pdf/1512.04087.pdf)
pub struct TOSARSALambda<S, M: Projector<S>, P> {
    pub q_func: Shared<MultiLFA<S, M>>,
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,

    trace: Trace,
    q_old: f64,
}

impl<S, M, P> TOSARSALambda<S, M, P>
where
    M: Projector<S>,
{
    pub fn new<T1, T2>(
        trace: Trace,
        q_func: Shared<MultiLFA<S, M>>,
        policy: Shared<P>,
        alpha: T1,
        gamma: T2,
    ) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        TOSARSALambda {
            q_func,
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),

            trace,
            q_old: 0.0,
        }
    }

    #[inline(always)]
    fn update_traces(&mut self, phi: Vector<f64>, decay_rate: f64) {
        let trace_update = (
            1.0 -
            self.alpha.value() * decay_rate * self.trace.get().dot(&phi)
        ) * phi;

        self.trace.decay(decay_rate);
        self.trace.update(&trace_update);
    }
}

impl<S, M: Projector<S>, P> Algorithm for TOSARSALambda<S, M, P> {
    fn step_hyperparams(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();
    }
}

impl<S, M, P> OnlineLearner<S, P::Action> for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S, Action = usize>,
{
    fn handle_transition(&mut self, t: &Transition<S, P::Action>) {
        let (s, ns) = (t.from.state(), t.to.state());

        let phi_s = self.q_func.borrow().projector.project(s);

        // Update traces:
        let n_bases = self.q_func.borrow().projector.dim();
        let decay_rate = self.trace.lambda.value() * self.gamma.value();

        self.update_traces(phi_s.clone().expanded(n_bases), decay_rate);

        // Update weight vectors:
        let z = self.trace.get();
        let qsa = self.q_func.borrow().evaluate_action_phi(&phi_s, t.action);
        let q_old = self.q_old;

        let residual = if t.terminated() {
            self.q_old = 0.0;
            self.trace.decay(0.0);

            t.reward - q_old

        } else {
            let na = self.sample_behaviour(ns);
            let nqsna = self.q_func.borrow().evaluate_action(ns, na);

            self.q_old = nqsna;

            t.reward + self.gamma * nqsna - q_old
        };

        self.q_func.borrow_mut().update_action_phi(
            &Projection::Dense(z), t.action,
            self.alpha * residual,
        );

        self.q_func.borrow_mut().update_action_phi(
            &phi_s, t.action,
            self.alpha * (q_old - qsa),
        );
    }
}

impl<S, M, P> Controller<S, P::Action> for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: Policy<S>,
{
    fn sample_target(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }

    fn sample_behaviour(&mut self, s: &S) -> P::Action {
        self.policy.borrow_mut().sample(s)
    }
}

impl<S, M, P> ValuePredictor<S> for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: FinitePolicy<S>,
{
    fn predict_v(&mut self, s: &S) -> f64 {
        self.predict_qs(s).dot(&self.policy.borrow_mut().probabilities(s))
    }
}

impl<S, M, P> ActionValuePredictor<S, P::Action> for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
    P: FinitePolicy<S>,
{
    fn predict_qs(&mut self, s: &S) -> Vector<f64> {
        self.q_func.borrow().evaluate(s).unwrap()
    }

    fn predict_qsa(&mut self, s: &S, a: P::Action) -> f64 {
        self.q_func.borrow().evaluate_action(&s, a)
    }
}

impl<S, M, P> Parameterised for TOSARSALambda<S, M, P>
where
    M: Projector<S>,
{
    fn weights(&self) -> Matrix<f64> {
        self.q_func.borrow().weights()
    }
}
