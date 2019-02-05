use crate::core::*;
use crate::domains::Transition;
use crate::geometry::Matrix;
use crate::fa::Parameterised;
use crate::policies::{Policy, ParameterisedPolicy};
use std::marker::PhantomData;

pub struct REINFORCE<P> {
    pub policy: Shared<P>,

    pub alpha: Parameter,
    pub gamma: Parameter,
}

impl<P> REINFORCE<P> {
    pub fn new<T1, T2>(policy: Shared<P>, alpha: T1, gamma: T2) -> Self
    where
        T1: Into<Parameter>,
        T2: Into<Parameter>,
    {
        REINFORCE {
            policy,

            alpha: alpha.into(),
            gamma: gamma.into(),
        }
    }
}

impl<P: Algorithm> Algorithm for REINFORCE<P> {
    fn handle_terminal(&mut self) {
        self.alpha = self.alpha.step();
        self.gamma = self.gamma.step();

        self.policy.borrow_mut().handle_terminal();
    }
}

impl<S, P> BatchLearner<S, P::Action> for REINFORCE<P>
where
    P: ParameterisedPolicy<S>,
    P::Action: Clone,
{
    fn handle_batch(&mut self, batch: &[Transition<S, P::Action>]) {
        let z = batch.len() as f64;
        let mut ret = 0.0;

        for t in batch.into_iter().rev() {
            ret = t.reward + self.gamma * ret;

            self.policy.borrow_mut().update(
                t.from.state(),
                t.action.clone(),
                self.alpha * ret / z
            );
        }
    }
}

impl<S, P: ParameterisedPolicy<S>> Controller<S, P::Action> for REINFORCE<P> {
    fn sample_target(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }

    fn sample_behaviour(&mut self, s: &S) -> P::Action { self.policy.borrow_mut().sample(s) }
}

impl<P: Parameterised> Parameterised for REINFORCE<P> {
    fn weights(&self) -> Matrix<f64> {
        self.policy.weights()
    }
}
