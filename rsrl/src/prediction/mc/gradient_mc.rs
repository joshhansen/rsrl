use crate::{
    BatchLearner,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised, StateFunction},
    prediction::ValuePredictor,
};

#[derive(Parameterised)]
pub struct GradientMC<S, V> {
    #[weights] pub v_func: V,

    pub alpha: f64,
    pub gamma: f64,

    prior_state: S,
}

impl<S, V> GradientMC<S, V> {
    pub fn new(v_func: V, alpha: f64, gamma: f64, initial_state: S) -> Self {
        GradientMC {
            v_func,

            alpha,
            gamma,

            prior_state: initial_state,
        }
    }
}

impl<S, A, V> BatchLearner<S, A> for GradientMC<S, V>
where
    V: StateFunction<S, Output = f64>
{
    fn handle_batch(&mut self, batch: &[Transition<S, A>]) {
        let mut sum = 0.0;

        batch.into_iter().rev().for_each(|ref t| {
            sum = t.reward + self.gamma * sum;

            // let s = t.from.state();
            let v = self.v_func.evaluate(&self.prior_state);

            self.v_func.update(&self.prior_state, sum - v);

            self.prior_state = t.to.owned_state();
        })
    }
}

impl<S, V> ValuePredictor<S> for GradientMC<S, V>
where
    V: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 {
        self.v_func.evaluate(s)
    }
}
