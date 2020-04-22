use crate::{
    OnlineLearner,
    domains::Transition,
    fa::{Weights, WeightsView, WeightsViewMut, Parameterised, StateFunction},
    prediction::ValuePredictor,
};

#[derive(Clone, Debug, Serialize, Deserialize, Parameterised)]
pub struct TD<S, V> {
    #[weights] pub v_func: V,

    pub alpha: f64,
    pub gamma: f64,

    prior_state: S,
}

impl<S, V> TD<S, V> {
    pub fn new(v_func: V, alpha: f64, gamma: f64, initial_state: S) -> Self {
        TD {
            v_func,

            alpha,
            gamma,

            prior_state: initial_state,
        }
    }
}

impl<S, A, V> OnlineLearner<S, A> for TD<S, V>
where
    V: StateFunction<S, Output = f64>
{
    fn handle_transition(&mut self, t: &Transition<S, A>) {
        // let s = t.from.state();
        let v = self.v_func.evaluate(&self.prior_state);

        let td_error = if t.terminated() {
            t.reward - v
        } else {
            t.reward + self.gamma * self.v_func.evaluate(t.to.state()) - v
        };

        self.v_func.update(&self.prior_state, self.alpha * td_error);

        self.prior_state = t.to.owned_state();
    }
}

impl<S, V> ValuePredictor<S> for TD<S, V>
where
    V: StateFunction<S, Output = f64>
{
    fn predict_v(&self, s: &S) -> f64 { self.v_func.evaluate(s) }
}
