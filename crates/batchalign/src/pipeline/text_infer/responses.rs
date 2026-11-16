//! One admission binds actual collected items to the external response population.
use super::*;

type ItemResults<Response, Failure> = Vec<Result<Response, ItemFailure<Failure>>>;

pub(super) struct MatchedPopulation<Item, Response, Failure> {
    items: Vec<(usize, Item)>,
    responses: ItemResults<Response, Failure>,
}
impl<Item, Response, Failure> MatchedPopulation<Item, Response, Failure> {
    pub(super) fn admit(
        items: Vec<(usize, Item)>,
        responses: ItemResults<Response, Failure>,
    ) -> Result<Self, ServerError> {
        if items.len() != responses.len() {
            return Err(ServerError::Worker(
                crate::worker::error::WorkerError::Protocol(format!(
                    "text response population has {} results for {} collected items",
                    responses.len(),
                    items.len()
                )),
            ));
        }
        Ok(Self { items, responses })
    }
    pub(super) fn into_parts(self) -> (Vec<(usize, Item)>, ItemResults<Response, Failure>) {
        (self.items, self.responses)
    }
}
