//! Concrete submitted options become the prepared morphology state once.

use crate::chat_ops::morphosyntax_ops::{MultilingualPolicy, TokenizationMode};
use crate::execution::worker_gateway::MorphotagRuntimeOptions;
use crate::options::MorphotagOptions;
use std::sync::Arc;

impl MorphotagRuntimeOptions {
    pub(crate) fn from_submitted(options: &MorphotagOptions) -> Self {
        Self {
            tokenization_mode: TokenizationMode::from(options.retokenize),
            multilingual_policy: MultilingualPolicy::from_skip_flag(options.skipmultilang),
            mwt: Arc::new(options.common.mwt.clone()),
            l2_policy: (!options.no_l2_morphotag).into(),
            pos_hint_policy: (!options.no_pos_hints).into(),
            ca_policy: options.ca_policy,
            review_level: options.review_level,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards: Submitted retokenization, multilingual skipping and authored expansion bytes reach the actual prepared morphology options.
    /// Kind: policy
    /// Not a type: Concrete options bind the command family, while the selected policy alternatives and user-authored expansion bytes require checking the values delivered by the producer.
    /// State: None. Concrete submitted options exercise the actual preparation builder without a snapshot, host, filesystem or worker.
    /// Concurrency: None. The source options and prepared result belong to this case.
    #[test]
    fn submitted_morphology_policies_reach_runtime_options() {
        let mut options = MorphotagOptions {
            retokenize: true,
            skipmultilang: true,
            ..Default::default()
        };
        options
            .common
            .mwt
            .insert("gonna".into(), vec!["going".into(), "to".into()]);
        let prepared = MorphotagRuntimeOptions::from_submitted(&options);
        assert_eq!(
            prepared.tokenization_mode,
            TokenizationMode::StanzaRetokenize
        );
        assert_eq!(
            prepared.multilingual_policy,
            MultilingualPolicy::SkipNonPrimary
        );
        assert_eq!(
            prepared.mwt.get("gonna"),
            Some(&vec!["going".to_owned(), "to".to_owned()])
        );
    }

    /// Guards: Default submitted morphology options preserve tokens and process every declared language.
    /// Kind: policy
    /// Not a type: The command proof selects the actual builder, while the default tokenization and multilingual policy choices remain semantic alternatives.
    /// State: None. The default input and prepared policy contain only pure values.
    /// Concurrency: None. No resources or mutable observations are shared.
    #[test]
    fn default_morphology_policies_reach_runtime_options() {
        let prepared = MorphotagRuntimeOptions::from_submitted(&MorphotagOptions::default());
        assert_eq!(prepared.tokenization_mode, TokenizationMode::Preserve);
        assert_eq!(prepared.multilingual_policy, MultilingualPolicy::ProcessAll);
    }
}
