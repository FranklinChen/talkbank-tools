//! `FormType → SynthesisRule` table for non-analyzable words.
//!
//! Editing the [`scat_synthesis`] match body is the single place to
//! change the POS prefix or form-type-derived features for any
//! `@<letter>` marker. The match is exhaustive over [`FormType`]
//! adding a variant is a compile error here.
//!
//! The scat values follow the established CHAT-MOR convention for
//! special-form markers (`neo` for neologism, `bab` for babbling, etc.),
//! preserved by `n:let|<surface>` for letters so CLAN's noun-class
//! matchers count them correctly. See the [Special Form Markers]
//! section of the CHAT manual for the marker definitions.
//!
//! [Special Form Markers]: https://talkbank.org/0info/manuals/CHAT.html#SpecialForm_Marker

use smallvec::SmallVec;
use talkbank_model::model::FormType;
use talkbank_model::model::dependent_tier::mor::{MorFeature, MorFeatureShapeError};

/// One row of the table: scat code (POS prefix as a flat string, interned
/// at use-site by [`super::synthesize`]) plus zero or more flat features.
pub(crate) struct SynthesisRule {
    pub(crate) scat: &'static str,
    pub(crate) features: SmallVec<[MorFeature; 1]>,
}

impl SynthesisRule {
    fn bare(scat: &'static str) -> Self {
        Self {
            scat,
            features: SmallVec::new(),
        }
    }

    fn with_feature(mut self, value: impl AsRef<str>) -> Result<Self, MorFeatureShapeError> {
        self.features.push(MorFeature::flat(value)?);
        Ok(self)
    }
}

/// Map a `FormType` to its synthesis rule. Exhaustive match, adding a
/// `FormType` variant without updating this is a compile error.
pub(crate) fn scat_synthesis(form_type: &FormType) -> Result<SynthesisRule, MorFeatureShapeError> {
    Ok(match form_type {
        FormType::B => SynthesisRule::bare("bab"),
        FormType::C => SynthesisRule::bare("chi"),
        FormType::D => SynthesisRule::bare("dia"),
        FormType::F => SynthesisRule::bare("fam"),
        FormType::I => SynthesisRule::bare("co"),
        FormType::K => SynthesisRule::bare("n:let"),
        FormType::L => SynthesisRule::bare("n:let"),
        FormType::N => SynthesisRule::bare("neo"),
        FormType::O => SynthesisRule::bare("on"),
        FormType::P => SynthesisRule::bare("phon"),
        FormType::Q => SynthesisRule::bare("meta"),
        FormType::SAS => SynthesisRule::bare("sas"),
        FormType::SI => SynthesisRule::bare("sing"),
        FormType::SL => SynthesisRule::bare("sign"),
        FormType::T => SynthesisRule::bare("test"),
        FormType::U => SynthesisRule::bare("uni"),
        FormType::WP => SynthesisRule::bare("wplay"),
        FormType::X => SynthesisRule::bare("unk"),

        // Markers without an established scat assignment in the
        // CHAT-MOR convention; conservative defaults, easy to revise.
        FormType::FP => SynthesisRule::bare("co"),
        FormType::G => SynthesisRule::bare("unk"),
        FormType::LS => SynthesisRule::bare("n:let").with_feature("Plur")?,

        // A marker naming no declared form. This is INVALID CHAT, not an
        // extension point: chatter rejects it with E203 ("Undeclared form
        // marker"), verified against 0.11.0 on `ba@a`. The variant exists so
        // the PARSER can preserve the raw text rather than corrupting
        // `word@zz` into `word@z:zz`, leaving the VALIDATOR to reject the file;
        // reaching here therefore means we are tagging a file that E203 has
        // already condemned.
        //
        // So this arm is damage limitation, not a synthesis rule. `unk` plus
        // the marker's own text is the only honest output: inventing a scat
        // for a form CHAT does not define would fabricate a value, and
        // dropping the code would destroy the evidence of what was wrong. The
        // real fix belongs upstream, in whatever produced the invalid marker.
        FormType::Undeclared(code) => SynthesisRule::bare("unk").with_feature(code)?,

        FormType::UserDefined(code) => SynthesisRule::bare("unk").with_feature(code)?,
    })
}
