pub(crate) const ENG_SIMPLE: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\thello world .
@End
";

pub(crate) const ENG_MULTI_UTT: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tthe dog is running .
*PAR:\tI like cats .
*PAR:\tshe went to the store .
@End
";

pub(crate) const SPA_SIMPLE: &str = "\
@UTF8
@Begin
@Languages:\tspa
@Participants:\tPAR Participant
@ID:\tspa|test|PAR|||||Participant|||
*PAR:\tel gato es grande .
@End
";

// Italian single-word utterances, carrying BOTH sides of the multi-word-token
// decision. Getting one side right at the cost of the other is the failure this
// fixture exists to catch.
//
// The nouns are forms stanza's MWT destroyed: each was split into a nonexistent
// verb plus a clitic (attenzione -> attenzi + ne, cavallo -> cava + lo,
// gallina -> galli + na, bello -> ib + lo), corrupting %mor with invented verbs.
//
// The imperatives are genuine verb+enclitic multi-word tokens that MUST keep
// splitting. They are an OPEN class, so no surface pattern separates `giralo`
// (turn it) from `cavallo` (horse): both are a verb-shaped base plus a real
// clitic. A closed-class allowlist fix passes the noun assertions and silently
// destroys every one of these.
//
// `eccolo` is the third case: `ecco`+clitic, a genuinely CLOSED class, and the
// form on which a naive part-of-speech probe fails (stanza tags the unsplit
// `eccolo` as ADJ, not VERB).
//
// Measured against stanza 1.13.0 on 2026-07-28; see
// `scripts/analysis/italian_mwt_damage/` in the operator workspace.
pub(crate) const ITA_SINGLE_WORD_UTTERANCES: &str = "\
@UTF8
@Begin
@Languages:\tita
@Participants:\tCHI Target_Child, MOT Mother
@ID:\tita|test|CHI|||||Target_Child|||
@ID:\tita|test|MOT|||||Mother|||
*CHI:\tattenzione .
*CHI:\tmacchine .
*CHI:\tgallina .
*CHI:\tcavallo .
*CHI:\tmucche .
*CHI:\tpersone .
*CHI:\tbello .
*MOT:\tdammelo .
*MOT:\tdiglielo .
*MOT:\tgiralo .
*MOT:\tprendilo .
*MOT:\tguardalo .
*MOT:\teccolo .
@End
";

// Italian multi-word utterances covering the genuine multi-word tokens that
// MUST keep expanding (alla -> a + il, della -> di + il, dai -> da + il in its
// contraction reading) alongside the forms that must NOT be split.
//
// The second group is limitation 3: over-splitting that survives IN CONTEXT,
// where the single-word gate cannot reach it. Every one of these was measured
// against stanza 1.13.0 on 2026-07-28 in exactly the context written here, so
// each line genuinely triggers the defect rather than merely looking like it
// might:
//
//     la stazione e molto grande .        -> la          = il + i    (DET/DET)
//     secondo la mia opinione hai ragione . -> hai       = ha + i    (VERB/DET)
//     questa e la mozzarella .            -> mozzarella  = mozzar+la (VERB/PRON)
//     mangiamo le tagliatelle stasera .   -> tagliatelle = tagliate+le
//     prendi il pennarello rosso .        -> pennarello  = pennar+lo
//
// Note the contexts are load-bearing: `hai ragione .` alone does NOT trigger,
// which is why the longer sentence is used. Italian is pro-drop, so subjectless
// 2sg is the normal spoken form and the case most likely to be mis-analyzed.
//
// Anchors must stay unique: find_mor_line_for matches the first `*` line
// containing the substring, so `stazione` appears in two lines and neither is
// anchored on it.
pub(crate) const ITA_MULTI_WORD_UTTERANCES: &str = "\
@UTF8
@Begin
@Languages:\tita
@Participants:\tMOT Mother
@ID:\tita|test|MOT|||||Mother|||
*MOT:\tvado alla stazione della citta .
*MOT:\tsecondo la mia opinione hai ragione .
*MOT:\tdai il libro a me .
*MOT:\tvieni dai bambini .
*MOT:\tguarda le macchine .
*MOT:\tci sono molte persone qui .
*MOT:\tla stazione e molto grande .
*MOT:\tquesta e la mozzarella .
*MOT:\tmangiamo le tagliatelle stasera .
*MOT:\tprendi il pennarello rosso .
*MOT:\taprilo adesso .
*MOT:\tleggila piano .
*MOT:\tdimmi la verita .
*MOT:\tbuttalo via subito .
@End
";

pub(crate) const COMPARE_MAIN: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tthe big dog is running .
*PAR:\tI like cats .
@End
";

pub(crate) const COMPARE_GOLD: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tthe dog is running quickly .
*PAR:\tI like cats .
@End
";

pub(crate) const ENG_COREF: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tCHI Target_Child
@ID:\teng|test|CHI||female|||Target_Child|||
*CHI:\tthe dog ran .
*CHI:\tit was fast .
*CHI:\tthe cat slept .
@End
";

pub(crate) const ENG_RETOKENIZE: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tgonna eat cookies .
@End
";

pub(crate) const SPA_MULTI_UTT: &str = "\
@UTF8
@Begin
@Languages:\tspa
@Participants:\tPAR Participant
@ID:\tspa|test|PAR|||||Participant|||
*PAR:\tel perro corre .
*PAR:\tme gustan los gatos .
@End
";

pub(crate) const ENG_MULTI_SPEAKER_PARITY: &str =
    include_str!("../../../../../batchalign/tests/support/parity/eng_multi_speaker.cha");

/// The one valid utterance whose subject/copula analysis this control checks.
pub(crate) const ENG_DISFLUENCY: &str = "\
@UTF8
@Begin
@Languages:\teng
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tmm-hmm that's right .
@End
";

pub(crate) const ENG_SPA_L2: &str = "\
@UTF8
@Begin
@Languages:\teng, spa
@Participants:\tPAR Participant
@ID:\teng|test|PAR|||||Participant|||
*PAR:\tI went to the tienda@s:spa yesterday .
*PAR:\tshe was muy@s:spa nice .
*PAR:\twe talked about los@s:spa niños@s:spa .
*PAR:\tso I said hello back .
@End
";

pub(crate) const DEU_ENG_L2: &str = "\
@UTF8
@Begin
@Languages:\tdeu, eng
@Participants:\tEVA Participant
@ID:\tdeu|test|EVA|||||Participant|||
*EVA:\tich möchte film@s studies@s machen .
*EVA:\twir haben eine drug@s factory@s oben .
@End
";

pub(crate) const DEU_ENG_CONTRACTIONS: &str = "\
@UTF8
@Begin
@Languages:\tdeu, eng
@Participants:\tPAR Participant
@ID:\tdeu|test|PAR|||||Participant|||
*PAR:\tich glaube it's@s:eng working@s:eng und don't@s:eng stop@s:eng .
@End
";

pub(crate) const DEU_ENG_PHRASAL: &str = "\
@UTF8
@Begin
@Languages:\tdeu, eng
@Participants:\tPAR Participant
@ID:\tdeu|test|PAR|||||Participant|||
*PAR:\tich möchte wake@s up@s jetzt .
*PAR:\tdie kinder give@s up@s immer .
*PAR:\tsie pick@s up@s das buch .
*PAR:\tdie zeit ist time@s out@s .
@End
";

pub(crate) const CAT_SPA_L2: &str = "\
@UTF8
@Begin
@Languages:\tcat, spa
@Participants:\tMOT Mother
@ID:\tcat|test|MOT|||||Mother|||
*MOT:\tavui anem al cole@s per jugar .
*MOT:\tla nina és molt bonita@s .
@End
";

pub(crate) const DAN_ENG_L2: &str = "\
@UTF8
@Begin
@Languages:\tdan, eng
@Participants:\tPAR Participant
@ID:\tdan|test|PAR|||||Participant|||
*PAR:\tjeg kan godt lide hendes computer@s game@s .
*PAR:\thun er meget happy@s today@s .
@End
";

pub(crate) const FRA_NLD_L2: &str = "\
@UTF8
@Begin
@Languages:\tfra, nld
@Participants:\tCHI Target_Child
@ID:\tfra|test|CHI|||||Target_Child|||
*CHI:\tvoici opa@s et oma@s .
*CHI:\tje dis ja@s:nld maintenant .
@End
";
