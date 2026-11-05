//! Input-owned lexical preparation and completed-row reuse.

use super::*;

#[derive(Clone, Copy)]
struct WordId(usize);

#[derive(Clone, Copy)]
struct ReferenceClassId(usize);

struct Word<'input> {
    prepared: PreparedFuzzyWord<'input>,
    row: Row,
}

enum Row {
    Unobserved,
    Complete(EdgeSpan),
}

/// Only a fully emitted row can supply reusable reference addresses.
#[derive(Clone, Copy)]
struct EdgeSpan {
    start: usize,
    end: usize,
}

struct ReferenceClass {
    word: WordId,
    occurrences: usize,
}

/// Exact original spellings share preparation, never fuzzy equivalence classes.
/// The source borrows, policy and allowance cannot be replaced on execution.
pub(super) struct FuzzyCandidates<'input> {
    reference: &'input [String],
    words: Vec<Word<'input>>,
    payload: Vec<WordId>,
    reference_classes: Vec<ReferenceClass>,
    reference_order: Vec<ReferenceClassId>,
    policy: FuzzyComparison,
    edge_limit: usize,
}

impl<'input> FuzzyCandidates<'input> {
    pub(super) fn prepare<C: WorkControl + ?Sized>(
        payload: &'input [String],
        reference: &'input [String],
        policy: FuzzyComparison,
        edge_limit: usize,
        control: &C,
    ) -> Result<Self, C::Error> {
        let mut dictionary = BTreeMap::new();
        let mut words = Vec::new();
        let mut intern = |original: &'input str| {
            *dictionary.entry(original).or_insert_with(|| {
                let id = WordId(words.len());
                words.push(Word {
                    prepared: PreparedFuzzyWord::new(original),
                    row: Row::Unobserved,
                });
                id
            })
        };
        let mut source = Vec::with_capacity(payload.len());
        for original in payload {
            control.check()?;
            source.push(intern(original));
        }
        let mut class_by_spelling = BTreeMap::new();
        let mut reference_classes: Vec<ReferenceClass> = Vec::new();
        let mut reference_order = Vec::with_capacity(reference.len());
        for original in reference {
            control.check()?;
            let id = *class_by_spelling
                .entry(original.as_str())
                .or_insert_with(|| {
                    let id = ReferenceClassId(reference_classes.len());
                    reference_classes.push(ReferenceClass {
                        word: intern(original),
                        occurrences: 0,
                    });
                    id
                });
            reference_classes[id.0].occurrences += 1;
            reference_order.push(id);
        }
        Ok(Self {
            reference,
            words,
            payload: source,
            reference_classes,
            reference_order,
            policy,
            edge_limit,
        })
    }

    pub(super) fn finish<C: WorkControl + ?Sized>(
        mut self,
        control: &C,
    ) -> Result<Result<sparse::Candidates<'input>, CorrespondenceBudget>, C::Error> {
        let mut edges = Vec::new();
        for (payload, word) in self.payload.into_iter().enumerate() {
            control.check()?;
            let (source, count) = match self.words[word.0].row {
                Row::Complete(span) => (RowSource::Retained(span), span.end - span.start),
                Row::Unobserved => {
                    let mut matches = Vec::with_capacity(self.reference_classes.len());
                    let mut count = 0;
                    for other in &self.reference_classes {
                        control.check()?;
                        let relation = if self.words[word.0]
                            .prepared
                            .matches(&self.words[other.word.0].prepared, self.policy)
                        {
                            count += other.occurrences;
                            Relation::Matching
                        } else {
                            Relation::Distinct
                        };
                        matches.push(relation);
                    }
                    (RowSource::Observed(matches), count)
                }
            };
            let prepared = PreparedRow {
                edges: &mut edges,
                reference_order: &self.reference_order,
                source,
                payload,
                count,
                edge_limit: self.edge_limit,
            };
            let admitted = match prepared.admit() {
                Ok(admitted) => admitted,
                Err(budget) => return Ok(Err(budget)),
            };
            let complete = admitted.append(control)?;
            if let Row::Unobserved = self.words[word.0].row {
                self.words[word.0].row = Row::Complete(complete);
            }
        }
        Ok(Ok(sparse::Candidates::bind(self.reference, edges)))
    }
}

enum Relation {
    Matching,
    Distinct,
}

enum RowSource {
    Observed(Vec<Relation>),
    Retained(EdgeSpan),
}

struct PreparedRow<'row> {
    edges: &'row mut Vec<Edge>,
    reference_order: &'row [ReferenceClassId],
    source: RowSource,
    payload: usize,
    count: usize,
    edge_limit: usize,
}

/// Construction admits the entire row before any of its edges are appended.
struct AdmittedRow<'row>(PreparedRow<'row>);

impl<'row> PreparedRow<'row> {
    fn admit(self) -> Result<AdmittedRow<'row>, CorrespondenceBudget> {
        if self.count > self.edge_limit - self.edges.len() {
            return Err(CorrespondenceBudget::candidate_edges(self.edge_limit));
        }
        Ok(AdmittedRow(self))
    }
}

impl AdmittedRow<'_> {
    fn append<C: WorkControl + ?Sized>(self, control: &C) -> Result<EdgeSpan, C::Error> {
        let row = self.0;
        let start = row.edges.len();
        match row.source {
            RowSource::Observed(matches) => {
                for (reference, class) in row.reference_order.iter().enumerate() {
                    control.check()?;
                    match matches[class.0] {
                        Relation::Matching => row.edges.push(Edge {
                            payload: row.payload,
                            reference,
                        }),
                        Relation::Distinct => {}
                    }
                }
            }
            RowSource::Retained(span) => {
                for index in span.start..span.end {
                    control.check()?;
                    let reference = row.edges[index].reference;
                    row.edges.push(Edge {
                        payload: row.payload,
                        reference,
                    });
                }
            }
        }
        Ok(EdgeSpan {
            start,
            end: row.edges.len(),
        })
    }
}
