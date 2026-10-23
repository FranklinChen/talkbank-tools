//! A packed bit vector with a declared length, and the one way to read it
//! past its ends.
//!
//! The packed arithmetic stores a score function's unit increments as bits
//! (`u64` words, bit `i` in word `i / 64`) and shifts them across word
//! boundaries, so it reads bits before the first and after the last. The
//! encoding DEFINES every bit outside `0..length` as zero: the zero is the
//! definition of the representation (a score function is constant outside
//! its domain), not a value standing in for a missing one.
//!
//! [`PackedBits`] owns that definition. It holds exactly `length.div_ceil(64)`
//! words and clears every stored bit at or past `length`, in every
//! constructor and after every edit, so a word it does not store and a bit
//! past its length are the same zero. Reads go through [`ZeroPadded`], a view
//! of one `PackedBits`, by [`WindowStart`]: a window that begins inside the
//! vector or before it. There is no read by raw word index, so an index slip
//! cannot read some other vector's padding as data.

/// Where a window of a packed vector begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::dp_align) enum WindowStart {
    /// At this bit of the vector: the window is the vector shifted right.
    At(usize),
    /// This many zero bits before bit 0: the window is the vector shifted
    /// left, its first bits the zero-extension before the vector.
    Before(usize),
}

/// A packed bit vector of a declared length.
///
/// Invariant, held by every constructor and edit: `words.len() ==
/// length.div_ceil(64)`, and every stored bit at or past `length` is zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::dp_align) struct PackedBits {
    words: Vec<u64>,
    length: usize,
}

/// Why words cannot be a packed vector of the declared length. Only tests
/// build a vector from raw words; production vectors are made by the
/// constructors below, which hold the invariant by construction.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(in crate::dp_align) enum PackedBitsRefusal {
    /// The word count is not the one the length needs.
    #[error("{words} words for a vector of {length} bits")]
    WordCount {
        /// Words supplied.
        words: usize,
        /// Declared length in bits.
        length: usize,
    },
    /// A bit at or past the declared length is set.
    #[error("a bit at or past the declared length {length} is set")]
    BitPastLength {
        /// Declared length in bits.
        length: usize,
    },
}

impl PackedBits {
    /// The all-zero vector of `length` bits.
    pub(in crate::dp_align) fn zeros(length: usize) -> Self {
        Self {
            words: vec![0; length.div_ceil(64)],
            length,
        }
    }

    /// `words` as a vector of `length` bits, refused unless the word count
    /// is the length's and no bit at or past it is set.
    #[cfg(test)]
    pub(in crate::dp_align) fn from_words(
        words: Vec<u64>,
        length: usize,
    ) -> Result<Self, PackedBitsRefusal> {
        if words.len() != length.div_ceil(64) {
            return Err(PackedBitsRefusal::WordCount {
                words: words.len(),
                length,
            });
        }
        let checked = Self { words, length };
        let mut cleared = checked.clone();
        cleared.clear_past_length();
        match cleared == checked {
            true => Ok(checked),
            false => Err(PackedBitsRefusal::BitPastLength { length }),
        }
    }

    /// The vector of `length` bits whose word `block` is `word(block)`, the
    /// bits past `length` cleared. Fallible only by `word`.
    pub(in crate::dp_align) fn try_from_fn<E>(
        length: usize,
        mut word: impl FnMut(usize) -> Result<u64, E>,
    ) -> Result<Self, E> {
        let mut words = Vec::with_capacity(length.div_ceil(64));
        for block in 0..length.div_ceil(64) {
            words.push(word(block)?);
        }
        let mut bits = Self { words, length };
        bits.clear_past_length();
        Ok(bits)
    }

    /// The vector of `length` bits whose word `block` is `word(block)`, the
    /// bits past `length` cleared.
    pub(in crate::dp_align) fn from_fn(length: usize, word: impl FnMut(usize) -> u64) -> Self {
        let mut bits = Self {
            words: (0..length.div_ceil(64)).map(word).collect(),
            length,
        };
        bits.clear_past_length();
        bits
    }

    /// The `span` bits of `view` from `start`: a shift, zero-extended.
    pub(in crate::dp_align) fn window(
        view: ZeroPadded<'_>,
        start: WindowStart,
        span: usize,
    ) -> Self {
        Self::from_fn(span, |block| view.window_word(start, block))
    }

    /// The declared length in bits.
    pub(in crate::dp_align) fn length(&self) -> usize {
        self.length
    }

    /// The stored words, `length.div_ceil(64)` of them.
    pub(in crate::dp_align) fn words(&self) -> &[u64] {
        &self.words
    }

    /// The view every read past an end goes through.
    pub(in crate::dp_align) fn padded(&self) -> ZeroPadded<'_> {
        ZeroPadded { words: &self.words }
    }

    /// Change the length: a longer vector gains zero bits, a shorter one
    /// loses its bits past the new length.
    pub(in crate::dp_align) fn resize(&mut self, length: usize) {
        self.words.resize(length.div_ceil(64), 0);
        self.length = length;
        self.clear_past_length();
    }

    /// Edit the words in place (`edit` receives them and the length), then
    /// clear any bit the edit left at or past the length, whether or not it
    /// succeeded, so the vector never leaves this call outside its
    /// invariant.
    pub(in crate::dp_align) fn edit<E>(
        &mut self,
        edit: impl FnOnce(&mut [u64], usize) -> Result<(), E>,
    ) -> Result<(), E> {
        let result = edit(&mut self.words, self.length);
        self.clear_past_length();
        result
    }

    fn clear_past_length(&mut self) {
        if !self.length.is_multiple_of(64)
            && let Some(last) = self.words.last_mut()
        {
            *last &= (1u64 << (self.length % 64)) - 1;
        }
    }
}

/// A read-only view of one [`PackedBits`], whose stored words end where its
/// declared length does, so a word past the storage is a word past the
/// vector: zero by definition. Only [`PackedBits::padded`] makes one.
#[derive(Debug, Clone, Copy)]
pub(in crate::dp_align) struct ZeroPadded<'a> {
    words: &'a [u64],
}

impl ZeroPadded<'_> {
    /// Word `block` of the window that begins at `start`: bit `j` of the
    /// result is bit `64 * block + j` of the window.
    pub(in crate::dp_align) fn window_word(self, start: WindowStart, block: usize) -> u64 {
        match start {
            WindowStart::At(offset) => {
                let at = offset / 64 + block;
                let shift = offset % 64;
                match shift {
                    0 => self.stored(at),
                    _ => (self.stored(at) >> shift) | (self.stored(at + 1) << (64 - shift)),
                }
            }
            WindowStart::Before(offset) => {
                // The window is the vector shifted left by `offset`: its
                // word `block` is made of the vector's words `at` and
                // `at - 1`, either of which may lie before bit 0.
                let Some(at) = block.checked_sub(offset / 64) else {
                    return 0;
                };
                let shift = offset % 64;
                let low = self.stored(at) << shift;
                match (shift, at.checked_sub(1)) {
                    (0, _) | (_, None) => low,
                    (_, Some(previous)) => low | (self.stored(previous) >> (64 - shift)),
                }
            }
        }
    }

    /// The stored word `at`, or the zero word past the storage, which is
    /// past the declared length.
    fn stored(self, at: usize) -> u64 {
        match self.words.get(at) {
            Some(&word) => word,
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bits_of(bits: &PackedBits) -> Vec<bool> {
        (0..bits.length())
            .map(|i| bits.words()[i / 64] >> (i % 64) & 1 == 1)
            .collect()
    }

    /// A vector is refused when its word count is not its length's or a bit
    /// past its length is set: the invariant every read relies on.
    #[test]
    fn a_vector_holds_exactly_its_declared_bits() {
        assert!(PackedBits::from_words(vec![0b1011], 4).is_ok());
        assert_eq!(
            PackedBits::from_words(vec![0b1011, 0], 4),
            Err(PackedBitsRefusal::WordCount {
                words: 2,
                length: 4
            })
        );
        assert_eq!(
            PackedBits::from_words(vec![0b1_0000], 4),
            Err(PackedBitsRefusal::BitPastLength { length: 4 })
        );
        let mut grown = PackedBits::from_words(vec![u64::MAX], 64).unwrap();
        grown.resize(70);
        assert_eq!(grown.words(), [u64::MAX, 0]);
        grown.resize(3);
        assert_eq!(grown.words(), [0b111]);
    }

    /// Every window, shifted either way and reaching past either end, reads
    /// the vector's bits where they are and zero everywhere else.
    #[test]
    fn a_window_is_the_zero_extended_shift() {
        for length in [0usize, 1, 7, 63, 64, 65, 130] {
            let words: Vec<u64> = (0..length.div_ceil(64))
                .map(|i| 0x9E37_79B9_7F4A_7C15u64.rotate_left(i as u32 * 7))
                .collect();
            let mut bits = PackedBits::zeros(length);
            bits.edit::<()>(|stored, _| {
                stored.copy_from_slice(&words);
                Ok(())
            })
            .unwrap();
            let truth = bits_of(&bits);
            let bit = |i: i64| -> bool {
                usize::try_from(i)
                    .ok()
                    .and_then(|i| truth.get(i))
                    .is_some_and(|&bit| bit)
            };
            for offset in [0usize, 1, 5, 63, 64, 65, 129, 200] {
                for span in [0usize, 1, 64, 65, 200] {
                    for (start, origin) in [
                        (WindowStart::At(offset), offset as i64),
                        (WindowStart::Before(offset), -(offset as i64)),
                    ] {
                        let window = PackedBits::window(bits.padded(), start, span);
                        let expected: Vec<bool> =
                            (0..span as i64).map(|j| bit(origin + j)).collect();
                        assert_eq!(bits_of(&window), expected, "{start:?} span {span}");
                    }
                }
            }
        }
    }
}
