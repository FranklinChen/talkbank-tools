//! Submitter admission and its database representation.

/// Who submitted a job: the client's address as the server saw it, and the
/// name that address resolved to, when one was found.
///
/// Neither text is ever empty. The fields are private, so the constructors
/// below are the only routes in: [`Self::client`] where an HTTP submission is
/// born, [`Self::direct_cli`] for the in-process CLI, and
/// [`Self::from_columns`] at the database boundary, which is the one place
/// the empty string still means "absent" (the columns are
/// `TEXT NOT NULL DEFAULT ''`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Submitter {
    /// The client's address. Conflict detection keys on it.
    address: String,
    /// The name the address resolved to, for display.
    name: Option<String>,
}

impl Submitter {
    /// The address the in-process direct CLI is recorded under.
    const DIRECT_CLI_ADDRESS: std::net::Ipv4Addr = std::net::Ipv4Addr::LOCALHOST;
    /// The name the in-process direct CLI is recorded under.
    const DIRECT_CLI_NAME: &'static str = "direct-cli";
    /// How the database columns spell an absent submitter or name.
    const ABSENT_COLUMN: &'static str = "";

    /// An HTTP client, by its peer address and the name that address resolved
    /// to. A resolution that named nothing (a Tailscale peer reporting an
    /// empty host name) records no name rather than an empty one.
    #[must_use]
    pub fn client(address: std::net::IpAddr, resolved_name: String) -> Self {
        Self {
            address: address.to_string(),
            name: (!resolved_name.is_empty()).then_some(resolved_name),
        }
    }

    /// The in-process direct CLI, which has no network peer. It is recorded
    /// as a loopback client named `direct-cli`, the encoding it has always
    /// had, so its jobs still conflict with each other and display the same.
    #[must_use]
    pub fn direct_cli() -> Self {
        Self {
            address: Self::DIRECT_CLI_ADDRESS.to_string(),
            name: Some(Self::DIRECT_CLI_NAME.to_owned()),
        }
    }

    /// The client's address.
    #[must_use]
    pub fn address(&self) -> &str {
        &self.address
    }

    /// The name the address resolved to, when one was recorded.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Read the two database columns, where `''` encodes absence.
    ///
    /// An empty address is no submitter. A name with no address is not
    /// something this code writes ([`Self::columns`] cannot produce it), so it
    /// is not a submitter either; it is its own case, so the reader must say
    /// what it does with the name rather than lose it without a word.
    pub(crate) fn from_columns(address: String, name: String) -> StoredSubmitter {
        match (address.is_empty(), name.is_empty()) {
            (true, true) => StoredSubmitter::Absent,
            (true, false) => StoredSubmitter::NameWithoutAddress { name },
            (false, _) => StoredSubmitter::Recorded(Self {
                address,
                name: (!name.is_empty()).then_some(name),
            }),
        }
    }

    /// The two database columns `(submitted_by, submitted_by_name)` for an
    /// optional submitter, spelling absence as the columns' `''`.
    pub(crate) const fn columns(submitter: Option<&Self>) -> (&str, &str) {
        match submitter {
            Some(submitter) => (
                submitter.address.as_str(),
                match &submitter.name {
                    Some(name) => name.as_str(),
                    None => Self::ABSENT_COLUMN,
                },
            ),
            None => (Self::ABSENT_COLUMN, Self::ABSENT_COLUMN),
        }
    }
}

/// What a job row's two submitter columns hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StoredSubmitter {
    /// An address, and the name it resolved to when there was one.
    Recorded(Submitter),
    /// Both columns empty: no submitter was recorded.
    Absent,
    /// A name with an empty address. Not written by this build; a row from
    /// another build or a hand edit. Not a submitter (conflict detection
    /// keys on the address), and the reader reports the name it drops.
    NameWithoutAddress {
        /// The name the row held.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::{StoredSubmitter, Submitter};
    use std::net::Ipv4Addr;

    /// Guards: empty, named and unresolved submitter columns round-trip with
    /// the exact database representation and preserve genuine absence.
    /// Kind: wire
    /// Not a type: the private Submitter fields constrain domain values, while
    /// SQLite column strings still need this precise absent/name/address codec.
    /// State: none, the adapter consumes and produces immutable column values.
    /// Concurrency: none, no shared state or external resource is accessed.
    #[test]
    fn database_columns_round_trip_including_absence() {
        assert_eq!(Submitter::columns(None), ("", ""));
        assert_eq!(
            Submitter::from_columns(String::new(), String::new()),
            StoredSubmitter::Absent
        );

        let named = Submitter::client(Ipv4Addr::new(10, 0, 0, 7).into(), "lab-mac".into());
        let (address, name) = Submitter::columns(Some(&named));
        assert_eq!((address, name), ("10.0.0.7", "lab-mac"));
        assert_eq!(
            Submitter::from_columns(address.to_owned(), name.to_owned()),
            StoredSubmitter::Recorded(named)
        );

        let unnamed = Submitter::client(Ipv4Addr::new(10, 0, 0, 8).into(), String::new());
        assert_eq!(unnamed.name(), None, "an empty resolution records no name");
        let (address, name) = Submitter::columns(Some(&unnamed));
        assert_eq!(
            Submitter::from_columns(address.to_owned(), name.to_owned()),
            StoredSubmitter::Recorded(unnamed)
        );
    }

    /// Guards: an orphan name is returned as `NameWithoutAddress` for reporting.
    /// Kind: boundary
    /// Not a type: stored string columns can contain an orphan name even though
    /// Submitter constructors cannot create that malformed database input.
    /// State: none, this directly admits the external column bytes.
    /// Concurrency: none, the column admission is a pure value operation.
    #[test]
    fn a_name_without_an_address_is_no_submitter_and_is_handed_back() {
        assert_eq!(
            Submitter::from_columns(String::new(), "orphan".into()),
            StoredSubmitter::NameWithoutAddress {
                name: "orphan".into()
            }
        );
    }
}
