//! Compare transcript pairs and print every compare metric, with a roll-up.
//!
//! Runs the library comparison directly on CHAT files, so a benchmark's
//! scoring can be measured on existing transcripts without transcribing
//! anything. Each argument pair is a main (hypothesis) file and its gold file:
//!
//! ```text
//! cargo run -p batchalign-transform --example compare_metrics -- \
//!     main1.cha gold1.cha [main2.cha gold2.cha ...]
//! ```
//!
//! Output is CSV with a `pair,metric,value` header: every row of each pair's
//! `.compare.csv`, then `ALL` rows summing every count across pairs. Rates are
//! not summed; recompute them from the summed counts. Both files are parsed
//! leniently, as `compare` parses a gold companion, and the gold is taken as a
//! complete reference.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use batchalign_transform::compare::{
    CompareMetricValue, CompareMetricsCsvTable, GoldCoverage, compare,
};
use talkbank_model::{ChatFile, ErrorCollector};
use talkbank_parser::TreeSitterParser;

fn parse(parser: &TreeSitterParser, path: &Path) -> Result<ChatFile, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let errors = ErrorCollector::new();
    Ok(parser.parse_chat_file_streaming(&text, &errors))
}

fn main() -> Result<(), Box<dyn Error>> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (pairs, []) = arguments.as_chunks::<2>() else {
        return Err("arguments must be main/gold pairs: main1.cha gold1.cha ...".into());
    };
    if pairs.is_empty() {
        return Err("give at least one main/gold pair".into());
    }

    let parser = TreeSitterParser::new().map_err(|error| format!("parser: {error}"))?;
    let mut totals: BTreeMap<String, usize> = BTreeMap::new();
    let mut output = csv::Writer::from_writer(std::io::stdout());
    output.write_record(["pair", "metric", "value"])?;

    for [main_path, gold_path] in pairs {
        let main = parse(&parser, Path::new(main_path))?;
        let gold = parse(&parser, Path::new(gold_path))?;
        let metrics = compare(&main, &gold, GoldCoverage::Complete).metrics;
        let table = CompareMetricsCsvTable::from_metrics(&metrics)?;

        let label = Path::new(main_path)
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| main_path.clone());
        for row in &table.rows {
            let key = row.metric.to_csv_field();
            let value = row.value.to_csv_field();
            match row.value {
                CompareMetricValue::Count(count) => {
                    *totals.entry(key.clone()).or_default() += count;
                }
                CompareMetricValue::Decimal(_) | CompareMetricValue::Rate(_) => {}
            }
            output.write_record([label.as_str(), key.as_str(), value.as_str()])?;
        }
    }

    for (key, total) in &totals {
        output.write_record(["ALL", key.as_str(), total.to_string().as_str()])?;
    }
    output.flush()?;
    Ok(())
}
