//! Converting a Python value into `serde_json::Value` by EXACT type.
//!
//! Every worker response and every provider payload crosses the boundary
//! through this function, so what it accepts is the real wire contract. Until
//! 2026-09-16 it asked its questions in the wrong order: numeric extraction ran
//! before strings and containers, and each test was "can this be read as an
//! `i64`" rather than "is this an `int`". PyO3's numeric extraction honours
//! `__int__` / `__float__` / `__index__`, so anything number-like answered yes.
//! A `numpy.float64` became a JSON number silently, which is defensible, and a
//! type nobody had considered became one too, which is not: the conversion was
//! deciding what a value MEANT rather than reading what it WAS.
//!
//! The order below is the contract, and each step is an exact-type test:
//!
//! 1. `None`
//! 2. `bool` (before `int`, because Python's `bool` IS an `int` subclass, and
//!    an exact test is what keeps `True` from becoming `1`)
//! 3. `str`
//! 4. anything with `model_dump` (a Pydantic model, dumped in JSON mode and
//!    re-read through this same function)
//! 5. `dict`, then `list` or `tuple`
//! 6. exact `int`, refused when it does not fit the signed or unsigned 64-bit
//!    range rather than wrapping
//! 7. exact `float`, refused when it is not finite, NAMING THE PATH so a NaN
//!    deep inside a response is locatable instead of being reported as "invalid
//!    float in callback response"
//! 8. a numpy scalar, read through numpy's own `item()` and re-asked as the
//!    Python scalar it yields
//! 9. anything else: refused, naming the type and the path
//!
//! Step 8 is admission by NAME, not a return to coercion, and the difference is
//! the whole point of the order above. `numpy.float64` IS a `float` subclass and
//! `numpy.int64` is a real integer, so both are exact numbers whose value
//! survives the crossing unchanged; an object that merely ANSWERS `__float__`
//! is not, and is still refused at step 9. The first version of this module
//! removed the numeric coercion and the numpy case together, which was a defect:
//! whisper forced alignment returns `float64` word timings, so every FA group
//! was refused and its words were left unaligned while the job still reported
//! success.
//!
//! Containers are tested before numbers on purpose: a container is never a
//! number, and testing numbers first is what let a number-like container-ish
//! object take the wrong branch.

use crate::error::BatchalignBoundaryError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyModule, PyString, PyTuple};

/// Where a value sits inside the payload being converted.
///
/// A borrowed cons list rather than an owned `Vec<String>`: the happy path
/// converts millions of values and must not allocate a breadcrumb for each one,
/// while a refusal (which allocates once, when rendering) is rare.
#[derive(Clone, Copy)]
pub(crate) enum ValuePath<'a> {
    /// The value handed to the conversion.
    Root,
    /// A value under a dictionary key.
    Key {
        /// The enclosing value's path.
        parent: &'a ValuePath<'a>,
        /// The key this value sits under.
        key: &'a str,
    },
    /// A value at a sequence position.
    Index {
        /// The enclosing value's path.
        parent: &'a ValuePath<'a>,
        /// The position this value sits at.
        index: usize,
    },
}

impl std::fmt::Display for ValuePath<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Root => f.write_str("$"),
            Self::Key { parent, key } => write!(f, "{parent}.{key}"),
            Self::Index { parent, index } => write!(f, "{parent}[{index}]"),
        }
    }
}

/// Convert a generic Python value into a JSON value for Rust-side parsing.
///
/// The Rust boundary, not Python, decides which shapes are accepted; see the
/// module documentation for the exact order and why it is that order.
pub(crate) fn py_to_json_value(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    convert(value, &ValuePath::Root)
}

fn refuse(path: &ValuePath<'_>, detail: &str) -> PyErr {
    BatchalignBoundaryError::internal(format!("{path}: {detail}")).into_py_err()
}

fn convert(value: &Bound<'_, PyAny>, path: &ValuePath<'_>) -> PyResult<serde_json::Value> {
    if value.is_none() {
        return Ok(serde_json::Value::Null);
    }
    // Before `int`: Python's `bool` is an `int` subclass, so an inexact test
    // would turn `True` into `1` and lose the distinction the wire draws.
    if value.is_exact_instance_of::<PyBool>() {
        return Ok(serde_json::Value::Bool(value.extract::<bool>()?));
    }
    if value.is_exact_instance_of::<PyString>() {
        return Ok(serde_json::Value::String(value.extract::<String>()?));
    }
    if value.hasattr("model_dump")? {
        let kwargs = PyDict::new(value.py());
        kwargs.set_item("mode", "json")?;
        let dumped = value.call_method("model_dump", (), Some(&kwargs))?;
        return convert(dumped.as_any(), path);
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut object = serde_json::Map::with_capacity(dict.len());
        for (key, item) in dict.iter() {
            // An exact `str` key only. A key that merely converts to one (an
            // enum member, a numpy scalar) would silently rename the field it
            // addresses, which is indistinguishable downstream from the
            // producer having sent a different field.
            if !key.is_exact_instance_of::<PyString>() {
                return Err(refuse(
                    path,
                    &format!(
                        "object key is a {}, and only an exact str names a JSON field",
                        type_name(&key)
                    ),
                ));
            }
            let key = key.extract::<String>()?;
            let child = ValuePath::Key {
                parent: path,
                key: &key,
            };
            let converted = convert(&item.into_any(), &child)?;
            object.insert(key, converted);
        }
        return Ok(serde_json::Value::Object(object));
    }
    if let Ok(list) = value.cast::<PyList>() {
        return convert_sequence(list.iter(), list.len(), path);
    }
    if let Ok(tuple) = value.cast::<PyTuple>() {
        return convert_sequence(tuple.iter(), tuple.len(), path);
    }
    if value.is_exact_instance_of::<PyInt>() {
        return convert_int(value, path);
    }
    if value.is_exact_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        return serde_json::Number::from_f64(number)
            .map(serde_json::Value::Number)
            .ok_or_else(|| {
                refuse(
                    path,
                    &format!("{number} is not a finite number, so it has no JSON representation"),
                )
            });
    }
    // After the exact scalar tests: a numpy scalar is a real number, but it is
    // not an EXACT `int` or `float`, so it reaches this point and is admitted
    // by name rather than by the numeric coercion this module removed.
    if let Some(scalar) = numpy_scalar_item(value)? {
        return convert(&scalar, path);
    }
    Err(refuse(
        path,
        &format!(
            "value is a {}, which is not one of the shapes this boundary accepts \
             (None, bool, str, a model with model_dump, dict, list, tuple, int, \
             float, numpy scalar)",
            type_name(value)
        ),
    ))
}

/// `numpy.generic`, the base class of every numpy scalar, or `None` where numpy
/// is not importable in this interpreter.
///
/// Looked up once: the happy path converts millions of values, and importing a
/// module per value would cost more than the conversion. `None` is cached too,
/// so an interpreter without numpy pays the import attempt once and then simply
/// never matches this arm.
static NUMPY_GENERIC: pyo3::sync::PyOnceLock<Option<Py<PyAny>>> = pyo3::sync::PyOnceLock::new();

/// The Python scalar inside a numpy scalar, or `None` when this is not one.
///
/// The test is `isinstance(value, numpy.generic)`, which is exact about what the
/// value IS: it admits `float64`, `int64`, `bool_` and the rest, and says
/// nothing about an object that merely implements `__float__`. `item()` is
/// numpy's own conversion to the native Python scalar, so the value that
/// continues through `convert` is an ordinary `bool`, `int`, `float` or `str`
/// and is held to exactly the same rules as one written by hand. A numpy scalar
/// whose `item()` yields something this boundary does not accept (a `complex`,
/// say) is refused by the ordinary path, naming that type.
fn numpy_scalar_item<'py>(value: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = value.py();
    let generic = NUMPY_GENERIC.get_or_init(py, || {
        PyModule::import(py, "numpy")
            .and_then(|module| module.getattr("generic"))
            .map(pyo3::Bound::unbind)
            .ok()
    });
    let Some(generic) = generic.as_ref() else {
        return Ok(None);
    };
    if !value.is_instance(generic.bind(py))? {
        return Ok(None);
    }
    Ok(Some(value.call_method0("item")?))
}

fn convert_sequence<'py>(
    items: impl Iterator<Item = Bound<'py, PyAny>>,
    len: usize,
    path: &ValuePath<'_>,
) -> PyResult<serde_json::Value> {
    let mut converted = Vec::with_capacity(len);
    for (index, item) in items.enumerate() {
        let child = ValuePath::Index {
            parent: path,
            index,
        };
        converted.push(convert(&item, &child)?);
    }
    Ok(serde_json::Value::Array(converted))
}

/// Read an exact Python `int`, refusing one that does not fit 64 bits.
///
/// Python integers are unbounded. `serde_json` numbers are not, and an `i64`
/// extraction that overflows raises, so the two ranges are tried explicitly and
/// a value outside both is refused by name. Wrapping it would produce a number
/// that is not the one the producer sent.
fn convert_int(value: &Bound<'_, PyAny>, path: &ValuePath<'_>) -> PyResult<serde_json::Value> {
    if let Ok(signed) = value.extract::<i64>() {
        return Ok(serde_json::Value::Number(signed.into()));
    }
    if let Ok(unsigned) = value.extract::<u64>() {
        return Ok(serde_json::Value::Number(unsigned.into()));
    }
    Err(refuse(
        path,
        "integer does not fit the signed or unsigned 64-bit range, \
         so it cannot cross this boundary without changing value",
    ))
}

/// Convert a JSON value into the Python object it describes.
///
/// The inverse direction of [`py_to_json_value`], and the route by which a
/// TYPED Rust value reaches a Python model: serialize the type once with serde,
/// convert here, and let the Python model parse what arrives. That keeps one
/// spelling of every tag and field name, the one the `Serialize` derive
/// produces and the JSON Schema in `ipc-schema/` is generated from. A dict
/// assembled by hand at the call site would be a second spelling of the same
/// wire shape, free to drift from both while still compiling.
pub(crate) fn json_value_to_py<'py>(
    py: Python<'py>,
    value: &serde_json::Value,
) -> PyResult<Bound<'py, PyAny>> {
    match value {
        serde_json::Value::Null => Ok(py.None().into_bound(py)),
        serde_json::Value::Bool(flag) => Ok(PyBool::new(py, *flag).to_owned().into_any()),
        serde_json::Value::Number(number) => json_number_to_py(py, number),
        serde_json::Value::String(text) => Ok(PyString::new(py, text).into_any()),
        serde_json::Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_value_to_py(py, item)?)?;
            }
            Ok(list.into_any())
        }
        serde_json::Value::Object(entries) => {
            let dict = PyDict::new(py);
            for (key, item) in entries {
                dict.set_item(key, json_value_to_py(py, item)?)?;
            }
            Ok(dict.into_any())
        }
    }
}

/// Read a JSON number as the Python number it is.
///
/// Asked in the order the JSON data model allows: an integer stays an integer,
/// so a count does not reach Python as `2.0` and come back as a float.
fn json_number_to_py<'py>(
    py: Python<'py>,
    number: &serde_json::Number,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(value) = number.as_u64() {
        return Ok(value.into_pyobject(py)?.into_any());
    }
    if let Some(value) = number.as_i64() {
        return Ok(value.into_pyobject(py)?.into_any());
    }
    if let Some(value) = number.as_f64() {
        return Ok(value.into_pyobject(py)?.into_any());
    }
    Err(BatchalignBoundaryError::internal(format!(
        "{number} is not representable as a Python number"
    ))
    .into_py_err())
}

/// The value's Python type name, for a refusal an operator can act on.
fn type_name(value: &Bound<'_, PyAny>) -> String {
    value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_else(|_| "value of an unreadable type".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The path rendering is what makes a refusal locatable, and it is pure
    /// string work, so it is checked without a Python interpreter. The
    /// conversion itself is exercised from the Python side, against the real
    /// interpreter, in `batchalign/tests/test_py_json_bridge.py`: that is the
    /// only place the exact-type questions mean anything.
    #[test]
    fn a_path_renders_the_route_to_the_offending_value() {
        let root = ValuePath::Root;
        assert_eq!(root.to_string(), "$");

        let monologues = ValuePath::Key {
            parent: &root,
            key: "monologues",
        };
        let first = ValuePath::Index {
            parent: &monologues,
            index: 0,
        };
        let elements = ValuePath::Key {
            parent: &first,
            key: "elements",
        };
        let second = ValuePath::Index {
            parent: &elements,
            index: 1,
        };
        let start = ValuePath::Key {
            parent: &second,
            key: "start_s",
        };
        assert_eq!(start.to_string(), "$.monologues[0].elements[1].start_s");
    }
}
