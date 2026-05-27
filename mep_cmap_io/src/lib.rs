/*!
mep_cmap_io — Fast file I/O for MEP-CMAP Analyser
===================================================

Rust-backed readers for all text-based formats.  The bottleneck in every
format is the same: reading large ASCII-encoded floating-point files line
by line with Python's float() converter.  Rust's f64::from_str runs
10–20× faster for this workload and avoids the GIL entirely during I/O.

Exported Python functions
-------------------------
Spike2
  spike2_list_channels(path)               -> list[str]
  spike2_extract_waveform(path, ch_idx)    -> (np.ndarray, int, str | None)
  spike2_extract_stim_times(path, marker)  -> dict[str, list[float]]

LabChart
  labchart_list_channels(path)             -> list[str]
  labchart_extract_waveform(path, ch_idx)  -> (np.ndarray, int, str | None)
  labchart_extract_stim_times(path, label) -> dict[str, list[float]]

Generic TSV / KinEMG CSV
  generic_tsv_sniff(path, delimiter, skip_rows)
      -> (n_rows: int, n_cols: int, fs_detected: float | None)

  generic_tsv_extract_waveform(
      path, delimiter, skip_rows, channel_idx, layout,
      time_col, channels_json
  ) -> (np.ndarray, str | None)

  generic_tsv_extract_stim_times(
      path, delimiter, skip_rows, stim_col, layout,
      fs, time_col, trials_stacked
  ) -> dict[str, list[float]]

Design notes for generic TSV
-----------------------------
Column-wise (layout = "column_wise"):
  rows = time samples, cols = channels.
  Loaded into a flat Vec<Vec<f64>> then column-extracted.

Row-wise (layout = "row_wise"):
  rows = channels, cols = time samples.
  The Delsys Trigno pattern: row 0 = TTL trigger, row 1 = EMG.
  Samples-per-row can be 400 k+; loading is done by parsing each row
  into a pre-allocated Vec<f64> and returning the requested row directly.

Startup-artifact handling (row-wise trigger detection):
  The very first sample in a Delsys trigger row is often a large negative
  transient (e.g. -0.75 V).  We threshold on global_max * 0.5 which is
  ~2.5 V for a 5 V TTL rail — comfortably above noise and below any real
  trigger edge.
*/

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{IntoPyArray, PyArray1};
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead};

// ─────────────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Open a file and return all lines in a Vec<String>.
fn open_lines(path: &str) -> io::Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::with_capacity(4 * 1024 * 1024, file);
    reader.lines().collect::<Result<Vec<_>, _>>()
}

/// Map a delimiter name string to its character.
fn delim_char(name: &str) -> char {
    match name {
        "comma" => ',',
        "space" => ' ',
        _       => '\t',   // "tab" or anything else
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spike2 helpers
// ─────────────────────────────────────────────────────────────────────────────

struct Spike2Summary {
    rows: Vec<(usize, i64, Option<String>)>,
}

fn spike2_parse_summary(lines: &[String]) -> Spike2Summary {
    let mut rows: Vec<(usize, i64, Option<String>)> = Vec::new();
    let mut in_summary = false;
    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("\"SUMMARY\"") {
            in_summary = true;
            continue;
        }
        if in_summary {
            if rows.len() > 0 && i > rows[0].0 + 40 {
                break;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let kind = parts[1].trim().trim_matches('"');
                if kind == "Waveform" {
                    let fs = parts[2..]
                        .iter()
                        .filter_map(|t| {
                            let t = t.trim().trim_matches('"');
                            t.parse::<f64>().ok()
                        })
                        .find(|&v| v >= 100.0)
                        .map(|v| v as i64)
                        .unwrap_or(0);
                    let unit = parts
                        .iter()
                        .map(|t| t.trim().trim_matches('"'))
                        .find(|t| {
                            !t.is_empty()
                                && t.chars().all(|c| c.is_alphabetic() || c == 'µ' || c == 'μ')
                                && (t.ends_with('V') || t.ends_with('v'))
                        })
                        .map(|s| s.to_owned());
                    rows.push((i, fs, unit));
                }
            }
        }
    }
    Spike2Summary { rows }
}

// ─────────────────────────────────────────────────────────────────────────────
// Spike2 public functions
// ─────────────────────────────────────────────────────────────────────────────

#[pyfunction]
fn spike2_list_channels(path: &str) -> PyResult<Vec<String>> {
    let lines = open_lines(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {path}: {e}"))
    })?;
    let mut names: Vec<String> = Vec::new();
    let mut in_summary = false;
    let mut count = 0usize;
    for line in &lines {
        if line.starts_with("\"SUMMARY\"") {
            in_summary = true;
            continue;
        }
        if in_summary {
            count += 1;
            if count > 40 {
                break;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 && parts[1].trim().trim_matches('"') == "Waveform" {
                let name = parts[2].trim().trim_matches('"');
                names.push(if name.is_empty() {
                    format!("Chan {}", names.len() + 1)
                } else {
                    name.to_owned()
                });
            }
        }
    }
    if names.is_empty() {
        names.push("Waveform-1".to_owned());
    }
    Ok(names)
}

#[pyfunction]
fn spike2_extract_waveform(
    py: Python<'_>,
    path: &str,
    channel_idx: usize,
) -> PyResult<(Py<PyArray1<f64>>, i64, Option<String>)> {
    let lines = open_lines(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {path}: {e}"))
    })?;

    let summary = spike2_parse_summary(&lines);
    if summary.rows.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No Waveform channels found in SUMMARY.",
        ));
    }
    let (_, fs, unit) = summary
        .rows
        .get(channel_idx)
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "Channel #{} requested but only {} found.",
                channel_idx + 1,
                summary.rows.len()
            ))
        })?
        .clone();

    let start_pos = lines
        .iter()
        .position(|l| l.starts_with("\"START\""))
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No START block found.")
        })?
        + 1;

    let mut pos = start_pos;
    for _ in 0..channel_idx {
        while pos < lines.len() && !lines[pos].starts_with("\"CHANNEL\"") {
            pos += 1;
        }
        pos += 2;
    }

    let mut samples: Vec<f64> = Vec::with_capacity(1 << 20);
    for line in &lines[pos..] {
        if line.starts_with("\"CHANNEL\"") {
            break;
        }
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if let Ok(v) = t.parse::<f64>() {
            samples.push(v);
        }
    }

    let arr = samples.into_pyarray(py).into();
    Ok((arr, fs, unit))
}

#[pyfunction]
fn spike2_extract_stim_times(
    py: Python<'_>,
    path: &str,
    marker_name: &str,
) -> PyResult<PyObject> {
    let lines = open_lines(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {path}: {e}"))
    })?;

    let dict = PyDict::new(py);
    let mut block_start: Option<usize> = None;

    for (i, line) in lines.iter().enumerate() {
        if line.trim().starts_with("\"Marker\"") && i + 2 < lines.len() {
            let current = lines[i + 2].trim().trim_matches('"');
            if current == marker_name {
                block_start = Some(i + 3);
                break;
            }
        }
    }

    let Some(start) = block_start else {
        return Ok(dict.into());
    };

    let mut stim_map: HashMap<String, Vec<f64>> = HashMap::new();
    for line in &lines[start..] {
        if line.trim().starts_with("\"CHANNEL\"") {
            break;
        }
        let parts: Vec<&str> = line.trim().splitn(2, '\t').collect();
        if parts.len() < 2 {
            continue;
        }
        let Ok(ts) = parts[0].parse::<f64>() else {
            continue;
        };
        let label_part = parts[1].trim().trim_matches('"');
        let label = label_part
            .chars()
            .next()
            .map(|c| c.to_uppercase().to_string())
            .unwrap_or_else(|| "A".to_owned());
        stim_map.entry(label).or_default().push(ts);
    }

    for (k, v) in &stim_map {
        dict.set_item(k, v)?;
    }
    Ok(dict.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// LabChart helpers
// ─────────────────────────────────────────────────────────────────────────────

struct LcBlock {
    fs:         i64,
    edt_sec:    f64,
    channels:   Vec<String>,
    units:      Vec<String>,
    data_start: usize,
    data_end:   usize,
}

fn labchart_parse_blocks(lines: &[String]) -> Vec<LcBlock> {
    // Anchor on "Interval=" — the first line of every LabChart block header.
    // Previously anchored on "ChannelTitle=" which placed the header window
    // *after* Interval= and ExcelDateTime=, so both were never found and every
    // block received the fallback values (fs=2000, edt_sec=0.0).  With all
    // edt_sec==0 every block's abs_block_start collapsed to ~0, causing all
    // stim times to be identical and the waveform blocks to overwrite each other.
    let block_starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("Interval="))
        .map(|(i, _)| i)
        .collect();

    let mut blocks: Vec<LcBlock> = Vec::new();
    for (b_idx, &start) in block_starts.iter().enumerate() {
        let header_end = start + 9;
        let header = &lines[start..header_end.min(lines.len())];

        // "Interval=\t0.001 s" — value is tab-separated and has a unit suffix.
        // Must split on tab first, then take the first whitespace-delimited token.
        let fs = header
            .iter()
            .find(|l| l.starts_with("Interval="))
            .and_then(|l| l.split('\t').nth(1))
            .and_then(|v| {
                v.trim()
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse::<f64>().ok())
            })
            .filter(|&v| v > 0.0)
            .map(|interval| (1.0 / interval).round() as i64)
            .unwrap_or(2000);

        // "ExcelDateTime=\t46141.45\t29/04/2026 ..." — value is the first tab field;
        // trailing human-readable date must be discarded before parsing.
        let edt_sec = header
            .iter()
            .find(|l| l.starts_with("ExcelDateTime="))
            .and_then(|l| l.split('\t').nth(1))
            .and_then(|v| v.trim().parse::<f64>().ok())
            .map(|edt| (edt - 25569.0) * 86400.0)
            .unwrap_or(0.0);

        let channels: Vec<String> = header
            .iter()
            .find(|l| l.starts_with("ChannelTitle="))
            .map(|l| {
                l.trim()
                    .split('\t')
                    .skip(1)
                    .map(|s| s.trim().to_owned())
                    .collect()
            })
            .unwrap_or_default();

        let units: Vec<String> = header
            .iter()
            .find(|l| l.starts_with("UnitName"))
            .map(|l| {
                l.trim()
                    .split('\t')
                    .skip(1)
                    .map(|s| s.trim().to_owned())
                    .collect()
            })
            .unwrap_or_default();

        let data_start = start + 9;
        let data_end = block_starts
            .get(b_idx + 1)
            .copied()
            .unwrap_or(lines.len());

        blocks.push(LcBlock {
            fs,
            edt_sec,
            channels,
            units,
            data_start,
            data_end,
        });
    }
    blocks
}

fn lc_abs_start(blocks: &[LcBlock], lines: &[String]) -> f64 {
    let t_local = lines
        .get(blocks[0].data_start)
        .and_then(|l| l.trim().split('\t').next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);
    blocks[0].edt_sec + t_local
}

// ─────────────────────────────────────────────────────────────────────────────
// LabChart public functions
// ─────────────────────────────────────────────────────────────────────────────

#[pyfunction]
fn labchart_list_channels(path: &str) -> PyResult<Vec<String>> {
    let lines = open_lines(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {path}: {e}"))
    })?;
    let blocks = labchart_parse_blocks(&lines);
    if blocks.is_empty() {
        return Ok(vec!["Channel 1".to_owned()]);
    }
    let names: Vec<String> = blocks[0]
        .channels
        .iter()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .collect();
    if names.is_empty() {
        Ok(vec!["Channel 1".to_owned()])
    } else {
        Ok(names)
    }
}

#[pyfunction]
fn labchart_extract_waveform(
    py: Python<'_>,
    path: &str,
    channel_idx: usize,
) -> PyResult<(Py<PyArray1<f64>>, i64, Option<String>)> {
    let lines = open_lines(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {path}: {e}"))
    })?;
    let blocks = labchart_parse_blocks(&lines);
    if blocks.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "No LabChart data blocks found.",
        ));
    }

    let fs = blocks[0].fs;
    let unit = blocks[0]
        .units
        .get(channel_idx)
        .map(|u| u.trim().trim_matches('*').to_owned())
        .filter(|u| !u.is_empty());

    let t0_abs = lc_abs_start(&blocks, &lines);
    let col = channel_idx + 1;

    let last = &blocks[blocks.len() - 1];
    let t_last_local = lines
        .get(last.data_end.saturating_sub(1))
        .and_then(|l| l.trim().split('\t').next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.5);
    let total_samples =
        ((((last.edt_sec + t_last_local) - t0_abs) * fs as f64).ceil() as usize) + 10;

    let mut output = vec![0.0f64; total_samples];

    for block in &blocks {
        let t_local_start = lines
            .get(block.data_start)
            .and_then(|l| l.trim().split('\t').next())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let sample_offset =
            ((((block.edt_sec + t_local_start) - t0_abs) * fs as f64).round() as usize)
                .min(total_samples);

        let mut samples: Vec<f64> = Vec::with_capacity(block.data_end - block.data_start);
        for line in &lines[block.data_start..block.data_end] {
            let parts: Vec<&str> = line.trim().split('\t').collect();
            if parts.len() > col {
                if let Ok(v) = parts[col].parse::<f64>() {
                    samples.push(v);
                }
            }
        }
        if samples.is_empty() {
            continue;
        }
        let end_idx = sample_offset + samples.len();
        if end_idx > output.len() {
            output.resize(end_idx, 0.0);
        }
        output[sample_offset..end_idx].copy_from_slice(&samples);
    }

    let arr = output.into_pyarray(py).into();
    Ok((arr, fs, unit))
}

#[pyfunction]
fn labchart_extract_stim_times(
    py: Python<'_>,
    path: &str,
    marker_name: &str,
) -> PyResult<PyObject> {
    let lines = open_lines(path).map_err(|e| {
        pyo3::exceptions::PyIOError::new_err(format!("Cannot open {path}: {e}"))
    })?;
    let blocks = labchart_parse_blocks(&lines);
    let dict = PyDict::new(py);
    if blocks.is_empty() {
        return Ok(dict.into());
    }

    let fs = blocks[0].fs as f64;
    let t0_abs = lc_abs_start(&blocks, &lines);
    let label = marker_name
        .chars()
        .next()
        .map(|c| c.to_uppercase().to_string())
        .unwrap_or_else(|| "A".to_owned());

    let channels = &blocks[0].channels;
    let stim_ch_idx = channels
        .iter()
        .position(|c| {
            let lc = c.to_lowercase();
            lc.contains("stim") || lc.contains("trig") || lc.contains("ttl")
        })
        .unwrap_or_else(|| 3_usize.min(channels.len().saturating_sub(1)));
    let stim_col = stim_ch_idx + 1;

    let mut stim_times: Vec<f64> = Vec::new();

    for block in &blocks {
        let mut time_v: Vec<f64> = Vec::new();
        let mut stim_v: Vec<f64> = Vec::new();

        for line in &lines[block.data_start..block.data_end] {
            let parts: Vec<&str> = line.trim().split('\t').collect();
            if parts.len() > stim_col {
                if let (Ok(t), Ok(s)) =
                    (parts[0].parse::<f64>(), parts[stim_col].parse::<f64>())
                {
                    time_v.push(t);
                    stim_v.push(s);
                }
            }
        }
        if time_v.is_empty() {
            continue;
        }

        let t_local_start = lines
            .get(block.data_start)
            .and_then(|l| l.trim().split('\t').next())
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(time_v[0]);
        let abs_block_start = (block.edt_sec + t_local_start) - t0_abs;

        let t0_idx = time_v
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        if time_v[t0_idx].abs() < 2.0 / fs {
            stim_times.push(abs_block_start + (time_v[t0_idx] - time_v[0]));
            continue;
        }

        let max_stim = stim_v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_stim > 0.1 {
            let threshold = max_stim * 0.5;
            for i in 1..stim_v.len() {
                if stim_v[i - 1] < threshold && stim_v[i] >= threshold {
                    stim_times.push(abs_block_start + (time_v[i] - time_v[0]));
                    break;
                }
            }
        }
    }

    if !stim_times.is_empty() {
        dict.set_item(label, stim_times)?;
    }
    Ok(dict.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic TSV / KinEMG CSV helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Parse one text line into f64 values, splitting on `sep`.
/// Skips empty fields.  Returns an empty Vec on any parse failure.
#[inline]
fn parse_row(line: &str, sep: char) -> Vec<f64> {
    line.trim()
        .split(sep)
        .filter(|s| !s.is_empty())
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .unwrap_or_default()
}

/// Load the full file as a 2-D Vec<Vec<f64>>.
/// `skip` header lines are discarded.  Empty or unparseable lines are skipped.
fn load_2d(path: &str, sep: char, skip: usize) -> Vec<Vec<f64>> {
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let reader = io::BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (i, line_res) in reader.lines().enumerate() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };
        if i < skip {
            continue;
        }
        let row = parse_row(&line, sep);
        if !row.is_empty() {
            rows.push(row);
        }
    }
    rows
}

// ─────────────────────────────────────────────────────────────────────────────
// Generic TSV public functions
// ─────────────────────────────────────────────────────────────────────────────

/// Quickly inspect a file's shape and look for an embedded sampling rate.
///
/// Returns `(n_data_rows, n_cols_first_row, fs_detected)`.
///
/// Reads only the first data line to count columns, then counts total data
/// lines by scanning the rest of the file.  For a 15 MB file this completes
/// in < 100 ms.  fs_detected is Some(hz) if a line matching common
/// "Sample Clock Rate" / "Sampling Rate" patterns is found in the header.
#[pyfunction]
fn generic_tsv_sniff(
    path: &str,
    delimiter: &str,
    skip_rows: usize,
) -> PyResult<(usize, usize, Option<f64>)> {
    let sep    = delim_char(delimiter);
    let file   = fs::File::open(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Cannot open {path}: {e}")))?;
    let reader = io::BufReader::with_capacity(4 * 1024 * 1024, file);

    // Regex-free fs detection: look for lines containing rate-like keywords
    // with a numeric value.  Covers:
    //   "Sample Clock Rate,2000.00"
    //   "Sampling Rate: 4000"
    //   "fs=2148.1481"
    let fs_keywords = ["sample clock rate", "sampling rate", "samplerate", "fs=", "hz="];

    let mut n_data_rows = 0usize;
    let mut n_cols      = 0usize;
    let mut fs_detected: Option<f64> = None;
    let mut found_first_data = false;

    for (i, line_res) in reader.lines().enumerate() {
        let line = match line_res {
            Ok(l) => l,
            Err(_) => continue,
        };
        let trimmed = line.trim();

        if i < skip_rows {
            // Still in header — scan for fs
            let lower = trimmed.to_lowercase();
            if fs_detected.is_none() {
                for kw in &fs_keywords {
                    if lower.contains(kw) {
                        // Extract first number after the keyword
                        let after = match lower.find(kw) {
                            Some(pos) => &trimmed[pos + kw.len()..],
                            None => continue,
                        };
                        // Strip leading non-numeric chars (spaces, colons, equals, commas)
                        let num_str: String = after
                            .chars()
                            .skip_while(|c| !c.is_ascii_digit() && *c != '-')
                            .take_while(|c| c.is_ascii_digit() || *c == '.')
                            .collect();
                        if let Ok(v) = num_str.parse::<f64>() {
                            if v > 1.0 {
                                fs_detected = Some(v);
                                break;
                            }
                        }
                    }
                }
            }
            continue;
        }

        if trimmed.is_empty() {
            continue;
        }

        let row = parse_row(trimmed, sep);
        if row.is_empty() {
            continue;
        }

        if !found_first_data {
            n_cols      = row.len();
            found_first_data = true;
        }
        n_data_rows += 1;
    }

    Ok((n_data_rows, n_cols, fs_detected))
}

/// Extract a single EMG channel as a 1-D numpy array.
///
/// Parameters
/// ----------
/// path          : file path
/// delimiter     : "tab" | "comma" | "space"
/// skip_rows     : number of header lines to skip
/// channel_idx   : 0-based index into the EMG channels list (already filtered
///                 by the Python caller using the sidecar config)
/// layout        : "column_wise" | "row_wise"
/// target_col    : for column_wise — 0-based column index of this channel;
///                 for row_wise    — 0-based row index of this channel
/// unit          : unit string (passed through unchanged)
///
/// Returns `(samples: np.ndarray[float64], unit: str | None)`.
/// The sampling rate is not returned here; callers read it from the sidecar.
#[pyfunction]
fn generic_tsv_extract_waveform(
    py: Python<'_>,
    path: &str,
    delimiter: &str,
    skip_rows: usize,
    target_col: usize,
    layout: &str,
    unit: Option<String>,
) -> PyResult<(Py<PyArray1<f64>>, Option<String>)> {
    let sep  = delim_char(delimiter);
    let rows = load_2d(path, sep, skip_rows);

    if rows.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "generic_tsv: file contained no parseable data rows.",
        ));
    }

    let samples: Vec<f64> = if layout == "row_wise" {
        // target_col is actually a row index for row-wise files
        rows.get(target_col)
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "generic_tsv row_wise: row index {target_col} out of range \
                     (file has {} rows).",
                    rows.len()
                ))
            })?
            .clone()
    } else {
        // column_wise: extract target_col from every row
        rows.iter()
            .filter_map(|row| row.get(target_col).copied())
            .collect()
    };

    let arr = samples.into_pyarray(py).into();
    Ok((arr, unit))
}

/// Detect stimulation times from the designated trigger channel.
///
/// Parameters
/// ----------
/// path           : file path
/// delimiter      : "tab" | "comma" | "space"
/// skip_rows      : header lines to skip
/// stim_col       : column index (column_wise) or row index (row_wise) of
///                  the stim/trigger channel in the raw data
/// layout         : "column_wise" | "row_wise"
/// fs             : sampling rate in Hz
/// time_col       : column index of the time axis, or -1 if absent
/// trials_stacked : true if the time axis resets each trial (column-wise only)
/// label          : single-char event label, e.g. "A"
///
/// Returns a dict mapping the label to a list of stim times in seconds,
/// relative to the first sample of the waveform array.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn generic_tsv_extract_stim_times(
    py: Python<'_>,
    path: &str,
    delimiter: &str,
    skip_rows: usize,
    stim_col: usize,
    layout: &str,
    fs: f64,
    time_col: i64,        // -1 = absent
    trials_stacked: bool,
    label: &str,
) -> PyResult<PyObject> {
    let sep  = delim_char(delimiter);
    let rows = load_2d(path, sep, skip_rows);
    let dict = PyDict::new(py);

    if rows.is_empty() {
        return Ok(dict.into());
    }

    let event_label = label
        .chars()
        .next()
        .map(|c| c.to_uppercase().to_string())
        .unwrap_or_else(|| "A".to_owned());

    let mut stim_times: Vec<f64> = Vec::new();

    if layout == "row_wise" {
        // ── Row-wise: stim_col is a row index ────────────────────────────────
        let stim_row = match rows.get(stim_col) {
            Some(r) => r,
            None    => return Ok(dict.into()),
        };

        // Threshold on global max (see startup-artifact note in module doc)
        let global_max = stim_row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let thr = global_max * 0.5;
        if thr <= 0.0 {
            return Ok(dict.into());
        }

        let mut prev_above = stim_row[0] >= thr;
        for (i, &v) in stim_row.iter().enumerate().skip(1) {
            let above = v >= thr;
            if above && !prev_above {
                stim_times.push(i as f64 / fs);
            }
            prev_above = above;
        }

    } else {
        // ── Column-wise ───────────────────────────────────────────────────────
        let stim_signal: Vec<f64> = rows
            .iter()
            .filter_map(|row| row.get(stim_col).copied())
            .collect();

        if stim_signal.is_empty() {
            return Ok(dict.into());
        }

        let t_col_valid = time_col >= 0;
        let tc = time_col as usize;

        if trials_stacked && t_col_valid {
            // ── Stacked trials: time axis resets each trial ───────────────────
            let t_axis: Vec<f64> = rows
                .iter()
                .filter_map(|row| row.get(tc).copied())
                .collect();

            // Find reset boundaries
            let mut trial_starts: Vec<usize> = vec![0];
            for i in 1..t_axis.len() {
                if t_axis[i] < t_axis[i - 1] - 1e-9 {
                    trial_starts.push(i);
                }
            }
            trial_starts.push(stim_signal.len());

            for w in trial_starts.windows(2) {
                let s = w[0];
                let e = w[1];
                let sweep = &stim_signal[s..e];
                let max_s = sweep.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let thr   = max_s * 0.5;
                if thr <= 0.0 {
                    continue;
                }
                let mut prev = sweep[0] >= thr;
                for (j, &v) in sweep.iter().enumerate().skip(1) {
                    let above = v >= thr;
                    if above && !prev {
                        stim_times.push((s + j) as f64 / fs);
                        break;
                    }
                    prev = above;
                }
            }
        } else {
            // ── Continuous (no resets) ────────────────────────────────────────
            let global_max = stim_signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let thr = global_max * 0.5;
            if thr <= 0.0 {
                return Ok(dict.into());
            }

            let mut prev_above = stim_signal[0] >= thr;
            for (i, &v) in stim_signal.iter().enumerate().skip(1) {
                let above = v >= thr;
                if above && !prev_above {
                    let t = if t_col_valid {
                        rows.get(i)
                            .and_then(|row| row.get(tc).copied())
                            .unwrap_or(i as f64 / fs)
                    } else {
                        i as f64 / fs
                    };
                    stim_times.push(t);
                }
                prev_above = above;
            }
        }
    }

    if !stim_times.is_empty() {
        dict.set_item(event_label, stim_times)?;
    }
    Ok(dict.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// CFWB binary reader  (.adibin / ADInstruments binary export)
// ─────────────────────────────────────────────────────────────────────────────
//
// Format spec: ADIBinaryFormat.h (ADInstruments, 2001-2009)
// http://cdn.adinstruments.com/adi-web/manuals/translatebinary/ADIBinaryFormat.h
//
// File layout (all values little-endian, 1-byte packed):
//
//   [68 bytes]  File header (CFWBINARY)
//   [96 bytes × NChannels]  Channel headers (CFWBCHANNEL)
//   [interleaved samples]   NChannels (or NChannels+1 if TimeChannel=1) values
//                           per sample, DataFormat = 1 (f64) / 2 (f32) / 3 (i16)
//
// For i16 data: physical = scale × (raw + offset)
// For f32/f64:  scale=1.0, offset=0.0 (applied anyway for safety)

fn cfwb_read_i32(data: &[u8], off: usize) -> i32 {
    i32::from_le_bytes(data[off..off + 4].try_into().unwrap_or([0; 4]))
}
fn cfwb_read_f64(data: &[u8], off: usize) -> f64 {
    f64::from_le_bytes(data[off..off + 8].try_into().unwrap_or([0; 8]))
}
fn cfwb_read_f32(data: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(data[off..off + 4].try_into().unwrap_or([0; 4]))
}
fn cfwb_read_i16(data: &[u8], off: usize) -> i16 {
    i16::from_le_bytes(data[off..off + 2].try_into().unwrap_or([0; 2]))
}
fn cfwb_read_cstr(data: &[u8], off: usize, max_len: usize) -> String {
    let slice = &data[off..(off + max_len).min(data.len())];
    let end   = slice.iter().position(|&b| b == 0).unwrap_or(max_len);
    String::from_utf8_lossy(&slice[..end]).trim().to_owned()
}

struct CfwbHeader {
    secs_per_tick:       f64,
    n_channels:          usize,
    samples_per_channel: usize,
    time_channel:        bool,
    data_format:         i32,   // 1 = f64, 2 = f32, 3 = i16
}

struct CfwbChan {
    title:  String,
    units:  String,
    scale:  f64,
    offset: f64,
}

struct CfwbFile {
    header:      CfwbHeader,
    channels:    Vec<CfwbChan>,
    data_offset: usize,         // byte offset of first sample in `raw`
    raw:         Vec<u8>,
}

fn cfwb_parse(path: &str) -> Result<CfwbFile, String> {
    let raw = fs::read(path).map_err(|e| format!("Cannot read {path}: {e}"))?;

    if raw.len() < 68 {
        return Err("File too small to be a valid CFWB binary".to_owned());
    }
    if &raw[0..4] != b"CFWB" {
        return Err(format!(
            "Not a CFWB binary file — magic bytes are {:?}, expected b\"CFWB\"",
            &raw[0..4]
        ));
    }

    // ── File header ───────────────────────────────────────────────────────────
    // Offset  Size  Field
    //  0       4    magic[4]
    //  4       4    Version (i32)
    //  8       8    secsPerTick (f64)
    // 16       4    Year
    // 20       4    Month
    // 24       4    Day
    // 28       4    Hour
    // 32       4    Minute
    // 36       8    Second (f64)
    // 44       8    trigger (f64)
    // 52       4    NChannels
    // 56       4    SamplesPerChannel
    // 60       4    TimeChannel
    // 64       4    DataFormat
    // = 68 bytes total

    let secs_per_tick       = cfwb_read_f64(&raw, 8);
    let n_channels          = cfwb_read_i32(&raw, 52).max(0) as usize;
    let samples_per_channel = cfwb_read_i32(&raw, 56).max(0) as usize;
    let time_channel        = cfwb_read_i32(&raw, 60) != 0;
    let data_format         = cfwb_read_i32(&raw, 64);

    if n_channels == 0 {
        return Err("CFWB file reports 0 channels".to_owned());
    }
    if secs_per_tick <= 0.0 {
        return Err("CFWB file has invalid secsPerTick".to_owned());
    }

    // ── Channel headers ───────────────────────────────────────────────────────
    // Each: 32 (Title) + 32 (Units) + 8 (scale) + 8 (offset)
    //      + 8 (RangeHigh) + 8 (RangeLow) = 96 bytes
    let mut channels    = Vec::with_capacity(n_channels);
    let mut byte_offset = 68_usize;
    for _ in 0..n_channels {
        if byte_offset + 96 > raw.len() {
            return Err("CFWB file truncated in channel headers".to_owned());
        }
        let title  = cfwb_read_cstr(&raw, byte_offset,      32);
        let units  = cfwb_read_cstr(&raw, byte_offset + 32, 32);
        let scale  = cfwb_read_f64(&raw,  byte_offset + 64);
        let offset = cfwb_read_f64(&raw,  byte_offset + 72);
        channels.push(CfwbChan { title, units, scale, offset });
        byte_offset += 96;
    }

    Ok(CfwbFile {
        header: CfwbHeader {
            secs_per_tick,
            n_channels,
            samples_per_channel,
            time_channel,
            data_format,
        },
        channels,
        data_offset: byte_offset,
        raw,
    })
}

/// Extract one channel of samples from a parsed CFWB file.
/// Returns a Vec<f64> of length samples_per_channel.
fn cfwb_extract_channel(cf: &CfwbFile, channel_idx: usize) -> Vec<f64> {
    let h   = &cf.header;
    let ch  = &cf.channels[channel_idx];

    // Column index inside each interleaved sample row.
    // If TimeChannel=1 the first column is elapsed time — skip it.
    let col_offset = if h.time_channel { 1 } else { 0 };
    let data_cols  = h.n_channels + if h.time_channel { 1 } else { 0 };
    let col        = col_offset + channel_idx;

    let (bytes_per_val, stride) = match h.data_format {
        2 => (4_usize, data_cols * 4),
        3 => (2_usize, data_cols * 2),
        _ => (8_usize, data_cols * 8),   // default: f64
    };
    let _ = bytes_per_val; // used implicitly via stride

    let mut out = Vec::with_capacity(h.samples_per_channel);
    let base    = cf.data_offset;

    for s in 0..h.samples_per_channel {
        let off = base + s * stride + col * match h.data_format {
            2 => 4,
            3 => 2,
            _ => 8,
        };
        if off + match h.data_format { 2 => 4, 3 => 2, _ => 8 } > cf.raw.len() {
            break;
        }
        let physical = match h.data_format {
            2 => {
                let raw_f32 = cfwb_read_f32(&cf.raw, off) as f64;
                ch.scale * (raw_f32 + ch.offset)
            }
            3 => {
                let raw_i16 = cfwb_read_i16(&cf.raw, off) as f64;
                ch.scale * (raw_i16 + ch.offset)
            }
            _ => {
                let raw_f64 = cfwb_read_f64(&cf.raw, off);
                ch.scale * (raw_f64 + ch.offset)
            }
        };
        out.push(physical);
    }
    out
}

// ── Public Python functions ───────────────────────────────────────────────────

/// List channel names from a CFWB binary file.
#[pyfunction]
fn cfwb_list_channels(path: &str) -> PyResult<Vec<String>> {
    let cf = cfwb_parse(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
    Ok(cf.channels.iter().map(|c| c.title.clone()).collect())
}

/// Extract a single EMG channel waveform from a CFWB binary file.
/// Returns (samples: np.ndarray, fs: int, unit: str | None)
#[pyfunction]
fn cfwb_extract_waveform(
    py:          Python<'_>,
    path:        &str,
    channel_idx: usize,
) -> PyResult<(Py<PyArray1<f64>>, i64, Option<String>)> {
    let cf = cfwb_parse(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let idx = channel_idx.min(cf.header.n_channels.saturating_sub(1));
    let unit = {
        let u = cf.channels[idx].units.clone();
        if u.is_empty() { None } else { Some(u) }
    };
    let fs = (1.0 / cf.header.secs_per_tick).round() as i64;

    let samples = cfwb_extract_channel(&cf, idx);
    let arr     = samples.into_pyarray(py).into();
    Ok((arr, fs, unit))
}

/// Detect stimulation times from a CFWB binary file.
///
/// Auto-detects a trigger channel by searching channel titles for
/// "stim", "trig", or "ttl" (case-insensitive); falls back to the
/// last channel if none found.  Rising edges on that channel are
/// returned as absolute timestamps in seconds.
#[pyfunction]
fn cfwb_extract_stim_times(
    py:          Python<'_>,
    path:        &str,
    marker_name: &str,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    let cf   = cfwb_parse(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;

    let label: String = marker_name.chars().next()
        .map(|c| c.to_uppercase().to_string())
        .unwrap_or_else(|| "A".to_owned());

    // Find stim/trigger channel
    let stim_idx = cf.channels.iter().enumerate()
        .find(|(_, c)| {
            let low = c.title.to_lowercase();
            low.contains("stim") || low.contains("trig") || low.contains("ttl")
        })
        .map(|(i, _)| i)
        .unwrap_or(cf.header.n_channels.saturating_sub(1));

    let stim_sig = cfwb_extract_channel(&cf, stim_idx);
    if stim_sig.is_empty() {
        return Ok(dict.into());
    }

    let max_val = stim_sig.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_val <= 0.0 {
        return Ok(dict.into());
    }
    let thr = max_val * 0.5;
    let fs  = 1.0 / cf.header.secs_per_tick;

    let stim_times: Vec<f64> = stim_sig.windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] < thr && w[1] >= thr)
        .map(|(i, _)| (i + 1) as f64 / fs)
        .collect();

    if !stim_times.is_empty() {
        dict.set_item(label, stim_times)?;
    }
    Ok(dict.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────────────────────────

#[pymodule]
fn mep_cmap_io(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(spike2_list_channels, m)?)?;
    m.add_function(wrap_pyfunction!(spike2_extract_waveform, m)?)?;
    m.add_function(wrap_pyfunction!(spike2_extract_stim_times, m)?)?;
    m.add_function(wrap_pyfunction!(labchart_list_channels, m)?)?;
    m.add_function(wrap_pyfunction!(labchart_extract_waveform, m)?)?;
    m.add_function(wrap_pyfunction!(labchart_extract_stim_times, m)?)?;
    m.add_function(wrap_pyfunction!(generic_tsv_sniff, m)?)?;
    m.add_function(wrap_pyfunction!(generic_tsv_extract_waveform, m)?)?;
    m.add_function(wrap_pyfunction!(generic_tsv_extract_stim_times, m)?)?;
    m.add_function(wrap_pyfunction!(cfwb_list_channels, m)?)?;
    m.add_function(wrap_pyfunction!(cfwb_extract_waveform, m)?)?;
    m.add_function(wrap_pyfunction!(cfwb_extract_stim_times, m)?)?;
    Ok(())
}
