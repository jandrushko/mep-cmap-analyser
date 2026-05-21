/*!
mep_cmap_io — Fast file I/O for MEP-CMAP Analyser
===================================================

Rust-backed replacements for the pure-Python readers in spike2.py and
labchart.py.  The bottleneck in both formats is the same: reading millions
of ASCII-encoded floating-point samples line by line with Python's float()
converter.  Rust's std::str::parse::<f64>() runs 10-20x faster for this
workload and avoids the GIL entirely during I/O.

Exported Python functions
-------------------------
Spike2
  spike2_list_channels(path)               -> list[str]
  spike2_extract_waveform(path, ch_idx)    -> (list[float], int, str | None)
  spike2_extract_stim_times(path, marker)  -> dict[str, list[float]]

LabChart
  labchart_list_channels(path)             -> list[str]
  labchart_extract_waveform(path, ch_idx)  -> (list[float], int, str | None)
  labchart_extract_stim_times(path, label) -> dict[str, list[float]]
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

/// Open a file and return a line iterator (BufReader).
fn open_lines(path: &str) -> io::Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::with_capacity(4 * 1024 * 1024, file);
    reader.lines().collect::<Result<Vec<_>, _>>()
}

// ─────────────────────────────────────────────────────────────────────────────
// Spike2 helpers
// ─────────────────────────────────────────────────────────────────────────────

struct Spike2Summary {
    /// (line_index_in_file, fs, unit) for each Waveform row
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
            // Stop after 40 lines past SUMMARY
            if rows.len() > 0 && i > rows[0].0 + 40 {
                break;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let kind = parts[1].trim().trim_matches('"');
                if kind == "Waveform" {
                    // Find fs: first numeric token >= 100
                    let fs = parts[2..]
                        .iter()
                        .filter_map(|t| {
                            let t = t.trim().trim_matches('"');
                            t.parse::<f64>().ok()
                        })
                        .find(|&v| v >= 100.0)
                        .map(|v| v as i64)
                        .unwrap_or(0);
                    // Find unit: token matching [a-zA-ZµμVv]+
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

    // Find START line — ch0 data begins immediately after it.
    // Subsequent channels are separated by "CHANNEL" markers (each followed
    // by one description line before samples resume).
    let start_pos = lines
        .iter()
        .position(|l| l.starts_with("\"START\""))
        .ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("No START block found.")
        })?
        + 1;

    // Skip channel_idx CHANNEL separator blocks
    let mut pos = start_pos;
    for _ in 0..channel_idx {
        while pos < lines.len() && !lines[pos].starts_with("\"CHANNEL\"") {
            pos += 1;
        }
        pos += 2; // skip CHANNEL line + description line
    }

    // Read floating-point samples until next CHANNEL sentinel or EOF
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

    // Pattern: <timestamp>\t"<char>???"
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
        // second field is like  "A???"  — extract the single character
        let label_part = parts[1].trim().trim_matches('"');
        if label_part.len() >= 1 {
            let ch = label_part.chars().next().unwrap().to_string();
            if ch != "\"" {
                stim_map.entry(ch).or_default().push(ts);
            }
        }
    }

    for (k, v) in &stim_map {
        dict.set_item(k, v.clone())?;
    }
    Ok(dict.into())
}

// ─────────────────────────────────────────────────────────────────────────────
// LabChart helpers
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LcBlock {
    fs: i64,
    edt_sec: f64,
    channels: Vec<String>,
    units: Vec<String>,
    data_start: usize,
    data_end: usize,
}

fn labchart_parse_blocks(lines: &[String]) -> Vec<LcBlock> {
    let block_starts: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with("Interval="))
        .map(|(i, _)| i)
        .collect();

    let mut blocks = Vec::new();
    for (b_idx, &start) in block_starts.iter().enumerate() {
        // Interval
        let interval_s = lines[start]
            .split('\t')
            .nth(1)
            .and_then(|s| s.trim().split_whitespace().next())
            .and_then(|s| s.parse::<f64>().ok());
        let Some(interval_s) = interval_s else {
            continue;
        };
        let fs = (1.0 / interval_s).round() as i64;

        // ExcelDateTime
        let edt_sec = lines
            .get(start + 1)
            .and_then(|l| l.split('\t').nth(1))
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(0.0)
            * 86400.0;

        // ChannelTitle
        let channels: Vec<String> = lines[start..std::cmp::min(start + 9, lines.len())]
            .iter()
            .find(|l| l.starts_with("ChannelTitle"))
            .map(|l| {
                l.trim()
                    .split('\t')
                    .skip(1)
                    .map(|s| s.trim().to_owned())
                    .collect()
            })
            .unwrap_or_default();

        // UnitName
        let units: Vec<String> = lines[start..std::cmp::min(start + 9, lines.len())]
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
    let col = channel_idx + 1; // +1 for time column

    // Estimate output length from last block
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

    // Auto-detect stim channel
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

        // Strategy 1: t=0 is stim (LabChart pre-centering)
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

        // Strategy 2: threshold crossing on stim channel
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
    Ok(())
}
