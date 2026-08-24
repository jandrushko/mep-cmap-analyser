"""Build the TMS-RAT v1.0 mapping file from the official workbook.

The workbook is the authority for item ids, titles, descriptions and the
per-item rating vocabulary. Only the SOURCE classification is ours.
"""
import json, os, re, sys
from openpyxl import load_workbook

#: The official workbook, kept beside this script so the mapping can always be
#: regenerated and checked against the version it was built from. A mapping
#: whose source is not in the repository is a file nobody can verify, and the
#: first person to edit it by hand silently detaches it from the real tool.
HERE = os.path.dirname(os.path.abspath(__file__))
XLSX = os.path.join(HERE, "TMS_Reporting_Assessment_Tool_v1.0.xlsx")

#: Written into the package, where mep_cmap.reporting reads it from.
OUT = os.path.join(HERE, os.pardir, "mep_cmap", "schema", "tms_rat_v1_0.json")

# ── Source classification ────────────────────────────────────────────────
# derived      : the tool computes it from the recording or the analysis config
# nibs_bids    : present in the NIBS-BIDS sidecar IF the analyst filled it in
# nibs_gap     : the item needs a field NIBS-BIDS (BEP037 v0.3.0) does not have
# participants : comes from a BIDS participants.tsv if one is supplied
# out_of_scope : the tool cannot know it, and no plausible input would supply it
#
# "provider" names the function that will supply a derived value. Nothing is
# implemented yet; the name is the contract.
SOURCE = {
 # A. Participant bio -- participants.tsv or nothing
 "A1": ("participants", "age"),
 "A2": ("participants", "sex"),
 "A3": ("participants", "handedness"),
 "A4": ("participants", "height"),
 "A5": ("participants", "weight"),
 # B. Participant state -- screening is never visible to an analysis tool
 "B1": ("out_of_scope", None), "B2": ("out_of_scope", None),
 "B3": ("out_of_scope", None), "B4": ("out_of_scope", None),
 "B5": ("out_of_scope", None), "B6": ("out_of_scope", None),
 "B7": ("out_of_scope", None),
 # C. Design
 "C1": ("out_of_scope", None),          # how many experiments an ARTICLE reports
 "C2": ("participants", "population"),  # healthy/clinical: a participants.tsv column
 "C3": ("participants", "groups"),
 "C4": ("derived", "protocol_description"),
 "C5": ("derived", "time_of_day"),      # recording start time from the file header
 "C6": ("derived", "session_duration"), # span of stimulus times; queue for intervals
 "C7": ("out_of_scope", None),
 "C8": ("derived", "control_condition"),# conditioned vs reference assignment
 "C9": ("out_of_scope", None),
 # D. Hardware
 "D1": ("nibs_bids", "Manufacturer+ManufacturersModelName"),
 "D2": ("nibs_bids", "CoilType+CoilModel"),
 "D3": ("nibs_gap",  "CoilDiameter"),
 "D4": ("nibs_bids", "PulseShape"),
 # E. Locations
 "E1": ("nibs_bids", "TargetingMethod+NeuronavigationSystem+StructuralMRI"),
 "E2": ("nibs_gap",  "CoilLocationOnScalp"),   # target coords are the BRAIN, not skin
 "E3": ("nibs_bids", "TargetRegion+TargetCoordinates+TargetCoordinateSystem"),
 "E4": ("nibs_bids", "CoilOrientation"),
 "E5": ("nibs_bids", "InducedCurrentDirection"),
 "E6": ("nibs_gap",  "CoilStabilityMethod+CoilStabilityTolerance"),
 # F. Intensity
 "F1": ("nibs_bids", "IntensityReference+IntensityScaling"),
 "F2": ("derived",   "machine_intensity"),   # StimulationIntensity per set, mean+SD
 "F3": ("derived",   "participant_intensity"),
 # G. Single pulse -- entirely the tool's
 "G0": ("gate", "single_pulse_relevant"),
 "G1": ("derived", "trial_to_trial_interval"),
 "G2": ("derived", "number_of_pulses"),
 "G3": ("derived", "evoked_response_raw"),
 # H. Motor threshold
 "H0": ("gate", "threshold_relevant"),
 "H1": ("nibs_gap",  "ThresholdLocationMethod"),
 "H2": ("nibs_bids", "RestingMotorThreshold|ActiveMotorThreshold"),
 "H3": ("nibs_gap",  "ThresholdCriterion"),
 "H4": ("nibs_bids", "MotorThresholdMethod"),
 "H5": ("nibs_bids", "RestingMotorThreshold|ActiveMotorThreshold"),
 # I. Motor variables
 "I0": ("gate", "motor_relevant"),
 "I1": ("out_of_scope", None),
 "I2": ("out_of_scope", None),
 "I3": ("nibs_gap", "TargetMuscleActivity"),  # %MVC; cSP implies contraction only
 # J. EMG and muscles
 "J0": ("gate", "emg_relevant"),
 "J1": ("derived", "emg_hardware"),      # inferred from format; model needs metadata
 "J2": ("derived", "emg_software"),
 "J3": ("out_of_scope", None),
 "J4": ("nibs_bids", "ElectrodeType+ElectrodeSize"),
 "J5": ("derived", "electrode_location"),   # TargetMuscle + Limb
 "J6": ("nibs_gap", "ElectrodeMontage+InterElectrodeDistance"),
 "J7": ("derived", "sampling_and_filtering"),
 "J8": ("derived", "emg_measure"),
 "J9": ("derived", "amplitude_normalisation"),
 # K. Afferent stimulation
 "K0": ("gate", "afferent_relevant"),
 "K1": ("nibs_gap", "PeripheralStimulatorManufacturer+Model"),
 "K2": ("nibs_gap", "AfferentElectrodeType+Orientation"),
 "K3": ("nibs_gap", "AfferentStimulusLocation"),
 "K4": ("nibs_gap", "AfferentStimulusTargetNerve"),
 "K5": ("nibs_gap", "AfferentPulseDuration+Waveform"),
 "K6": ("nibs_gap", "AfferentThresholdType"),
 "K7": ("nibs_gap", "AfferentThresholdValue"),
 # L. Conditioning
 "L0": ("gate", "conditioning_relevant"),
 "L1": ("nibs_bids", "ConditioningStimulusIntensity"),
 "L2": ("nibs_bids", "ConditioningStimulusIntensity+IntensityReference"),
 "L3": ("nibs_bids", "InterStimulusInterval|InterPulseInterval"),
 "L4": ("derived", "conditioned_order"),
 "L5": ("derived", "number_of_conditioned_pulses"),
 "L6": ("derived", "conditioned_response_raw"),
 "L7": ("derived", "conditioned_response_relative"),
 # M. SEP -- no EEG in this tool
 "M0": ("gate", "sep_relevant"),
 "M1": ("out_of_scope", None), "M2": ("out_of_scope", None),
 "M3": ("out_of_scope", None), "M4": ("out_of_scope", None),
 # N. Repetitive -- single/paired pulse tool
 "N0": ("gate", "repetitive_relevant"),
 "N1": ("out_of_scope", None), "N2": ("out_of_scope", None),
 "N3": ("out_of_scope", None), "N4": ("out_of_scope", None),
 "N5": ("out_of_scope", None), "N6": ("out_of_scope", None),
 "N7": ("out_of_scope", None), "N8": ("out_of_scope", None),
}

# ── What "Full" needs, fact by fact ──────────────────────────────────────
# An item is Full only when every listed fact is present. Listed here so a
# provider that supplies some of them rates Part and the validator can NAME
# what is missing, rather than reporting an incompleteness the analyst then
# has to diagnose. Also gives the methods generator its gap markers: an
# unfilled fact becomes "[TMS-RAT J1: EMG amplifier model]".
#
# Items needing a single fact are omitted; absence means "one fact, the
# provider either has it or does not".
#
# Wording follows the v1.0 guidance for each item.
FULL_REQUIRES = {
 "A1": ["central tendency", "dispersion"],
 "A2": ["count per category", "categories sum to N"],
 "A3": ["count per category", "categories sum to N"],
 "A4": ["central tendency", "dispersion"],
 "A5": ["central tendency", "dispersion"],
 "C6": ["session duration", "between-session interval"],
 "D1": ["manufacturer", "model"],
 "D2": ["coil geometry", "coil model or 'custom'"],
 "D3": ["diameter value", "which dimension it refers to"],
 "E1": ["navigation method", "scan type or template"],
 "E4": ["angle", "reference plane or point"],
 "E6": ["stability method", "numerical tolerance"],
 "F2": ["central tendency", "dispersion", "per condition"],
 "F3": ["central tendency", "dispersion", "per condition"],
 "G1": ["central tendency", "dispersion"],
 "G3": ["central tendency", "dispersion"],
 "H4": ["hits required", "pulses delivered"],
 "H5": ["central tendency", "dispersion"],
 "I3": ["activity level", "how it was monitored"],
 "J1": ["manufacturer", "model"],
 "J2": ["software", "version"],
 "J4": ["electrode type", "material", "size"],
 "J5": ["muscle", "side of body"],
 "J6": ["montage", "inter-electrode distance or anatomical detail"],
 "J7": ["sampling rate", "filter type", "cut-off frequencies",
        "analysis filtering if any"],
 "J8": ["feature measured", "time interval", "extraction algorithm"],
 "K2": ["electrode or stimulator type", "orientation on the body"],
 "K3": ["side of body", "location on body"],
 "K5": ["pulse duration", "waveform"],
 "K7": ["central tendency", "dispersion"],
 "L1": ["central tendency", "dispersion"],
 "L2": ["central tendency", "dispersion"],
 "L3": ["interval", "dispersion if variable"],
 "L6": ["central tendency", "dispersion"],
 "L7": ["central tendency", "dispersion"],
 "M1": ["frequency or inter-pulse interval", "repetitions or duration"],
 "M2": ["active electrode", "reference electrode"],
 "M3": ["component name", "amplitude or timing feature"],
 "M4": ["central tendency", "dispersion"],
 "N2": ["pulses per burst or total duration"],
}


# ── How level 2 combines a per-file fact into a study-level one ──────────
# The level-1 sidecar records one value per recording. A study-level
# assessment has to turn many of those into one, and HOW differs by item:
#
#   "unique"  the value must be the SAME in every file. A study whose files
#             disagree cannot state one value, and that is an analysis fault
#             before it is a reporting one -- two filter settings across 24
#             recordings is not a methods-section problem, it is two studies.
#             Level 2 reports the distinct values and which files hold them.
#   "mean_sd" pooled across files to a central tendency and a dispersion.
#             This is what makes the QUANTITATIVE items reachable at all:
#             one recording is N=1 and has no dispersion to report.
#   "sum"     totalled across files.
#   "list"    collected; the guidance accepts a full list of individual
#             values as Full reporting.
AGGREGATE = {
 # Must agree across the study
 "C4": "unique", "C8": "unique",
 "D1": "unique", "D2": "unique", "D3": "unique", "D4": "unique",
 "E1": "unique", "E4": "unique", "E5": "unique",
 "F1": "unique",
 "H2": "unique", "H4": "unique",
 "J1": "unique", "J2": "unique", "J4": "unique", "J6": "unique",
 "J7": "unique", "J8": "unique", "J9": "unique",
 "K1": "unique", "K4": "unique", "K5": "unique", "K6": "unique",
 "L3": "unique",
 # Pooled to mean and dispersion
 "A1": "mean_sd", "A4": "mean_sd", "A5": "mean_sd",
 "F2": "mean_sd", "F3": "mean_sd",
 "G1": "mean_sd", "G3": "mean_sd",
 "H5": "mean_sd", "K7": "mean_sd",
 "L1": "mean_sd", "L2": "mean_sd", "L6": "mean_sd", "L7": "mean_sd",
 "M4": "mean_sd",
 # Totalled
 "G2": "sum", "L5": "sum",
 # Collected
 "A2": "list", "A3": "list", "C5": "list", "C6": "list",
 "E2": "list", "E3": "list", "J5": "list", "K2": "list", "K3": "list",
}


def build():
    wb = load_workbook(XLSX, data_only=False)
    ws = wb["INPUT"]

    vocab = {}
    for dv in ws.data_validations.dataValidation:
        opts = [o.strip() for o in str(dv.formula1).strip('"').split(",")]
        for rng in str(dv.sqref).split():
            if ":" in rng:
                a, b = rng.split(":")
                for r in range(int(re.sub(r"\D", "", a)),
                               int(re.sub(r"\D", "", b)) + 1):
                    vocab[r] = opts
            else:
                vocab[int(re.sub(r"\D", "", rng))] = opts

    g = wb["GUIDE"]
    protocols = [g.cell(row=2, column=c).value for c in range(3, 10)]
    relevance = {}
    for r in range(3, 17):
        sec = g.cell(row=r, column=1).value
        if not sec or len(str(sec)) != 1:
            continue
        relevance[str(sec)] = {
            p: g.cell(row=r, column=3 + i).value
            for i, p in enumerate(protocols)}

    items = []
    for r in range(9, 130):
        lab = ws.cell(row=r, column=2).value
        if not lab:
            continue
        lab = str(lab)
        item_id = lab.split(".")[0].strip()
        title = lab.split(".", 1)[1].split("\n")[0].strip()
        src, provider = SOURCE.get(item_id, ("UNCLASSIFIED", None))
        desc = " ".join(str(ws.cell(row=r, column=5).value or "").split())
        items.append({
            "id": item_id,
            "section": item_id[0],
            "title": title,
            "description": desc,
            "quantitative": "QUANTITATIVE" in lab,
            "vocabulary": vocab.get(r, []),
            "input_row": r,
            "source": src,
            "provider": provider,
            "full_requires": FULL_REQUIRES.get(item_id, []),
            "aggregate": AGGREGATE.get(item_id),
        })

    unclassified = [i["id"] for i in items if i["source"] == "UNCLASSIFIED"]
    assert not unclassified, f"unclassified items: {unclassified}"

    return {
        "tool": "TMS-RAT",
        "tool_version": "1.0",
        "tool_date": "2025-06-02",
        "tool_url": "https://tms-rat.org/",
        "citation": ("Székely, O., Holmes, N.P., et al. Development and "
                     "Validation of the Transcranial Magnetic Stimulation "
                     "Reporting Assessment Tool (TMS-RAT). bioRxiv "
                     "2026.02.18.706460"),
        "mapping_version": "1",
        "_note": ("Item ids, titles, descriptions and vocabularies are read "
                  "from the official workbook and must not be edited here. "
                  "Only 'source' and 'provider' are this tool's own "
                  "classification."),
        "sections": {s: g.cell(row=r, column=2).value
                     for r, s in ((r, str(g.cell(row=r, column=1).value))
                                  for r in range(3, 17))
                     if s and len(s) == 1},
        "protocols": protocols,
        "section_relevance": relevance,
        "relevance_legend": {"✔": "required", "?": "sometimes", "✘": "not required"},
        "items": items,
    }

if __name__ == "__main__":
    if not os.path.exists(XLSX):
        sys.exit(f"Workbook not found: {XLSX}\n"
                 f"Download TMS-RAT v1.0 from https://tms-rat.org/ and save "
                 f"it beside this script.")
    m = build()
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1, ensure_ascii=False)
    from collections import Counter
    c = Counter(i["source"] for i in m["items"])
    print(f"wrote {os.path.normpath(OUT)}")
    print("items:", len(m["items"]))
    for k, v in c.most_common():
        print(f"   {k:<14} {v}")
