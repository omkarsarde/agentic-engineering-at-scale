import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputPath = path.resolve(process.argv[2] ?? "fixtures/cost-model.xlsx");
const previewDir = path.resolve(process.argv[3] ?? "workbook-preview");
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Summary");
const assumptions = workbook.worksheets.add("Assumptions");
const model = workbook.worksheets.add("Model");
const sensitivity = workbook.worksheets.add("Sensitivity");
const checks = workbook.worksheets.add("Checks");
const sources = workbook.worksheets.add("Sources");

const colors = {
  navy: "#17365D",
  teal: "#0F6B78",
  paleTeal: "#DDEBF7",
  input: "#FFF2CC",
  grid: "#D9E2F3",
  ink: "#1F2937",
  white: "#FFFFFF",
  green: "#E2F0D9",
  red: "#FCE4D6",
};

function title(sheet, range, text, subtitle) {
  range.merge();
  range.values = [[text]];
  range.format = {
    fill: colors.navy,
    font: { bold: true, color: colors.white, size: 18 },
    verticalAlignment: "center",
  };
  range.format.rowHeight = 32;
  const note = sheet.getRange("A2:K2");
  note.merge();
  note.values = [[subtitle]];
  note.format = { font: { color: "#596579", italic: true }, wrapText: true };
  note.format.rowHeight = 28;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(3);
}

function header(range) {
  range.format = {
    fill: colors.teal,
    font: { bold: true, color: colors.white },
    borders: { preset: "all", style: "thin", color: colors.grid },
    wrapText: true,
    verticalAlignment: "center",
  };
}

function tableBody(range) {
  range.format = {
    font: { color: colors.ink },
    borders: { preset: "all", style: "thin", color: colors.grid },
    verticalAlignment: "center",
  };
}

title(
  assumptions,
  assumptions.getRange("A1:F1"),
  "Training Cost Model — Assumptions",
  "Yellow cells are scenario inputs. The base price is illustrative; the dated cloud row is a reference, not a default.",
);
assumptions.getRange("A4:F4").values = [["Input", "Value", "Unit", "Treatment", "Evidence / note", "Checked"]];
header(assumptions.getRange("A4:F4"));
assumptions.getRange("A5:F13").values = [
  ["Parameters (N)", 30000000000, "parameters", "Editable", "Dense model parameter count", ""],
  ["Training tokens (D)", 600000000000, "tokens", "Editable", "Tokens presented to the optimizer", ""],
  ["Accelerators", 8192, "devices", "Editable", "Homogeneous training fleet", ""],
  ["Peak throughput", 1000, "TFLOP/s/device", "Editable", "Dense-equivalent peak used for the planning denominator", ""],
  ["Model FLOPs utilization", 0.4, "%", "Editable", "Measured/modelled model FLOPs divided by peak capacity", ""],
  ["Price", 4, "$/device-hour", "Illustrative", "Replace with an approved quote or internal transfer price", ""],
  ["Facility power", 1.2, "kW/device", "Editable", "Includes accelerator host and facility overhead", ""],
  ["GCP a3-highgpu-8g H100", 11.061250014875, "$/GPU-hour", "Dated reference", "$88.490000119 per 8-GPU VM; not used by the base case", "2026-07-19"],
  ["Compute estimate", 6, "FLOPs/(parameter·token)", "Mechanism", "Dense decoder-only training approximation C ≈ 6ND", "2026-07-19"],
];
tableBody(assumptions.getRange("A5:F13"));
assumptions.getRange("B5:B11").format.fill = colors.input;
assumptions.getRange("B5:B8").format.numberFormat = "#,##0";
assumptions.getRange("B9").format.numberFormat = "0%";
assumptions.getRange("B10:B12").format.numberFormat = "$#,##0.00";
assumptions.getRange("B13").format.numberFormat = "0";
assumptions.getRange("A4:F13").format.wrapText = true;
assumptions.getRange("A:A").format.columnWidth = 29;
assumptions.getRange("B:B").format.columnWidth = 18;
assumptions.getRange("C:C").format.columnWidth = 20;
assumptions.getRange("D:D").format.columnWidth = 18;
assumptions.getRange("E:E").format.columnWidth = 48;
assumptions.getRange("F:F").format.columnWidth = 14;

title(
  model,
  model.getRange("A1:D1"),
  "Formula Model",
  "Every output is formula-linked to Assumptions. Keep compute fixed when comparing MFU or price scenarios.",
);
model.getRange("A4:D4").values = [["Output", "Formula result", "Unit", "Interpretation"]];
header(model.getRange("A4:D4"));
model.getRange("A5:A12").values = [
  ["Training compute"],
  ["Accelerator-hours"],
  ["Wall time"],
  ["Training cost"],
  ["Average facility power"],
  ["Facility energy"],
  ["Cost per million tokens"],
  ["Training tokens per wall day"],
];
model.getRange("C5:D12").values = [
  ["FLOPs", "Dense estimate C = 6ND"],
  ["device-hours", "Compute divided by useful per-device throughput"],
  ["days", "Accelerator-hours divided by fleet size and 24"],
  ["USD", "Accelerator-hours multiplied by the selected rate"],
  ["MW", "Fleet size multiplied by facility kW per device"],
  ["MWh", "Accelerator-hours multiplied by facility kW per device"],
  ["$/M tokens", "Cost divided by training-token millions"],
  ["tokens/day", "Token budget divided by wall days"],
];
model.getRange("B5:B12").formulas = [
  ["='Assumptions'!$B$13*'Assumptions'!$B$5*'Assumptions'!$B$6"],
  ["=B5/('Assumptions'!$B$8*1000000000000*'Assumptions'!$B$9)/3600"],
  ["=B6/'Assumptions'!$B$7/24"],
  ["=B6*'Assumptions'!$B$10"],
  ["='Assumptions'!$B$7*'Assumptions'!$B$11/1000"],
  ["=B6*'Assumptions'!$B$11/1000"],
  ["=B8/('Assumptions'!$B$6/1000000)"],
  ["='Assumptions'!$B$6/B7"],
];
tableBody(model.getRange("A5:D12"));
model.getRange("B5:B12").format.fill = colors.paleTeal;
model.getRange("B5").format.numberFormat = "0.00E+00";
model.getRange("B6:B7").format.numberFormat = "#,##0.00";
model.getRange("B8").format.numberFormat = "$#,##0";
model.getRange("B9:B10").format.numberFormat = "#,##0.00";
model.getRange("B11").format.numberFormat = "$0.000";
model.getRange("B12").format.numberFormat = "0.00E+00";
model.getRange("A:A").format.columnWidth = 30;
model.getRange("B:B").format.columnWidth = 22;
model.getRange("C:C").format.columnWidth = 18;
model.getRange("D:D").format.columnWidth = 52;

title(
  sensitivity,
  sensitivity.getRange("A1:E1"),
  "MFU Sensitivity",
  "The workload, device count, peak throughput, rate, and power stay fixed; only model FLOPs utilization changes.",
);
sensitivity.getRange("A4:E4").values = [["MFU", "Accelerator-hours", "Wall days", "Training cost", "Energy (MWh)"]];
header(sensitivity.getRange("A4:E4"));
sensitivity.getRange("A5:A12").values = [[0.25], [0.3], [0.35], [0.4], [0.45], [0.5], [0.55], [0.6]];
sensitivity.getRange("B5:E5").formulas = [[
  "='Model'!$B$5/('Assumptions'!$B$8*1000000000000*A5)/3600",
  "=B5/'Assumptions'!$B$7/24",
  "=B5*'Assumptions'!$B$10",
  "=B5*'Assumptions'!$B$11/1000",
]];
sensitivity.getRange("B5:B12").fillDown();
sensitivity.getRange("C5:C12").fillDown();
sensitivity.getRange("D5:D12").fillDown();
sensitivity.getRange("E5:E12").fillDown();
tableBody(sensitivity.getRange("A5:E12"));
sensitivity.getRange("A5:A12").format.numberFormat = "0%";
sensitivity.getRange("B5:C12").format.numberFormat = "#,##0.00";
sensitivity.getRange("D5:D12").format.numberFormat = "$#,##0";
sensitivity.getRange("E5:E12").format.numberFormat = "#,##0.00";
sensitivity.getRange("A:E").format.columnWidth = 22;

title(
  summary,
  summary.getRange("A1:K1"),
  "Frontier Training Economics — Decision Summary",
  "Illustrative 30B-parameter / 600B-token dense run. Change yellow inputs on Assumptions; formulas and chart update automatically.",
);
summary.getRange("A4:C4").values = [["Decision KPI", "Value", "Unit"]];
header(summary.getRange("A4:C4"));
summary.getRange("A5:A10").values = [["Training compute"], ["Accelerator-hours"], ["Wall time"], ["Training cost"], ["Facility power"], ["Facility energy"]];
summary.getRange("C5:C10").values = [["FLOPs"], ["device-hours"], ["days"], ["USD"], ["MW"], ["MWh"]];
summary.getRange("B5:B10").formulas = [["='Model'!B5"], ["='Model'!B6"], ["='Model'!B7"], ["='Model'!B8"], ["='Model'!B9"], ["='Model'!B10"]];
tableBody(summary.getRange("A5:C10"));
summary.getRange("B5:B10").format.fill = colors.paleTeal;
summary.getRange("B5").format.numberFormat = "0.00E+00";
summary.getRange("B6:B7").format.numberFormat = "#,##0.00";
summary.getRange("B8").format.numberFormat = "$#,##0";
summary.getRange("B9:B10").format.numberFormat = "#,##0.00";
summary.getRange("E4:G4").values = [["Control", "Value", "Interpretation"]];
header(summary.getRange("E4:G4"));
summary.getRange("E5:E7").values = [["Model FLOPs utilization"], ["Price"], ["Accelerators"]];
summary.getRange("F5:F7").formulas = [["='Assumptions'!B9"], ["='Assumptions'!B10"], ["='Assumptions'!B7"]];
summary.getRange("G5:G7").values = [["Useful model FLOPs / peak FLOPs"], ["Illustrative $/device-hour"], ["Devices in the homogeneous fleet"]];
tableBody(summary.getRange("E5:G7"));
summary.getRange("F5").format.numberFormat = "0%";
summary.getRange("F6").format.numberFormat = "$0.00";
summary.getRange("F7").format.numberFormat = "#,##0";
summary.getRange("A14:B14").values = [["MFU", "Training cost"]];
summary.getRange("A15:B15").formulas = [["=TEXT('Sensitivity'!A5,\"0%\")", "='Sensitivity'!D5"]];
summary.getRange("A15:A22").fillDown();
summary.getRange("B15:B22").fillDown();
header(summary.getRange("A14:B14"));
tableBody(summary.getRange("A15:B22"));
summary.getRange("A15:A22").format.numberFormat = "0%";
summary.getRange("B15:B22").format.numberFormat = "$#,##0";
const chart = summary.charts.add("line", summary.getRange("A14:B22"));
chart.title = "Higher MFU lowers cost for fixed compute";
chart.hasLegend = false;
chart.xAxis = { axisType: "textAxis" };
chart.yAxis = { numberFormatCode: "$#,##0" };
chart.setPosition("D13", "K28");
summary.getRange("A:A").format.columnWidth = 28;
summary.getRange("B:B").format.columnWidth = 20;
summary.getRange("C:C").format.columnWidth = 17;
summary.getRange("D:D").format.columnWidth = 3;
summary.getRange("E:E").format.columnWidth = 29;
summary.getRange("F:F").format.columnWidth = 17;
summary.getRange("G:G").format.columnWidth = 38;

title(
  checks,
  checks.getRange("A1:C1"),
  "Model Checks",
  "These formula checks catch invalid scenario inputs and non-positive outputs before a cost number is used in a decision.",
);
checks.getRange("A4:C4").values = [["Check", "Status", "Reason"]];
header(checks.getRange("A4:C4"));
checks.getRange("A5:A10").values = [["Positive model size"], ["Positive token budget"], ["MFU is in (0, 1]"], ["Positive integer fleet"], ["Positive throughput and price"], ["Positive wall time and cost"]];
checks.getRange("B5:B10").formulas = [
  ["=IF('Assumptions'!B5>0,\"PASS\",\"FAIL\")"],
  ["=IF('Assumptions'!B6>0,\"PASS\",\"FAIL\")"],
  ["=IF(AND('Assumptions'!B9>0,'Assumptions'!B9<=1),\"PASS\",\"FAIL\")"],
  ["=IF(AND('Assumptions'!B7>0,'Assumptions'!B7=INT('Assumptions'!B7)),\"PASS\",\"FAIL\")"],
  ["=IF(AND('Assumptions'!B8>0,'Assumptions'!B10>0),\"PASS\",\"FAIL\")"],
  ["=IF(AND('Model'!B7>0,'Model'!B8>0),\"PASS\",\"FAIL\")"],
];
checks.getRange("C5:C10").values = [["N must be greater than zero"], ["D must be greater than zero"], ["Utilization is a fraction, not a percentage point value"], ["Fleet count must be a positive whole number"], ["Denominator and rate must be positive"], ["Final decision outputs must be positive"]];
checks.getRange("A12").values = [["MODEL STATUS"]];
checks.getRange("B12").formulas = [["=IF(COUNTIF(B5:B10,\"FAIL\")=0,\"PASS\",\"FAIL\")"]];
tableBody(checks.getRange("A5:C10"));
header(checks.getRange("A12"));
checks.getRange("B12").format = { font: { bold: true }, borders: { preset: "all", style: "thin", color: colors.grid } };
checks.getRange("B5:B10").conditionalFormats.add("cellIs", { operator: "equal", formula: '"PASS"', format: { fill: colors.green, font: { bold: true, color: "#375623" } } });
checks.getRange("B5:B10").conditionalFormats.add("cellIs", { operator: "equal", formula: '"FAIL"', format: { fill: colors.red, font: { bold: true, color: "#9C0006" } } });
checks.getRange("A:A").format.columnWidth = 30;
checks.getRange("B:B").format.columnWidth = 14;
checks.getRange("C:C").format.columnWidth = 52;

title(
  sources,
  sources.getRange("A1:E1"),
  "Sources and Scope",
  "Sources establish a dated input or a mechanism. They do not convert illustrative assumptions into vendor quotes.",
);
sources.getRange("A4:E4").values = [["Item", "Classification", "Claim used", "URL / provenance", "Checked"]];
header(sources.getRange("A4:E4"));
sources.getRange("A5:E7").values = [
  ["Dense training compute", "Mechanism", "C ≈ 6ND planning approximation", "Chapter 6 derivation and cited scaling-law literature", "2026-07-19"],
  ["Base price", "Illustrative assumption", "$4/device-hour", "Replace with approved quote or internal transfer price", ""],
  ["Google Cloud a3-highgpu-8g", "Dated vendor price", "$88.490000119/VM-hour for 8 H100 GPUs", "https://cloud.google.com/products/compute/pricing/accelerator-optimized", "2026-07-19"],
];
tableBody(sources.getRange("A5:E7"));
sources.getRange("A4:E7").format.wrapText = true;
sources.getRange("A:A").format.columnWidth = 30;
sources.getRange("B:B").format.columnWidth = 24;
sources.getRange("C:C").format.columnWidth = 42;
sources.getRange("D:D").format.columnWidth = 68;
sources.getRange("E:E").format.columnWidth = 14;

for (const sheet of [summary, assumptions, model, sensitivity, checks, sources]) {
  sheet.getUsedRange().format.autofitRows();
}

const keyInspection = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:G22",
  include: "values,formulas",
  tableMaxRows: 22,
  tableMaxCols: 7,
  maxChars: 7000,
});
const formulaInspection = await workbook.inspect({
  kind: "formula",
  sheetId: "Sensitivity",
  range: "A4:E12",
  options: { maxResults: 80 },
  maxChars: 5000,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});
await fs.writeFile(
  path.join(previewDir, "inspection.ndjson"),
  `${keyInspection.ndjson}\n${formulaInspection.ndjson}\n${errors.ndjson}\n`,
  "utf8",
);

for (const sheetName of ["Summary", "Assumptions", "Model", "Sensitivity", "Checks", "Sources"]) {
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(
    path.join(previewDir, `${sheetName.toLowerCase()}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });
console.log(JSON.stringify({ outputPath, previewDir }, null, 2));
