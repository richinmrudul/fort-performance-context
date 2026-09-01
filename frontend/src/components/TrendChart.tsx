import type { TrendObservation } from "../types";

type TrendChartProps = {
  observations: TrendObservation[];
};

const width = 720;
const height = 260;
const padding = { top: 26, right: 28, bottom: 42, left: 48 };

function pathFor(points: Array<{ x: number; y: number }>) {
  return points.map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`).join(" ");
}

export function TrendChart({ observations }: TrendChartProps) {
  const actualRows = observations.filter((row) => row.actual_kg !== null);
  const expectedRows = observations.filter((row) => row.expected_kg !== null);
  const values = [
    ...actualRows.map((row) => row.actual_kg as number),
    ...expectedRows.map((row) => row.expected_kg as number),
  ];
  const min = Math.floor(Math.min(...values) - 1);
  const max = Math.ceil(Math.max(...values) + 1);
  const xSpan = Math.max(1, observations.length - 1);
  const ySpan = Math.max(1, max - min);
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const point = (row: TrendObservation, value: number) => ({
    x: padding.left + ((row.sequence - 1) / xSpan) * chartWidth,
    y: padding.top + (1 - (value - min) / ySpan) * chartHeight,
  });

  const actualPoints = actualRows.map((row) => point(row, row.actual_kg as number));
  const expectedPoints = expectedRows.map((row) => point(row, row.expected_kg as number));
  const selected = observations.find((row) => row.is_selected);

  return (
    <figure className="trend-card">
      <svg
        className="trend-chart"
        role="img"
        aria-labelledby="trend-title trend-desc"
        viewBox={`0 0 ${width} ${height}`}
      >
        <title id="trend-title">Recent handgrip performance trend</title>
        <desc id="trend-desc">
          Actual handgrip observations are shown against expected values from prior personal history when available.
        </desc>
        <line x1={padding.left} x2={width - padding.right} y1={height - padding.bottom} y2={height - padding.bottom} />
        <line x1={padding.left} x2={padding.left} y1={padding.top} y2={height - padding.bottom} />
        {[min, Math.round((min + max) / 2), max].map((tick) => {
          const y = padding.top + (1 - (tick - min) / ySpan) * chartHeight;
          return (
            <g key={tick} className="trend-chart__tick">
              <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} />
              <text x={padding.left - 10} y={y + 4}>
                {tick}
              </text>
            </g>
          );
        })}
        <path className="trend-chart__expected" d={pathFor(expectedPoints)} />
        <path className="trend-chart__actual" d={pathFor(actualPoints)} />
        {actualRows.map((row) => {
          const coords = point(row, row.actual_kg as number);
          return (
            <circle
              key={row.sequence}
              className={row.is_selected ? "trend-chart__dot trend-chart__dot--selected" : "trend-chart__dot"}
              cx={coords.x}
              cy={coords.y}
              r={row.is_selected ? 6 : 4}
            />
          );
        })}
        <text className="trend-chart__axis-label" x={width / 2} y={height - 8}>
          Observation
        </text>
        <text className="trend-chart__axis-label" transform="translate(14 145) rotate(-90)">
          kg
        </text>
      </svg>
      <figcaption>
        {selected
          ? `${selected.label}: ${selected.actual_kg} kg actual; ${
              selected.expected_kg ?? "expected unavailable"
            } kg expected.`
          : "Observation sequence shown because no defensible dates are available."}
      </figcaption>
    </figure>
  );
}
