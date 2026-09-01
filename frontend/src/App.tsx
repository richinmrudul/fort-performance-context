import { useEffect, useMemo, useState } from "react";
import { MetricBlock } from "./components/MetricBlock";
import { TrendChart } from "./components/TrendChart";
import type { ContextItem, DemoData, TrendObservation } from "./types";
import "./styles/app.css";

function formatKg(value: number | null) {
  return value === null ? "Not available" : `${value.toFixed(value % 1 === 0 ? 0 : 1)} kg`;
}

function formatPercent(value: number | null) {
  return value === null ? "Expected value not available" : `${value.toFixed(1)}%`;
}

function signedKg(value: number | null) {
  if (value === null) return "Expected value not available";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)} kg`;
}

function signedPercent(value: number | null) {
  if (value === null) return "Expected value not available";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(1)}%`;
}

function valueWithUnit(item: ContextItem) {
  if (!item.available || item.value === null) return "Not available";
  if (item.unit === "/10") return `${item.value}/10`;
  if (item.unit === "fraction" && typeof item.value === "number") return `${Math.round(item.value * 100)}%`;
  return item.unit ? `${item.value} ${item.unit}` : String(item.value);
}

function comparisonCopy(direction: DemoData["selected_session"]["comparison_direction"]) {
  if (direction === "above") return "above expectation";
  if (direction === "below") return "below expectation";
  return "at expectation";
}

function compactTrendCopy(rows: TrendObservation[]) {
  const recent = rows.slice(-4).map((row) => `${row.actual_kg ?? "NA"} kg`);
  return recent.join(" -> ");
}

export function DemoExperience({ data }: { data: DemoData }) {
  const session = data.selected_session;
  const directionTone = session.comparison_direction === "below" ? "negative" : "positive";
  const missingContext = useMemo(
    () => data.context.items.find((item) => !item.available) ?? null,
    [data.context.items],
  );

  return (
    <main className="app-shell">
      <a className="skip-link" href="#performance-trend">
        Skip to trend
      </a>
      <header className="topbar" aria-label="Demo status">
        <div>
          <p className="eyebrow">Fort Performance Context</p>
          <h1>Post-workout performance review</h1>
        </div>
        <div className="status-pill" aria-label="Read-only static demo">
          Read-only demo
        </div>
      </header>

      <section className="hero-section" aria-labelledby="overview-title">
        <div className="hero-copy">
          <p className="eyebrow">{session.athlete_label}</p>
          <h2 id="overview-title">Today's performance</h2>
          <p className="hero-percent">{formatPercent(session.actual_percent_of_expected)}</p>
          <p className="hero-subtitle">of expected performance from prior personal history</p>
        </div>
        <dl className="metric-grid" aria-label="Selected session metrics">
          <MetricBlock label="Actual handgrip" value={formatKg(session.actual_kg)} tone="primary" />
          <MetricBlock label="Expected" value={formatKg(session.track_a_expected_kg)} detail={session.expected_label} />
          <MetricBlock
            label="Difference"
            value={signedKg(session.difference_kg)}
            detail={`${signedPercent(session.difference_percent)} ${comparisonCopy(session.comparison_direction)}`}
            tone={directionTone}
          />
          <MetricBlock
            label="Prior observations"
            value={String(session.prior_observation_count)}
            detail="Evaluated before this observation"
          />
        </dl>
      </section>

      <div className="content-grid">
        <section className="panel session-panel" aria-labelledby="session-title">
          <div className="section-heading">
            <p className="eyebrow">Session detail</p>
            <h2 id="session-title">{session.session_label}</h2>
          </div>
          <div className="comparison-row">
            <span>Actual</span>
            <strong>{formatKg(session.actual_kg)}</strong>
          </div>
          <div className="comparison-row">
            <span>{session.expected_label}</span>
            <strong>{formatKg(session.track_a_expected_kg)}</strong>
          </div>
          <div className="comparison-row comparison-row--accent">
            <span>Observed difference</span>
            <strong>
              {signedKg(session.difference_kg)} {comparisonCopy(session.comparison_direction)}
            </strong>
          </div>
          <p className="calm-note">
            Based on {session.prior_observation_count} prior valid handgrip observations. More personal history is
            needed before early observations receive an expected value.
          </p>
          <p className="calm-note">
            {data.missing_states.no_defensible_date}; labels use observation order rather than calendar dates.
          </p>
        </section>

        <section className="panel trend-panel" id="performance-trend" aria-labelledby="trend-title-heading">
          <div className="section-heading">
            <p className="eyebrow">Recent personal trend</p>
            <h2 id="trend-title-heading">Actual versus expected</h2>
          </div>
          <TrendChart observations={data.trend.observations} />
          <div className="trend-summary" aria-label="Recent actual observations">
            <span>Recent actuals</span>
            <strong>{compactTrendCopy(data.trend.observations)}</strong>
          </div>
        </section>

        <section className="panel context-panel" aria-labelledby="context-title">
          <div className="section-heading">
            <p className="eyebrow">Session context</p>
            <h2 id="context-title">{data.context.headline}</h2>
          </div>
          <p className="calm-note">{data.context.timing_summary}. {data.context.prediction_use}.</p>
          <div className="context-list">
            {data.context.items.map((item) => (
              <article className="context-item" key={item.key}>
                <div>
                  <h3>{item.label}</h3>
                  <p>{item.timing_label}</p>
                </div>
                <strong>{valueWithUnit(item)}</strong>
              </article>
            ))}
          </div>
          {missingContext ? <p className="calm-note">{data.missing_states.missing_context}</p> : null}
        </section>

        <section className="panel model-panel" aria-labelledby="model-title">
          <div className="section-heading">
            <p className="eyebrow">Transparency</p>
            <h2 id="model-title">Model and data limits</h2>
          </div>
          <ul className="model-list">
            <li>{data.model_card.track_a_summary}</li>
            <li>{data.model_card.track_b_summary}</li>
            <li>REST metadata provided no defensible prediction cutoff.</li>
            <li>Context is descriptive only in this prototype.</li>
          </ul>
          <dl className="method-grid">
            <div>
              <dt>Track A MAE</dt>
              <dd>{data.model_card.track_a_chronological_mae_kg.toFixed(3)} kg</dd>
            </div>
            <div>
              <dt>Evaluated rows</dt>
              <dd>{data.model_card.track_a_chronological_sample_count}</dd>
            </div>
            <div>
              <dt>Timing audit</dt>
              <dd>{data.model_card.timing_audit_decision.replace(/_/g, " ")}</dd>
            </div>
          </dl>
          <h3>Production data needed</h3>
          <ul className="requirements-list">
            {data.model_card.future_data_requirements.map((requirement) => (
              <li key={requirement}>{requirement}</li>
            ))}
          </ul>
        </section>
      </div>
    </main>
  );
}

export default function App() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/data/demo.json")
      .then((response) => {
        if (!response.ok) throw new Error(`Demo data request failed: ${response.status}`);
        return response.json() as Promise<DemoData>;
      })
      .then(setData)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Unable to load demo data"));
  }, []);

  if (error) {
    return (
      <main className="app-shell">
        <section className="panel">
          <h1>Demo data unavailable</h1>
          <p>{error}</p>
        </section>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="app-shell">
        <section className="panel loading-panel" aria-live="polite">
          Loading existing repository outputs...
        </section>
      </main>
    );
  }

  return <DemoExperience data={data} />;
}
