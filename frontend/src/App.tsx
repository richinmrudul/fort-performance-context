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

function observationSequenceCopy(count: number) {
  return count === 1 ? "1 observation in sequence" : `${count} observations in sequence`;
}

function compactTrendCopy(rows: TrendObservation[]) {
  const recent = rows.slice(-4).map((row) => `${row.actual_kg ?? "NA"} kg`);
  return recent.join(" -> ");
}

export function DemoExperience({ data }: { data: DemoData }) {
  const session = data.selected_session;
  const directionTone = session.comparison_direction === "below" ? "neutral" : "positive";
  const [methodOpen, setMethodOpen] = useState(false);
  const missingContext = useMemo(
    () => data.context.items.find((item) => !item.available) ?? null,
    [data.context.items],
  );

  return (
    <main className="app-shell">
      <a className="skip-link" href="#performance-trend">
        Skip to trend
      </a>
      <header className="topbar" aria-label="Product demo status">
        <a className="brand-mark" href="#overview-title" aria-label="Fort Performance Context home">
          Fort Performance Context
        </a>
        <nav className="topbar__nav" aria-label="Demo sections">
          <a href="#performance-trend">Trend</a>
          <a href="#context-title">Context</a>
          <a href="#methodology-title">Methodology</a>
        </nav>
        <span className="status-pill" aria-label="Read-only static demo">
          Read-only demo
        </span>
      </header>

      <section className="hero-section" aria-labelledby="overview-title">
        <div className="hero-copy">
          <p className="eyebrow">
            {session.athlete_label} / {session.session_label}
          </p>
          <h2 id="overview-title">Today's performance</h2>
          <p className="hero-subtitle">
            A post-workout handgrip result compared with the athlete's recent personal history.
          </p>
        </div>
        <div className="hero-result" aria-label="Selected session result">
          <p className="hero-percent">{formatPercent(session.actual_percent_of_expected)}</p>
          <p>of expectation</p>
        </div>
      </section>

      <section className="metrics-section" aria-labelledby="session-title">
        <div className="section-heading section-heading--wide">
          <p className="eyebrow">Actual versus expected</p>
          <h2 id="session-title">Neutral post-workout result</h2>
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
        <p className="section-note">
          Based on {session.prior_observation_count} prior valid handgrip observations. More personal history is needed
          before early observations receive an expected value.
        </p>
      </section>

      <div className="content-stack">
        <section className="trend-panel" id="performance-trend" aria-labelledby="trend-title-heading">
          <div className="section-heading section-heading--split">
            <div>
              <p className="eyebrow">Personal performance trend</p>
              <h2 id="trend-title-heading">Actual versus expected over observation order</h2>
            </div>
            <p>{observationSequenceCopy(data.trend.observations.length)}</p>
          </div>
          <TrendChart observations={data.trend.observations} />
          <div className="trend-summary" aria-label="Recent actual observations">
            <span>Recent actuals</span>
            <strong>{compactTrendCopy(data.trend.observations)}</strong>
          </div>
          <p className="section-note">
            {data.missing_states.no_defensible_date}; labels use observation order rather than calendar dates.
          </p>
        </section>

        <section className="context-panel" aria-labelledby="context-title">
          <div className="section-heading section-heading--split">
            <div>
              <p className="eyebrow">Descriptive session context</p>
              <h2 id="context-title">{data.context.headline}</h2>
            </div>
            <p>{data.context.timing_summary}</p>
          </div>
          <p className="section-note context-note">{data.context.prediction_use}.</p>
          <div className="context-list">
            {data.context.items.map((item) => (
              <article className="context-item" key={item.key}>
                <div>
                  <h3>{item.label}</h3>
                  <p>
                    {item.timing_label}. {item.interpretation}.
                  </p>
                </div>
                <strong>{valueWithUnit(item)}</strong>
              </article>
            ))}
          </div>
          {missingContext ? <p className="section-note">{data.missing_states.missing_context}</p> : null}
        </section>

        <section className="model-panel" aria-labelledby="model-title">
          <p className="eyebrow">Transparency</p>
          <h2 id="model-title">Methodology and data disclosure</h2>
          <details className="method-disclosure" open={methodOpen}>
            <summary
              id="methodology-title"
              aria-expanded={methodOpen}
              onClick={(event) => {
                event.preventDefault();
                setMethodOpen((isOpen) => !isOpen);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  setMethodOpen((isOpen) => !isOpen);
                }
              }}
            >
              Model and data limits
            </summary>
            <div className="method-body">
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
            </div>
          </details>
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
        <section className="fallback-panel">
          <h1>Demo data unavailable</h1>
          <p>{error}</p>
        </section>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="app-shell">
        <section className="fallback-panel loading-panel" aria-live="polite">
          Loading existing repository outputs...
        </section>
      </main>
    );
  }

  return <DemoExperience data={data} />;
}
