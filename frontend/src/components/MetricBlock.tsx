type MetricBlockProps = {
  label: string;
  value: string;
  detail?: string;
  tone?: "primary" | "neutral" | "positive" | "negative";
};

export function MetricBlock({ label, value, detail, tone = "neutral" }: MetricBlockProps) {
  return (
    <div className={`metric-block metric-block--${tone}`}>
      <dt>{label}</dt>
      <dd>{value}</dd>
      {detail ? <p>{detail}</p> : null}
    </div>
  );
}
