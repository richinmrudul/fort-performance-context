export type HistoryState =
  | "no_prior_observations"
  | "limited_history"
  | "expected_available";

export type ContextItem = {
  available: boolean;
  display_state: "shown_descriptively" | "not_available";
  group: string;
  interpretation: string;
  key: string;
  label: string;
  source_column: string;
  timing_label: string;
  timing_status: string;
  unit: string | null;
  value: number | string | null;
};

export type TrendObservation = {
  actual_kg: number | null;
  difference_kg: number | null;
  expected_kg: number | null;
  history_state: HistoryState;
  is_selected: boolean;
  label: string;
  prior_observation_count: number;
  sequence: number;
};

export type DemoData = {
  athlete_index: Array<{
    athlete_key: string;
    eligible_session_count: number;
    label: string;
  }>;
  context: {
    headline: string;
    items: ContextItem[];
    prediction_use: string;
    timing_summary: string;
  };
  missing_states: Record<string, string>;
  model_card: {
    expected_performance_definition: string;
    future_data_requirements: string[];
    row_order_policy: string;
    timing_audit_decision: string;
    track_a_chronological_athlete_count: number;
    track_a_chronological_mae_kg: number;
    track_a_chronological_sample_count: number;
    track_a_summary: string;
    track_b_best_contextual_delta_mae_kg_vs_track_a: number;
    track_b_summary: string;
  };
  selected_session: {
    actual_kg: number | null;
    actual_percent_of_expected: number | null;
    athlete_key: string;
    athlete_label: string;
    comparison_direction: "above" | "below" | "at";
    date_label: string | null;
    date_state: string;
    difference_kg: number | null;
    difference_percent: number | null;
    expected_label: string;
    history_state: HistoryState;
    prior_observation_count: number;
    session_key: string;
    session_label: string;
    track_a_expected_kg: number | null;
    weekday_label: string | null;
    x_axis_label: string;
  };
  trend: {
    athlete_key: string;
    axis_policy: string;
    observations: TrendObservation[];
  };
};
