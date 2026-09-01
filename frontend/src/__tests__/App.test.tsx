import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { DemoExperience } from "../App";
import type { DemoData } from "../types";
import demoData from "../../public/data/demo.json";

const data = demoData as DemoData;

function renderedText() {
  return document.body.textContent ?? "";
}

describe("Fort product demo", () => {
  it("renders the hero performance metrics", () => {
    render(<DemoExperience data={data} />);

    expect(screen.getByRole("heading", { name: "Today's performance" })).toBeInTheDocument();
    expect(screen.getByText("97.8%")).toBeInTheDocument();
    expect(screen.getAllByText("43.3 kg").length).toBeGreaterThan(0);
  });

  it("labels actual and expected performance clearly", () => {
    render(<DemoExperience data={data} />);

    const metrics = screen.getByLabelText("Selected session metrics");
    expect(within(metrics).getByText("Actual handgrip")).toBeInTheDocument();
    expect(within(metrics).getByText("Expected")).toBeInTheDocument();
    expect(screen.getAllByText("Expected from prior personal history").length).toBeGreaterThan(0);
  });

  it("shows the limited-history state through product copy", () => {
    render(<DemoExperience data={data} />);

    expect(screen.getByText(/More personal history is needed/)).toBeInTheDocument();
    expect(screen.getByText(/Based on 11 prior valid handgrip observations/)).toBeInTheDocument();
  });

  it("renders context as descriptive with timing-unverified labels", () => {
    render(<DemoExperience data={data} />);

    expect(screen.getByRole("heading", { name: "Context recorded for this study observation" })).toBeInTheDocument();
    expect(screen.getByText(/Timing relative to performance was not available/)).toBeInTheDocument();
    expect(screen.getAllByText(/Timing not verified/).length).toBeGreaterThan(2);
    expect(renderedText()).toContain("Shown for reference, not used in the trusted prediction");
  });

  it("renders the model-card content", () => {
    render(<DemoExperience data={data} />);

    expect(screen.getByRole("heading", { name: "Methodology and data disclosure" })).toBeInTheDocument();
    expect(screen.getByText("Model and data limits")).toBeInTheDocument();
    expect(screen.getByText("Track A uses prior personal handgrip history only.")).toBeInTheDocument();
    expect(screen.getByText(/did not improve generalization/)).toBeInTheDocument();
    expect(screen.getByText("metadata insufficient")).toBeInTheDocument();
  });

  it("does not present prohibited pre-workout or causal copy", () => {
    render(<DemoExperience data={data} />);
    const text = renderedText().toLowerCase();

    expect(text).not.toContain("recovery score");
    expect(text).not.toContain("readiness score");
    expect(text).not.toContain("predicted workout score");
    expect(text).not.toContain("you are not ready");
    expect(text).not.toContain("you should");
    expect(text).not.toContain("sleep caused");
    expect(text).not.toContain("caffeine improved");
    expect(text).not.toContain("context predicts");
  });
});
