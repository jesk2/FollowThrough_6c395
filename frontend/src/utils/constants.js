export const DEVICE_LABELS = [
  "Salience Nudge",
  "Implementation Intention",
  "Planning Correction",
  "Virtual Stakes",
  "Precommitment Lock",
];

export const DEVICE_DESCRIPTIONS = [
  "We'll send you a reminder 2 hours before your task.",
  "We'll ask you to plan the when, where, and first step of each task.",
  "Based on your history, we'll adjust your time estimates automatically.",
  "Missing tasks costs streak points and triggers a loss-framed notification.",
  "You'll need to confirm before rescheduling within 24 hours.",
];

export const CATEGORY_COLORS = {
  academic: "#6366f1",
  exercise: "#22c55e",
  work: "#f59e0b",
  personal: "#ec4899",
};

export const FAILURE_REASONS = [
  { value: "ran_out_of_time", label: "Ran out of time" },
  { value: "forgot", label: "Forgot" },
  { value: "chose_not_to", label: "Chose not to" },
  { value: "external_blocker", label: "External blocker" },
];
