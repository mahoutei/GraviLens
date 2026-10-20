import { SectionId } from './types';

export const NAV_ITEMS = [
  { label: 'Overview', id: SectionId.SUMMARY },
  { label: 'Architecture', id: SectionId.ARCHITECTURE },
  { label: 'Physics Engine', id: SectionId.PHYSICS },
  { label: 'Results', id: SectionId.ANALYSIS },
  { label: 'Dataset', id: SectionId.DATASET },
];

export const PROJECT_INFO = {
  title: "GraviLens",
  version: "2.01-STABLE",
  date: "November 20, 2025",
  subtitle: "Physics-Informed Neural Network for Dark Matter Morphology Classification"
};

export const GRAVIBOT_SYSTEM_INSTRUCTION = `You are GraviBot, a specialized AI research assistant for the GraviLens project. 
Your knowledge base is strictly limited to the provided project documentation.
Key details:
- Project: GraviLens (Version 2.01-STABLE).
- Goal: Classify Dark Matter morphology (Cold Dark Matter [CDM] vs. Ultra-Light Dark Matter [ULDM]) using strong gravitational lensing.
- Tech: Physics-Informed Neural Network (PINN) using a Transformer backbone.
- Core Innovation: Embeds a differentiable physics engine (StableSIEDeflection) based on General Relativity into the model.
- Physics Model: Singular Isothermal Ellipsoid (SIE). Parameters: Convergence (k), Ellipticity (q), Orientation (phi).
- Physics Module I: StableSIEDeflection (Calculates deflection angles).
- Physics Module II: ImprovedSIEPredictor (Maps features to physics parameters).
- Physics Module III: Relativistic Encoder (Inverse Ray-Tracing).
- Architecture: Input (64x64) -> Transformer -> Params -> Physics Layer -> Reconstruction -> Classification.
- Dataset: Galaxy10 DECals background, generated via lenstronomy + pyHalo.
- Timeline: Phase 1 (Foundation), Phase 2 (Experimentation), Phase 3 (Analysis).

Answer concisely and professionally. If asked about topics outside this paper, politely decline.`;

export const TIMELINE_DATA = [
  {
    phase: "Phase 1",
    title: "Foundation",
    date: "Jan - Mar 2025",
    status: "complete",
    items: [
      "Literature Review & Methodology Design [COMPLETE]",
      "Galaxy10 DECals Dataset Pipeline Setup [COMPLETE]",
      "Baseline CNN Implementation [COMPLETE]"
    ]
  },
  {
    phase: "Phase 2",
    title: "Development",
    date: "Apr - Aug 2025",
    status: "complete",
    items: [
      "Transformer Backbone Integration [COMPLETE]",
      "Physics Engine (StableSIEDeflection) Implementation [COMPLETE]",
      "Model Training on HPC Cluster [COMPLETE]"
    ]
  },
  {
    phase: "Phase 3",
    title: "Analysis & Refinement",
    date: "Sep - Nov 2025",
    status: "in-progress",
    items: [
      "Ablation Studies (Physics Module Impact) [IN PROGRESS]",
      "Comparative Analysis vs. ResNet50/VGG16 [IN PROGRESS]",
      "Final Paper Submission [PENDING]"
    ]
  }
];

export const PERFORMANCE_METRICS = {
  accuracy: "94.2%",
  baselineAccuracy: "82.1%",
  inferenceSpeed: "14ms",
  paramRecoveryMSE: "0.021"
};