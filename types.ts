export enum SectionId {
  HOME = 'home',
  SUMMARY = 'summary',
  PHYSICS = 'physics',
  ARCHITECTURE = 'architecture',
  ANALYSIS = 'analysis',
  DATASET = 'dataset'
}

export interface NavItem {
  label: string;
  id: SectionId;
}

export interface Message {
  role: 'user' | 'model' | 'system';
  content: string;
  timestamp: number;
}