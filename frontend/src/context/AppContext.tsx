import React, { createContext, useContext, useReducer, ReactNode } from 'react';

// Types
interface DataInfo {
  fileName: string;
  shape: [number, number];
  columns: string[];
  targetColumn: string;
  taskType: 'classification' | 'regression';
  preview: any[];
}

interface TrainedModel {
  name: string;
  displayName: string;
  metrics: any;
  parameters: any;
  trainingTime: number;
}

interface Explanation {
  method: string;
  type: 'global' | 'local';
  model_name: string;
  data: any;
}

interface AppState {
  data: DataInfo | null;
  models: TrainedModel[];
  explanations: Explanation[];
  isLoading: boolean;
  error: string | null;
}

// Initial State
const initialState: AppState = {
  data: null,
  models: [],
  explanations: [],
  isLoading: false,
  error: null,
};

// Actions
type AppAction = 
  | { type: 'SET_DATA'; payload: DataInfo }
  | { type: 'ADD_MODEL'; payload: TrainedModel }
  | { type: 'ADD_EXPLANATION'; payload: Explanation }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'RESET_STATE' };

// Reducer
function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case 'SET_DATA':
      return {
        ...state,
        data: action.payload,
        error: null,
      };
    case 'ADD_MODEL':
      return {
        ...state,
        models: [...state.models, action.payload],
        error: null,
      };
    case 'ADD_EXPLANATION':
      return {
        ...state,
        explanations: [...state.explanations, action.payload],
        error: null,
      };
    case 'SET_LOADING':
      return {
        ...state,
        isLoading: action.payload,
      };
    case 'SET_ERROR':
      return {
        ...state,
        error: action.payload,
        isLoading: false,
      };
    case 'RESET_STATE':
      return initialState;
    default:
      return state;
  }
}

// Context
interface AppContextType {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

// Provider
export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

// Hook
export function useAppContext() {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error('useAppContext must be used within an AppProvider');
  }
  return context;
}

export default AppContext;