import React, { createContext, useContext, useEffect, useMemo, useState } from 'react';

export type ThemeMode = 'light' | 'dark';

interface ThemeContextValue {
  mode: ThemeMode;
  toggle: () => void;
  setMode: (m: ThemeMode) => void;
}

const ThemeContext = createContext<ThemeContextValue | undefined>(undefined);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [mode, setModeState] = useState<ThemeMode>('light');

  useEffect(() => {
    // Initialize from localStorage or OS preference
    const stored = (localStorage.getItem('xplainml-theme') as ThemeMode | null);
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const initial: ThemeMode = stored || (prefersDark ? 'dark' : 'light');
    setModeState(initial);
  }, []);

  useEffect(() => {
    const root = document.body; // apply class on body for our CSS
    if (mode === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('xplainml-theme', mode);
  }, [mode]);

  const setMode = (m: ThemeMode) => setModeState(m);
  const toggle = () => setModeState((prev) => (prev === 'light' ? 'dark' : 'light'));

  const value = useMemo(() => ({ mode, toggle, setMode }), [mode]);
  
  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used within ThemeProvider');
  return ctx;
}
