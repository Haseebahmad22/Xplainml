import React from 'react';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  color?: 'blue' | 'purple' | 'green' | 'red';
}

export function LoadingSpinner({ size = 'md', color = 'blue' }: LoadingSpinnerProps) {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-8 h-8',
    lg: 'w-12 h-12'
  };

  const colorClasses = {
    blue: 'text-blue-600',
    purple: 'text-purple-600',
    green: 'text-green-600',
    red: 'text-red-600'
  };

  return (
    <div className={`inline-block ${sizeClasses[size]} ${colorClasses[color]} animate-spin`}>
      <svg className="w-full h-full" fill="none" viewBox="0 0 24 24">
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
    </div>
  );
}

interface ButtonProps {
  children: React.ReactNode;
  variant?: 'primary' | 'secondary' | 'danger' | 'success';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
  className?: string;
}

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  onClick,
  type = 'button',
  className = ''
}: ButtonProps) {
  const baseClasses = 'inline-flex items-center justify-center font-semibold rounded-lg transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variantClasses = {
    primary: 'bg-gradient-to-r from-blue-500 to-purple-600 text-white hover:from-blue-600 hover:to-purple-700 focus:ring-blue-500 shadow-lg hover:shadow-xl',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300 focus:ring-gray-500',
    danger: 'bg-gradient-to-r from-red-500 to-red-600 text-white hover:from-red-600 hover:to-red-700 focus:ring-red-500 shadow-lg hover:shadow-xl',
    success: 'bg-gradient-to-r from-green-500 to-green-600 text-white hover:from-green-600 hover:to-green-700 focus:ring-green-500 shadow-lg hover:shadow-xl'
  };

  const sizeClasses = {
    sm: 'px-3 py-2 text-sm',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base'
  };

  const isDisabled = disabled || loading;

  return (
    <button
      type={type}
      onClick={onClick}
      disabled={isDisabled}
      className={`
        ${baseClasses}
        ${variantClasses[variant]}
        ${sizeClasses[size]}
        ${isDisabled ? 'opacity-50 cursor-not-allowed' : 'transform hover:-translate-y-0.5'}
        ${className}
      `}
    >
      {loading && <LoadingSpinner size="sm" color="blue" />}
      <span className={loading ? 'ml-2' : ''}>{children}</span>
    </button>
  );
}

interface CardProps {
  children: React.ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hover?: boolean;
}

export function Card({ children, className = '', padding = 'md', hover = true }: CardProps) {
  const paddingClasses = {
    none: '',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8'
  };

  return (
    <div
      className={`
        bg-white rounded-xl shadow-lg border border-gray-100
        ${hover ? 'hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1' : ''}
        ${paddingClasses[padding]}
        ${className}
      `}
    >
      {children}
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  color?: 'blue' | 'purple' | 'green' | 'red' | 'yellow';
  change?: {
    value: number;
    type: 'increase' | 'decrease';
  };
}

export function MetricCard({ title, value, icon, color = 'blue', change }: MetricCardProps) {
  const colorClasses = {
    blue: 'bg-gradient-to-br from-blue-500 to-blue-600',
    purple: 'bg-gradient-to-br from-purple-500 to-purple-600',
    green: 'bg-gradient-to-br from-green-500 to-green-600',
    red: 'bg-gradient-to-br from-red-500 to-red-600',
    yellow: 'bg-gradient-to-br from-yellow-500 to-yellow-600'
  };

  return (
    <div className={`${colorClasses[color]} text-white rounded-xl p-6 shadow-lg`}>
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium opacity-90">{title}</p>
          <p className="text-3xl font-bold mt-2">{value}</p>
          {change && (
            <div className="flex items-center mt-2">
              <span className={`text-sm ${change.type === 'increase' ? 'text-green-200' : 'text-red-200'}`}>
                {change.type === 'increase' ? '↗' : '↘'} {Math.abs(change.value)}%
              </span>
            </div>
          )}
        </div>
        <div className="text-3xl opacity-80">
          {icon}
        </div>
      </div>
    </div>
  );
}

interface InputProps {
  label?: string;
  error?: string;
  help?: string;
  required?: boolean;
  type?: string;
  placeholder?: string;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  className?: string;
}

export function Input({
  label,
  error,
  help,
  required = false,
  type = 'text',
  placeholder,
  value,
  onChange,
  disabled = false,
  className = ''
}: InputProps) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        disabled={disabled}
        className={`
          w-full px-4 py-3 border rounded-lg transition-all duration-200
          focus:ring-2 focus:ring-blue-500 focus:border-blue-500 focus:outline-none
          ${error ? 'border-red-300 focus:ring-red-500 focus:border-red-500' : 'border-gray-300'}
          ${disabled ? 'bg-gray-50 cursor-not-allowed' : 'bg-white'}
        `}
      />
      {help && !error && (
        <p className="mt-2 text-sm text-gray-500">{help}</p>
      )}
      {error && (
        <p className="mt-2 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
}

interface SelectProps {
  label?: string;
  error?: string;
  help?: string;
  required?: boolean;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  className?: string;
  children: React.ReactNode;
}

export function Select({
  label,
  error,
  help,
  required = false,
  value,
  onChange,
  disabled = false,
  className = '',
  children
}: SelectProps) {
  return (
    <div className={className}>
      {label && (
        <label className="block text-sm font-medium text-gray-700 mb-2">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <select
        value={value}
        onChange={(e) => onChange?.(e.target.value)}
        disabled={disabled}
        className={`
          w-full px-4 py-3 border rounded-lg transition-all duration-200
          focus:ring-2 focus:ring-blue-500 focus:border-blue-500 focus:outline-none
          ${error ? 'border-red-300 focus:ring-red-500 focus:border-red-500' : 'border-gray-300'}
          ${disabled ? 'bg-gray-50 cursor-not-allowed' : 'bg-white'}
        `}
      >
        {children}
      </select>
      {help && !error && (
        <p className="mt-2 text-sm text-gray-500">{help}</p>
      )}
      {error && (
        <p className="mt-2 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
}

interface AlertProps {
  children: React.ReactNode;
  variant?: 'info' | 'success' | 'warning' | 'error';
  onClose?: () => void;
  className?: string;
}

export function Alert({ children, variant = 'info', onClose, className = '' }: AlertProps) {
  const variantClasses = {
    info: 'bg-blue-50 border-blue-200 text-blue-800',
    success: 'bg-green-50 border-green-200 text-green-800',
    warning: 'bg-yellow-50 border-yellow-200 text-yellow-800',
    error: 'bg-red-50 border-red-200 text-red-800'
  };

  return (
    <div className={`border rounded-lg p-4 ${variantClasses[variant]} ${className}`}>
      <div className="flex justify-between items-start">
        <div className="flex-1">{children}</div>
        {onClose && (
          <button
            onClick={onClose}
            className="ml-4 text-current opacity-70 hover:opacity-100 transition-opacity"
          >
            ✕
          </button>
        )}
      </div>
    </div>
  );
}

interface ProgressBarProps {
  progress: number;
  color?: 'blue' | 'purple' | 'green' | 'red';
  size?: 'sm' | 'md' | 'lg';
  showPercentage?: boolean;
  className?: string;
}

export function ProgressBar({
  progress,
  color = 'blue',
  size = 'md',
  showPercentage = true,
  className = ''
}: ProgressBarProps) {
  const colorClasses = {
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    green: 'bg-green-500',
    red: 'bg-red-500'
  };

  const sizeClasses = {
    sm: 'h-2',
    md: 'h-3',
    lg: 'h-4'
  };

  const clampedProgress = Math.min(100, Math.max(0, progress));

  return (
    <div className={className}>
      {showPercentage && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-gray-700">Progress</span>
          <span className="text-sm font-medium text-gray-700">{clampedProgress.toFixed(0)}%</span>
        </div>
      )}
      <div className={`w-full bg-gray-200 rounded-full overflow-hidden ${sizeClasses[size]}`}>
        <div
          className={`${colorClasses[color]} rounded-full transition-all duration-300 ease-out ${sizeClasses[size]}`}
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
    </div>
  );
}

// ----------------------
// Modal
// ----------------------
import { useEffect } from 'react';
import { createPortal } from 'react-dom';

interface ModalProps {
  open: boolean;
  onClose?: () => void;
  title?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  closeOnBackdrop?: boolean;
}

export function Modal({ open, onClose, title, children, footer, size = 'md', closeOnBackdrop = true }: ModalProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && onClose) onClose();
    };
    if (open) document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;

  const sizes = {
    sm: 'max-w-md',
    md: 'max-w-lg',
    lg: 'max-w-2xl',
    xl: 'max-w-4xl',
  };

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-center justify-center" role="dialog" aria-modal="true" aria-labelledby={title ? 'modal-title' : undefined}>
      <div
        className="absolute inset-0 bg-black/40"
        onClick={() => closeOnBackdrop && onClose?.()}
      />
      <div className={`relative w-full ${sizes[size]} mx-4`}> 
        <div className="bg-white rounded-xl shadow-2xl border border-gray-100 overflow-hidden">
          {(title || onClose) && (
            <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100 bg-gray-50">
              {title && <h3 id="modal-title" className="text-sm font-semibold text-gray-900">{title}</h3>}
              {onClose && (
                <button onClick={onClose} aria-label="Close" className="text-gray-500 hover:text-gray-800">✕</button>
              )}
            </div>
          )}
          <div className="p-6">{children}</div>
          {footer && (
            <div className="px-5 py-4 border-t border-gray-100 bg-gray-50">{footer}</div>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
}

// ----------------------
// Tabs
// ----------------------
interface TabItem {
  id: string;
  label: string;
  content: React.ReactNode;
  icon?: React.ReactNode;
}

interface TabsProps {
  tabs: TabItem[];
  activeId?: string;
  onChange?: (id: string) => void;
  variant?: 'underline' | 'pill';
}

export function Tabs({ tabs, activeId, onChange, variant = 'pill' }: TabsProps) {
  const [active, setActive] = React.useState(activeId || tabs[0]?.id);
  React.useEffect(() => { if (activeId) setActive(activeId); }, [activeId]);

  const set = (id: string) => {
    setActive(id);
    onChange?.(id);
  };

  return (
    <div>
      <div className="flex flex-wrap gap-2 mb-4">
        {tabs.map((t) => (
          <button
            key={t.id}
            className={
              variant === 'pill'
                ? `px-4 py-2 rounded-lg text-sm font-medium border ${active === t.id ? 'bg-blue-50 text-blue-700 border-blue-500' : 'bg-white text-gray-700 border-gray-300 hover:border-gray-400'}`
                : `px-3 py-2 text-sm font-medium border-b-2 ${active === t.id ? 'border-blue-600 text-gray-900' : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'}`
            }
            onClick={() => set(t.id)}
          >
            <span className="inline-flex items-center">{t.icon}<span className={t.icon ? 'ml-2' : ''}>{t.label}</span></span>
          </button>
        ))}
      </div>
      <div className="rounded-xl border border-gray-100 bg-white p-4 shadow-sm">
        {tabs.map((t) => (
          <div key={t.id} style={{ display: active === t.id ? 'block' : 'none' }}>{t.content}</div>
        ))}
      </div>
    </div>
  );
}

// ----------------------
// Tooltip
// ----------------------
interface TooltipProps {
  content: React.ReactNode;
  children: React.ReactNode;
  placement?: 'top' | 'right' | 'bottom' | 'left';
}

export function Tooltip({ content, children, placement = 'top' }: TooltipProps) {
  const [open, setOpen] = React.useState(false);
  const pos = {
    top: 'bottom-full left-1/2 -translate-x-1/2 -translate-y-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 translate-y-2',
    left: 'right-full top-1/2 -translate-y-1/2 -translate-x-2',
    right: 'left-full top-1/2 -translate-y-1/2 translate-x-2',
  }[placement];

  return (
    <span className="relative inline-block" onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      {children}
      {open && (
        <span className={`absolute px-2 py-1 text-xs rounded bg-gray-900 text-white shadow ${pos}`}>
          {content}
        </span>
      )}
    </span>
  );
}

// ----------------------
// Accordion
// ----------------------
interface AccordionItem { id: string; title: string; content: React.ReactNode; }
interface AccordionProps { items: AccordionItem[]; defaultOpenId?: string; }

export function Accordion({ items, defaultOpenId }: AccordionProps) {
  const [openId, setOpenId] = React.useState(defaultOpenId || items[0]?.id);
  return (
    <div className="space-y-3">
      {items.map((it) => (
        <div key={it.id} className="border border-gray-200 rounded-xl overflow-hidden bg-white">
          <button
            className="w-full text-left px-4 py-3 font-medium flex justify-between items-center hover:bg-gray-50"
            onClick={() => setOpenId((prev) => (prev === it.id ? '' : it.id))}
          >
            {it.title}
            <span className="text-gray-400">{openId === it.id ? '−' : '+'}</span>
          </button>
          {openId === it.id && (
            <div className="px-4 py-3 border-t border-gray-100 text-sm text-gray-700">{it.content}</div>
          )}
        </div>
      ))}
    </div>
  );
}

// ----------------------
// Skeleton
// ----------------------
interface SkeletonProps { width?: string | number; height?: string | number; circle?: boolean; className?: string; }
export function Skeleton({ width = '100%', height = 16, circle = false, className = '' }: SkeletonProps) {
  const style: React.CSSProperties = { width, height, borderRadius: circle ? '9999px' : '8px' };
  return <div className={`bg-gray-200 relative overflow-hidden ${className}`} style={style}>
    <div className="absolute inset-0 -translate-x-full animate-[shimmer_1.2s_infinite] bg-gradient-to-r from-transparent via-white/60 to-transparent" />
  </div>;
}

// ----------------------
// Stepper
// ----------------------
interface StepperStep { id: string; label: string; }
interface StepperProps { steps: StepperStep[]; current: number; onStepClick?: (index: number) => void; }
export function Stepper({ steps, current, onStepClick }: StepperProps) {
  return (
    <div className="flex items-center">
      {steps.map((s, idx) => {
        const done = idx < current;
        const active = idx === current;
        return (
          <div key={s.id} className="flex items-center">
            <button
              className={`w-8 h-8 rounded-full border flex items-center justify-center text-sm font-semibold ${
                done ? 'bg-green-500 text-white border-green-500' : active ? 'bg-blue-600 text-white border-blue-600' : 'bg-white text-gray-700 border-gray-300'
              }`}
              onClick={() => onStepClick?.(idx)}
              aria-current={active}
            >
              {done ? '✓' : idx + 1}
            </button>
            {idx !== steps.length - 1 && (
              <div className={`w-12 h-0.5 mx-2 ${done ? 'bg-green-500' : 'bg-gray-300'}`} />
            )}
          </div>
        );
      })}
    </div>
  );
}