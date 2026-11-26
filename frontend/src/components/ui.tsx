'use client';

import { forwardRef } from 'react';
import { Loader2 } from 'lucide-react';

export function cn(...classes: (string | false | null | undefined)[]): string {
  return classes.filter(Boolean).join(' ');
}

/* -------------------------------------------------------------------------- */
/* Surfaces                                                                    */
/* -------------------------------------------------------------------------- */

export function Card({
  children,
  className,
  hover = false,
}: {
  children: React.ReactNode;
  className?: string;
  hover?: boolean;
}) {
  return (
    <div
      className={cn(
        'rounded-2xl bg-white border border-ink-950/[0.06] shadow-[var(--shadow-card)]',
        hover &&
          'transition-all duration-300 hover:shadow-[var(--shadow-card-hover)] hover:-translate-y-px',
        className,
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  action,
  className,
}: {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn('flex items-start justify-between gap-4 px-6 pt-5 pb-4', className)}>
      <div className="min-w-0">
        <h3 className="text-[15px] font-semibold text-ink-950 leading-tight">{title}</h3>
        {subtitle && <p className="text-[13px] text-ink-500 mt-0.5">{subtitle}</p>}
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Buttons                                                                     */
/* -------------------------------------------------------------------------- */

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';
type ButtonSize = 'sm' | 'md';

const buttonVariants: Record<ButtonVariant, string> = {
  primary:
    'bg-primary-600 text-white hover:bg-primary-700 active:bg-primary-700 shadow-[0_1px_2px_rgba(79,70,229,0.35)]',
  secondary:
    'bg-white text-ink-800 border border-ink-950/10 hover:bg-ink-50 active:bg-ink-100',
  ghost: 'text-ink-600 hover:bg-ink-100 hover:text-ink-900',
  danger:
    'bg-negative-600 text-white hover:bg-negative-600/90 shadow-[0_1px_2px_rgba(220,38,38,0.3)]',
};

const buttonSizes: Record<ButtonSize, string> = {
  sm: 'h-8 px-3 text-[13px] gap-1.5 rounded-lg',
  md: 'h-10 px-4 text-sm gap-2 rounded-xl',
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = 'primary', size = 'md', loading, className, children, disabled, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      disabled={disabled || loading}
      className={cn(
        'focus-ring inline-flex items-center justify-center font-medium transition-all duration-150 select-none',
        'disabled:opacity-50 disabled:pointer-events-none',
        buttonVariants[variant],
        buttonSizes[size],
        className,
      )}
      {...props}
    >
      {loading && <Loader2 className="w-4 h-4 animate-spin" />}
      {children}
    </button>
  );
});

/* -------------------------------------------------------------------------- */
/* Badges                                                                      */
/* -------------------------------------------------------------------------- */

type BadgeTone = 'neutral' | 'positive' | 'negative' | 'caution' | 'primary' | 'ink';

const badgeTones: Record<BadgeTone, string> = {
  neutral: 'bg-ink-100 text-ink-700',
  positive: 'bg-positive-50 text-positive-600',
  negative: 'bg-negative-50 text-negative-600',
  caution: 'bg-caution-50 text-caution-600',
  primary: 'bg-primary-50 text-primary-600',
  ink: 'bg-ink-900 text-white',
};

export function Badge({
  children,
  tone = 'neutral',
  className,
}: {
  children: React.ReactNode;
  tone?: BadgeTone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-[11.5px] font-semibold tracking-wide',
        badgeTones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* -------------------------------------------------------------------------- */
/* Form controls                                                               */
/* -------------------------------------------------------------------------- */

export const inputClass =
  'focus-ring w-full h-10 rounded-xl border border-ink-950/10 bg-white px-3.5 text-sm text-ink-900 placeholder:text-ink-400 transition-colors hover:border-ink-950/20 focus:border-primary-500';

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  function Input({ className, ...props }, ref) {
    return <input ref={ref} className={cn(inputClass, className)} {...props} />;
  },
);

export const Select = forwardRef<
  HTMLSelectElement,
  React.SelectHTMLAttributes<HTMLSelectElement>
>(function Select({ className, children, ...props }, ref) {
  return (
    <select
      ref={ref}
      className={cn(inputClass, 'pr-8 appearance-none bg-[url("data:image/svg+xml;charset=utf-8,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%2716%27 height=%2716%27 viewBox=%270 0 24 24%27 fill=%27none%27 stroke=%27%235C6B8A%27 stroke-width=%272%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27%3E%3Cpath d=%27m6 9 6 6 6-6%27/%3E%3C/svg%3E")] bg-[position:right_12px_center] bg-no-repeat', className)}
      {...props}
    >
      {children}
    </select>
  );
});

export function Toggle({
  checked,
  onChange,
  label,
  description,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  description?: string;
}) {
  return (
    <label className="flex items-center justify-between gap-4 cursor-pointer select-none">
      <span>
        <span className="text-sm font-medium text-ink-900">{label}</span>
        {description && <span className="block text-[12.5px] text-ink-500">{description}</span>}
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        onClick={() => onChange(!checked)}
        className={cn(
          'focus-ring relative w-11 h-6 rounded-full transition-colors shrink-0',
          checked ? 'bg-primary-600' : 'bg-ink-200',
        )}
      >
        <span
          className={cn(
            'absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white shadow-sm transition-transform',
            checked && 'translate-x-5',
          )}
        />
      </button>
    </label>
  );
}

/* -------------------------------------------------------------------------- */
/* Tabs                                                                        */
/* -------------------------------------------------------------------------- */

export function Tabs({
  tabs,
  active,
  onChange,
  className,
}: {
  tabs: { id: string; label: string; icon?: React.ReactNode }[];
  active: string;
  onChange: (id: string) => void;
  className?: string;
}) {
  return (
    <div className={cn('flex items-center gap-1 p-1 bg-ink-100/70 rounded-xl w-fit', className)}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={cn(
            'focus-ring flex items-center gap-1.5 px-3.5 h-8 rounded-lg text-[13px] font-medium transition-all',
            active === tab.id
              ? 'bg-white text-ink-950 shadow-sm'
              : 'text-ink-500 hover:text-ink-800',
          )}
        >
          {tab.icon}
          {tab.label}
        </button>
      ))}
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Modal                                                                       */
/* -------------------------------------------------------------------------- */

export function Modal({
  open,
  onClose,
  title,
  children,
  footer,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
}) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 animate-fade-in"
      role="dialog"
      aria-modal="true"
    >
      <button
        aria-label="Close"
        className="absolute inset-0 bg-ink-950/40 backdrop-blur-[2px]"
        onClick={onClose}
      />
      <div className="relative w-full max-w-md rounded-2xl bg-white shadow-[var(--shadow-pop)] animate-fade-up p-6">
        <h3 className="text-lg font-semibold text-ink-950">{title}</h3>
        <div className="mt-3 text-sm text-ink-600 leading-relaxed">{children}</div>
        {footer && <div className="mt-6 flex justify-end gap-2">{footer}</div>}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Skeletons & loaders                                                         */
/* -------------------------------------------------------------------------- */

export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('skeleton-shimmer rounded-lg', className)} />;
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn('w-5 h-5 text-ink-400 animate-spin', className)} />;
}

/* -------------------------------------------------------------------------- */
/* Stat tile                                                                   */
/* -------------------------------------------------------------------------- */

export function Stat({
  label,
  value,
  sub,
  tone = 'neutral',
  icon,
  className,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: 'neutral' | 'positive' | 'negative' | 'primary';
  icon?: React.ReactNode;
  className?: string;
}) {
  const toneText =
    tone === 'positive'
      ? 'text-positive-600'
      : tone === 'negative'
        ? 'text-negative-600'
        : tone === 'primary'
          ? 'text-primary-600'
          : 'text-ink-950';
  return (
    <Card hover className={cn('p-5', className)}>
      <div className="flex items-center justify-between">
        <span className="text-[12.5px] font-medium text-ink-500 tracking-wide uppercase">
          {label}
        </span>
        {icon && (
          <span className="w-8 h-8 rounded-lg bg-ink-50 text-ink-500 flex items-center justify-center">
            {icon}
          </span>
        )}
      </div>
      <div className={cn('tabular mt-2 text-[26px] font-semibold tracking-tight', toneText)}>
        {value}
      </div>
      {sub && <div className="text-[12.5px] text-ink-500 mt-1">{sub}</div>}
    </Card>
  );
}
