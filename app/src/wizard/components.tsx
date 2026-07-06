/** Wizard-specific UI primitives. Lifted from mock-ups/wizard-interface/Nell Wizard.html. */

import { useId, type ReactNode } from "react";

export function WizardShell({
  step,
  totalSteps,
  title,
  subtitle,
  children,
  footer,
  avatar,
}: {
  step?: number;
  totalSteps?: number;
  title: string;
  subtitle?: string;
  children: ReactNode;
  footer?: ReactNode;
  avatar?: ReactNode;
}) {
  const showLeftPanel = avatar != null || (step != null && totalSteps != null);
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "100%",
        height: "100%",
        gap: 56,
      }}
    >
      {showLeftPanel && (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
          {avatar}
          {step != null && totalSteps != null && (
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
              <ProgressBar current={step} total={totalSteps} />
              <div
                style={{
                  textAlign: "center",
                  fontSize: 11,
                  fontWeight: 500,
                  color: "var(--text-mute)",
                  textTransform: "uppercase",
                  letterSpacing: "0.09em",
                }}
              >
                Step {step} of {totalSteps}
              </div>
            </div>
          )}
        </div>
      )}
      <div
        style={{
          width: 432,
          background: "var(--panel-strong)",
          backdropFilter: "blur(36px) saturate(1.5)",
          WebkitBackdropFilter: "blur(36px) saturate(1.5)",
          border: "1px solid var(--hairline)",
          borderRadius: 26,
          boxShadow: "var(--shadow)",
          overflow: "hidden",
          color: "var(--text)",
          animation: "msg-in 0.28s ease",
        }}
      >
        <div
          style={{
            padding: "22px 26px 16px",
            borderBottom: "1px solid var(--hairline)",
          }}
        >
          <div
            style={{
              fontSize: 10,
              color: "var(--text-mute)",
              textTransform: "uppercase",
              letterSpacing: "0.18em",
              fontWeight: 600,
              marginBottom: 4,
            }}
          >
            Companion Emergence
          </div>
          <div
            style={{
              fontSize: 21,
              fontWeight: 700,
              color: "var(--text)",
              lineHeight: 1.25,
              letterSpacing: "-0.01em",
            }}
          >
            {title}
          </div>
          {subtitle && (
            <div style={{ fontSize: 12.5, color: "var(--text-mid)", marginTop: 3, lineHeight: 1.55 }}>
              {subtitle}
            </div>
          )}
        </div>
        <div style={{ padding: "18px 26px 20px", maxHeight: 430, overflowY: "auto" }}>{children}</div>
        {footer && (
          <div
            style={{
              padding: "15px 26px",
              borderTop: "1px solid var(--hairline)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 8,
            }}
          >
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}

export function ProgressBar({ current, total }: { current: number; total: number }) {
  return (
    <div
      role="progressbar"
      aria-label="Wizard progress"
      aria-valuemin={1}
      aria-valuemax={total}
      aria-valuenow={current}
      style={{
        display: "flex",
        flexDirection: "row",
        gap: 5,
      }}
    >
      {Array.from({ length: total }, (_, i) => (
        <div
          key={i}
          style={{
            width: 24,
            height: 5,
            borderRadius: 3,
            background: i < current ? "var(--accent)" : "rgba(255,255,255,0.14)",
            transition: "background 0.35s",
          }}
        />
      ))}
    </div>
  );
}

export function WButton({
  children,
  onClick,
  disabled,
  variant = "primary",
  small,
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  variant?: "primary" | "ghost";
  small?: boolean;
}) {
  const isPrimary = variant === "primary";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        background: isPrimary
          ? disabled
            ? "color-mix(in srgb, var(--accent) 35%, transparent)"
            : "var(--accent)"
          : "var(--field)",
        color: isPrimary ? (disabled ? "rgba(255,255,255,0.5)" : "var(--linen)") : "var(--text-mid)",
        padding: small ? "9px 16px" : "10px 22px",
        borderRadius: 999,
        fontSize: small ? 13 : 13.5,
        fontWeight: isPrimary ? 700 : 600,
        border: isPrimary ? "none" : "1px solid var(--hairline)",
        boxShadow:
          isPrimary && !disabled
            ? "0 8px 22px color-mix(in srgb, var(--accent) 42%, transparent)"
            : "none",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "opacity 0.18s, transform 0.18s",
      }}
    >
      {children}
    </button>
  );
}

export function WInput({
  id,
  label,
  value,
  onChange,
  placeholder,
  mono,
  error,
  maxLength,
  onKeyDown,
}: {
  id?: string;
  label?: string;
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  mono?: boolean;
  error?: boolean;
  maxLength?: number;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
}) {
  const generatedId = useId();
  const inputId = id ?? generatedId;
  return (
    <input
      id={inputId}
      aria-label={label ?? placeholder}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      maxLength={maxLength}
      style={{
        width: "100%",
        padding: "12px 16px",
        background: "var(--field)",
        border: `1px solid ${error ? "#e07a6a" : "var(--hairline)"}`,
        borderRadius: 14,
        fontSize: 14,
        fontFamily: mono ? "DM Mono, Courier New, monospace" : "var(--font-ui)",
        color: "var(--text)",
        outline: "none",
        transition: "border 0.18s",
      }}
    />
  );
}

export function FieldLabel({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10.5,
        fontWeight: 600,
        color: "var(--text-mute)",
        textTransform: "uppercase",
        letterSpacing: "0.12em",
        marginBottom: 5,
      }}
    >
      {children}
    </div>
  );
}

export function FieldError({ msg }: { msg: string | null }) {
  if (!msg) return null;
  return (
    <div
      style={{
        marginTop: 6,
        fontSize: 11,
        color: "var(--crimson)",
        fontFamily: "var(--font-ui)",
      }}
    >
      {msg}
    </div>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div
      style={{
        fontSize: 10.5,
        fontWeight: 600,
        color: "var(--text-mute)",
        textTransform: "uppercase",
        letterSpacing: "0.12em",
        marginBottom: 8,
        marginTop: 4,
      }}
    >
      {children}
    </div>
  );
}

export function Divider() {
  return (
    <div
      style={{
        height: 1,
        background: "var(--hairline-soft)",
        margin: "16px 0 12px",
      }}
    />
  );
}

export function OptionCard({
  selected,
  onClick,
  title,
  description,
  badge,
}: {
  selected: boolean;
  onClick: () => void;
  title: string;
  description: string;
  badge?: string;
}) {
  return (
    <div
      onClick={onClick}
      role="radio"
      aria-checked={selected}
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onClick();
        }
      }}
      style={{
        padding: "14px 16px",
        background: selected
          ? "color-mix(in srgb, var(--accent) 13%, transparent)"
          : "var(--field)",
        border: `1px solid ${
          selected ? "color-mix(in srgb, var(--accent) 50%, transparent)" : "var(--hairline)"
        }`,
        borderRadius: 16,
        cursor: "pointer",
        marginBottom: 10,
        transition: "background 0.18s, border 0.18s",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 4,
        }}
      >
        <div
          aria-hidden="true"
          style={{
            width: 15,
            height: 15,
            borderRadius: "50%",
            border: `1.5px solid ${selected ? "var(--accent)" : "var(--text-mute)"}`,
            background: selected ? "var(--accent)" : "transparent",
            boxShadow: selected ? "inset 0 0 0 3px var(--panel-strong)" : "none",
            transition: "all 0.18s",
            flexShrink: 0,
          }}
        />
        <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text)", flex: 1 }}>
          {title}
        </div>
        {badge && (
          <div
            style={{
              fontSize: 9.5,
              fontWeight: 700,
              color: "var(--accent-text)",
              background: "color-mix(in srgb, var(--accent) 22%, transparent)",
              padding: "2px 7px",
              borderRadius: 999,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
            }}
          >
            {badge}
          </div>
        )}
      </div>
      <div
        style={{
          fontSize: 12,
          color: "var(--text-mid)",
          lineHeight: 1.5,
          paddingLeft: 25,
        }}
      >
        {description}
      </div>
    </div>
  );
}
