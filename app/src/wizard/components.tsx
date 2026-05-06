/** Wizard-specific UI primitives. Lifted from mock-ups/wizard-interface/Nell Wizard.html. */

import type { ReactNode } from "react";

export function WizardShell({
  step,
  totalSteps,
  title,
  subtitle,
  children,
  footer,
  avatar,
}: {
  step: number;
  totalSteps: number;
  title: string;
  subtitle?: string;
  children: ReactNode;
  footer?: ReactNode;
  avatar: ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "100vw",
        height: "100vh",
        gap: 48,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
        {avatar}
        <div style={{ width: 200 }}>
          <ProgressBar current={step} total={totalSteps} />
          <div
            style={{
              textAlign: "center",
              marginTop: 6,
              fontSize: 10,
              color: "var(--mauve)",
              fontFamily: "var(--font-disp)",
              letterSpacing: "0.06em",
            }}
          >
            Step {step} of {totalSteps}
          </div>
        </div>
      </div>
      <div
        style={{
          width: 380,
          background: "rgba(234,222,218,0.97)",
          border: "1px solid var(--border)",
          borderRadius: 12,
          boxShadow:
            "0 4px 32px rgba(42,31,31,0.28), 0 0 0 1px rgba(130,51,41,0.07)",
          overflow: "hidden",
          color: "var(--text)",
          animation: "msg-in 0.22s ease",
        }}
      >
        <div
          style={{
            padding: "16px 18px 14px",
            borderBottom: "1px solid var(--border)",
            background: "rgba(234,222,218,0.5)",
          }}
        >
          <div
            style={{
              fontSize: 9.5,
              color: "var(--text-mute)",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              fontFamily: "var(--font-disp)",
              marginBottom: 4,
            }}
          >
            Companion Emergence
          </div>
          <div style={{ fontSize: 15, fontWeight: 500, color: "var(--text)", lineHeight: 1.3 }}>
            {title}
          </div>
          {subtitle && (
            <div style={{ fontSize: 11, color: "var(--text-mid)", marginTop: 3, lineHeight: 1.5 }}>
              {subtitle}
            </div>
          )}
        </div>
        <div style={{ padding: "16px 18px", maxHeight: 460, overflowY: "auto" }}>{children}</div>
        {footer && (
          <div
            style={{
              padding: "12px 18px",
              borderTop: "1px solid var(--border)",
              background: "rgba(234,222,218,0.4)",
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
  const pct = (current / total) * 100;
  return (
    <div
      style={{
        height: 3,
        background: "rgba(191,184,173,0.3)",
        borderRadius: 2,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          background: "var(--accent)",
          transition: "width 0.4s ease",
        }}
      />
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
        background: isPrimary ? "var(--accent)" : "transparent",
        color: isPrimary ? "var(--linen)" : "var(--text-mid)",
        padding: small ? "5px 11px" : "8px 16px",
        borderRadius: 6,
        fontSize: small ? 11 : 12,
        fontWeight: 500,
        border: isPrimary ? "1px solid var(--accent)" : "1px solid var(--border)",
        opacity: disabled ? 0.4 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "opacity 0.18s, transform 0.18s",
      }}
    >
      {children}
    </button>
  );
}

export function WInput({
  value,
  onChange,
  placeholder,
  mono,
  error,
  maxLength,
  onKeyDown,
}: {
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  mono?: boolean;
  error?: boolean;
  maxLength?: number;
  onKeyDown?: (e: React.KeyboardEvent<HTMLInputElement>) => void;
}) {
  return (
    <input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={onKeyDown}
      placeholder={placeholder}
      maxLength={maxLength}
      style={{
        width: "100%",
        padding: "9px 12px",
        background: "rgba(255,255,255,0.6)",
        border: `1px solid ${error ? "var(--crimson)" : "var(--border)"}`,
        borderRadius: 6,
        fontSize: 12,
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
        color: "var(--text-mute)",
        fontFamily: "var(--font-disp)",
        textTransform: "uppercase",
        letterSpacing: "0.08em",
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
        color: "var(--text-mute)",
        fontFamily: "var(--font-disp)",
        textTransform: "uppercase",
        letterSpacing: "0.08em",
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
        background: "var(--border)",
        opacity: 0.5,
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
      role="button"
      tabIndex={0}
      onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && onClick()}
      style={{
        padding: "11px 14px",
        background: selected ? "rgba(130,51,41,0.08)" : "rgba(255,255,255,0.4)",
        border: `1px solid ${selected ? "var(--accent)" : "var(--border)"}`,
        borderRadius: 7,
        cursor: "pointer",
        marginBottom: 8,
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
          style={{
            width: 12,
            height: 12,
            borderRadius: "50%",
            border: `1.5px solid ${selected ? "var(--accent)" : "var(--text-mute)"}`,
            background: selected ? "var(--accent)" : "transparent",
            transition: "all 0.18s",
            flexShrink: 0,
          }}
        />
        <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text)", flex: 1 }}>
          {title}
        </div>
        {badge && (
          <div
            style={{
              fontSize: 9,
              color: "var(--accent)",
              background: "rgba(130,51,41,0.12)",
              padding: "2px 7px",
              borderRadius: 9,
              fontFamily: "var(--font-disp)",
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}
          >
            {badge}
          </div>
        )}
      </div>
      <div
        style={{
          fontSize: 11,
          color: "var(--text-mid)",
          lineHeight: 1.5,
          paddingLeft: 20,
        }}
      >
        {description}
      </div>
    </div>
  );
}
